# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Benchmark / profile generation latency for STEM and base models.

Measures prefill and autoregressive decode independently using CUDA events,
reporting mean / std / min / max over multiple trials.

Supports both single-GPU and distributed (multi-GPU) modes.  In distributed
mode the script additionally instruments the STEM communication primitives
(all-gather, all-to-all) so you can see exactly how much time is spent in
collective operations vs. local compute.

Usage examples
--------------
# Single-GPU — STEM only:
    python -m apps.main.benchmark_generation \
        stem_ckpt=<path> --context_len 512 --gen_len 128

# Single-GPU — both models side-by-side:
    python -m apps.main.benchmark_generation \
        stem_ckpt=<path> base_ckpt=<path>

# Distributed (4 GPUs) — profile STEM communication overhead:
    torchrun --nproc-per-node=4 -m apps.main.benchmark_generation \
        stem_ckpt=<path> --stem_parallel_size 4 \
        --context_len 512 --gen_len 128

# Distributed — compare different parallelism degrees:
    for N in 1 2 4 8; do
        torchrun --nproc-per-node=$N -m apps.main.benchmark_generation \
            stem_ckpt=<path> --stem_parallel_size $N
    done
"""

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from apps.main.generate import (
    KVCache,
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer as load_base_model,
    pack_prompts,
    sample_tokens,
)
from apps.main.stem_generate import (
    load_consolidated_model_and_tokenizer as load_stem_model,
)
from lingua.checkpoint import CONSOLIDATE_FOLDER, CONSOLIDATE_NAME, consolidate_checkpoints
from lingua.stem_checkpoint import CONSOLIDATE_STEM_NAME
from lingua.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    prefill_ms: List[float] = field(default_factory=list)
    decode_ms: List[float] = field(default_factory=list)
    total_ms: List[float] = field(default_factory=list)
    tokens_generated: int = 0

    # Communication profiling (only filled in distributed mode)
    prefill_comm_ms: List[float] = field(default_factory=list)
    decode_comm_ms: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Communication profiling instrumentation
# ---------------------------------------------------------------------------

class CommProfiler:
    """Wraps the STEM communication primitives to measure GPU time spent in
    collective operations (all-gather, all-to-all, reduce-scatter)."""

    def __init__(self):
        self.enabled = False
        self._events: List[tuple] = []  # [(start, end), ...]
        self._originals = {}

    def start(self):
        self.enabled = True
        self._events.clear()

    def stop_and_collect(self) -> float:
        """Synchronize, compute total comm time in ms, and reset."""
        self.enabled = False
        torch.cuda.synchronize()
        total_ms = 0.0
        for start_ev, end_ev in self._events:
            total_ms += start_ev.elapsed_time(end_ev)
        self._events.clear()
        return total_ms

    def _make_wrapper(self, orig_fn):
        profiler = self

        def wrapped(*args, **kwargs):
            if not profiler.enabled:
                return orig_fn(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = orig_fn(*args, **kwargs)
            end.record()
            profiler._events.append((start, end))
            return result

        return wrapped

    def install(self):
        """Monkey-patch the low-level STEM communication functions."""
        import lingua.stem_dist_utils as sdu

        targets = [
            "_gather_along_first_dim_stem",
            "_gather_along_last_dim_stem",
            "_split_along_first_dim_stem",
            "_split_along_last_dim_stem",
            "_reduce_scatter_along_first_dim_stem",
            "_all_to_all_stem",
        ]
        for name in targets:
            orig = getattr(sdu, name)
            self._originals[name] = orig
            setattr(sdu, name, self._make_wrapper(orig))

    def uninstall(self):
        """Restore original functions."""
        import lingua.stem_dist_utils as sdu

        for name, orig in self._originals.items():
            setattr(sdu, name, orig)
        self._originals.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(values: List[float]) -> Dict[str, float]:
    t = torch.tensor(values)
    return {
        "mean": t.mean().item(),
        "std": t.std().item() if len(t) > 1 else 0.0,
        "min": t.min().item(),
        "max": t.max().item(),
    }


def _print_results(label: str, result: BenchmarkResult, gen_len: int):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    n = len(result.prefill_ms)
    print(f"  Trials: {n}")
    print(f"  Tokens generated per trial: {gen_len}")

    pf = _stats(result.prefill_ms)
    print(f"\n  Prefill latency (ms):")
    print(f"    mean={pf['mean']:.2f}  std={pf['std']:.2f}  "
          f"min={pf['min']:.2f}  max={pf['max']:.2f}")

    dc = _stats(result.decode_ms)
    print(f"  Decode latency (ms) [{gen_len} tokens]:")
    print(f"    mean={dc['mean']:.2f}  std={dc['std']:.2f}  "
          f"min={dc['min']:.2f}  max={dc['max']:.2f}")

    per_tok = _stats([d / gen_len for d in result.decode_ms])
    print(f"  Per-token decode latency (ms):")
    print(f"    mean={per_tok['mean']:.2f}  std={per_tok['std']:.2f}  "
          f"min={per_tok['min']:.2f}  max={per_tok['max']:.2f}")

    tot = _stats(result.total_ms)
    print(f"  Total latency (prefill + decode) (ms):")
    print(f"    mean={tot['mean']:.2f}  std={tot['std']:.2f}  "
          f"min={tot['min']:.2f}  max={tot['max']:.2f}")

    tps = _stats([gen_len / (t / 1000.0) for t in result.decode_ms])
    print(f"  Decode throughput (tokens/s):")
    print(f"    mean={tps['mean']:.1f}  std={tps['std']:.1f}  "
          f"min={tps['min']:.1f}  max={tps['max']:.1f}")

    tps_total = _stats([gen_len / (t / 1000.0) for t in result.total_ms])
    print(f"  End-to-end throughput (tokens/s):")
    print(f"    mean={tps_total['mean']:.1f}  std={tps_total['std']:.1f}  "
          f"min={tps_total['min']:.1f}  max={tps_total['max']:.1f}")

    # Communication breakdown (if available)
    if result.prefill_comm_ms:
        pf_comm = _stats(result.prefill_comm_ms)
        pf_comp = _stats([p - c for p, c in zip(result.prefill_ms, result.prefill_comm_ms)])
        pf_pct = _stats([c / p * 100 for c, p in zip(result.prefill_comm_ms, result.prefill_ms)])
        print(f"\n  --- Communication breakdown (prefill) ---")
        print(f"  Comm (ms):    mean={pf_comm['mean']:.2f}  std={pf_comm['std']:.2f}")
        print(f"  Compute (ms): mean={pf_comp['mean']:.2f}  std={pf_comp['std']:.2f}")
        print(f"  Comm %:       mean={pf_pct['mean']:.1f}%")

    if result.decode_comm_ms:
        dc_comm = _stats(result.decode_comm_ms)
        dc_comp = _stats([d - c for d, c in zip(result.decode_ms, result.decode_comm_ms)])
        dc_pct = _stats([c / d * 100 for c, d in zip(result.decode_comm_ms, result.decode_ms)])
        dc_per_tok_comm = _stats([c / gen_len for c in result.decode_comm_ms])
        print(f"\n  --- Communication breakdown (decode) ---")
        print(f"  Comm (ms):            mean={dc_comm['mean']:.2f}  std={dc_comm['std']:.2f}")
        print(f"  Compute (ms):         mean={dc_comp['mean']:.2f}  std={dc_comp['std']:.2f}")
        print(f"  Comm %:               mean={dc_pct['mean']:.1f}%")
        print(f"  Comm per-token (ms):  mean={dc_per_tok_comm['mean']:.3f}")

    print()


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def _setup_distributed(stem_parallel_size: int):
    """Initialize torch.distributed and STEM process groups.

    Follows the same pattern as stem_eval.py / eval.py.
    """
    from lingua.distributed import DistributedArgs, setup_torch_distributed
    from lingua.stem_dist_utils import initialize_stem_process_group, is_stem_initialized

    if torch.distributed.is_initialized():
        pass
    elif torch.cuda.device_count() > 1:
        setup_torch_distributed(DistributedArgs())

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if stem_parallel_size > 1 and not is_stem_initialized():
        initialize_stem_process_group(stem_parallel_size)
        if dist.get_rank() == 0:
            logger.info(f"Initialized STEM process groups with parallel size: {stem_parallel_size}")


def _is_main_rank() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def benchmark_generation(
    model: nn.Module,
    tokenizer: Tokenizer,
    context_len: int,
    gen_len: int,
    n_trials: int,
    warmup: int,
    comm_profiler: Optional[CommProfiler] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> BenchmarkResult:
    """Run *n_trials* of prefill+decode and collect CUDA-event timings.

    When *comm_profiler* is provided, additionally measures time spent inside
    STEM collective operations (all-gather, all-to-all).
    """

    max_tokens = context_len + gen_len + 64
    gen_cfg = PackedCausalTransformerGeneratorArgs(
        temperature=0.0,
        max_gen_len=gen_len,
        max_tokens=max_tokens,
        dtype="bf16" if dtype == torch.bfloat16 else "fp32",
        device=device,
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    vocab_size = tokenizer.n_words
    prompt_ids = torch.randint(1, vocab_size, (context_len,)).tolist()

    result = BenchmarkResult(tokens_generated=gen_len)

    total_iters = warmup + n_trials
    for trial_idx in range(total_iters):
        is_warmup = trial_idx < warmup

        packed, lengths = pack_prompts([prompt_ids])
        packed, lengths = packed.cuda(), lengths.cuda()

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # ---- Prefill ----
        if comm_profiler:
            comm_profiler.start()

        start_pf = torch.cuda.Event(enable_timing=True)
        end_pf = torch.cuda.Event(enable_timing=True)

        start_pf.record()
        prompt_logits = generator.prefill(packed.unsqueeze(0), lengths)
        end_pf.record()

        torch.cuda.synchronize()
        prefill_ms = start_pf.elapsed_time(end_pf)
        prefill_comm_ms = comm_profiler.stop_and_collect() if comm_profiler else 0.0

        # ---- Decode ----
        all_tokens = sample_tokens(prompt_logits, gen_cfg.temperature)
        current_token = all_tokens[:, lengths.cumsum(0) - 1]

        if comm_profiler:
            comm_profiler.start()

        start_dc = torch.cuda.Event(enable_timing=True)
        end_dc = torch.cuda.Event(enable_timing=True)

        start_dc.record()
        for _ in range(gen_len - 1):
            next_logits = generator.generate_next_token(current_token)
            current_token = sample_tokens(next_logits.clone(), gen_cfg.temperature)
        end_dc.record()

        torch.cuda.synchronize()
        decode_ms = start_dc.elapsed_time(end_dc)
        decode_comm_ms = comm_profiler.stop_and_collect() if comm_profiler else 0.0

        if not is_warmup:
            result.prefill_ms.append(prefill_ms)
            result.decode_ms.append(decode_ms)
            result.total_ms.append(prefill_ms + decode_ms)
            if comm_profiler:
                result.prefill_comm_ms.append(prefill_comm_ms)
                result.decode_comm_ms.append(decode_comm_ms)

        if _is_main_rank():
            tag = "WARMUP" if is_warmup else f"trial {trial_idx - warmup + 1}/{n_trials}"
            comm_str = ""
            if comm_profiler:
                comm_str = (f"  pf_comm={prefill_comm_ms:.1f}ms  "
                            f"dc_comm={decode_comm_ms:.1f}ms")
            print(
                f"  [{tag}] prefill={prefill_ms:.1f}ms  "
                f"decode={decode_ms:.1f}ms  "
                f"total={prefill_ms + decode_ms:.1f}ms"
                f"{comm_str}"
            )

    return result


# ---------------------------------------------------------------------------
# Model loaders (distributed-safe)
# ---------------------------------------------------------------------------

def _consolidate_on_rank0(ckpt_path: str) -> str:
    """Only rank 0 runs consolidation; all ranks wait at a barrier, then
    return the path to the consolidated directory."""
    ckpt_dir = Path(ckpt_path)
    
    # Already consolidated (e.g. single-dir checkpoint with params.json + .pth)
    if (ckpt_dir / "params.json").exists() and (ckpt_dir / CONSOLIDATE_NAME).exists():
        if dist.is_initialized():
            dist.barrier()
        return str(ckpt_dir)

    consolidate_path = ckpt_dir / CONSOLIDATE_FOLDER

    if _is_main_rank():
        if not (consolidate_path / CONSOLIDATE_NAME).exists():
            consolidate_checkpoints(str(ckpt_dir))

    if dist.is_initialized():
        dist.barrier()

    return str(consolidate_path)


def _load_stem(ckpt_path: str):
    consolidated = _consolidate_on_rank0(ckpt_path)
    model, tokenizer, _ = load_stem_model(consolidated)
    return model, tokenizer


def _load_base(ckpt_path: str):
    consolidated = _consolidate_on_rank0(ckpt_path)
    model, tokenizer, _ = load_base_model(consolidated)
    return model, tokenizer


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Benchmark STEM vs base model generation latency"
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="key=value overrides (e.g. stem_ckpt=... base_ckpt=...)",
    )
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--stem_parallel_size", type=int, default=1,
        help="STEM model-parallel degree (set >1 for distributed benchmark; "
             "must match nproc-per-node)",
    )
    args = parser.parse_args()

    overrides = {}
    for o in args.overrides:
        k, v = o.split("=", 1)
        overrides[k] = v

    stem_ckpt = overrides.get("stem_ckpt")
    base_ckpt = overrides.get("base_ckpt")

    if not stem_ckpt and not base_ckpt:
        print("ERROR: supply at least one of stem_ckpt=<path> or base_ckpt=<path>")
        sys.exit(1)

    # ---- Distributed init ----
    is_distributed = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if is_distributed:
        _setup_distributed(args.stem_parallel_size)
    
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if _is_main_rank():
        world = dist.get_world_size() if dist.is_initialized() else 1
        print(f"Configuration: context_len={args.context_len}, gen_len={args.gen_len}, "
              f"n_trials={args.n_trials}, warmup={args.warmup}, dtype={args.dtype}")
        print(f"Distributed: world_size={world}, stem_parallel_size={args.stem_parallel_size}")

    results = {}

    # ---- STEM model ----
    if stem_ckpt:
        if _is_main_rank():
            print(f"\n{'#' * 60}")
            print(f"  Loading STEM model from: {stem_ckpt}")
            print(f"{'#' * 60}")

        model, tokenizer = _load_stem(stem_ckpt)

        if _is_main_rank():
            print(f"  Parameters: {_count_params(model):,}")

        # Install comm profiler when running distributed STEM
        comm_profiler = None
        if is_distributed and args.stem_parallel_size > 1:
            comm_profiler = CommProfiler()
            comm_profiler.install()

        if _is_main_rank():
            print(f"  Running benchmark (comm_profiling={'ON' if comm_profiler else 'OFF'}) ...")

        results["STEM Model"] = benchmark_generation(
            model, tokenizer,
            context_len=args.context_len,
            gen_len=args.gen_len,
            n_trials=args.n_trials,
            warmup=args.warmup,
            comm_profiler=comm_profiler,
            dtype=dtype,
        )

        if comm_profiler:
            comm_profiler.uninstall()

        del model
        torch.cuda.empty_cache()

    # ---- Base model ----
    if base_ckpt:
        if _is_main_rank():
            print(f"\n{'#' * 60}")
            print(f"  Loading base model from: {base_ckpt}")
            print(f"{'#' * 60}")

        model, tokenizer = _load_base(base_ckpt)

        if _is_main_rank():
            print(f"  Parameters: {_count_params(model):,}")
            print(f"  Running benchmark ...")

        results["Base Model"] = benchmark_generation(
            model, tokenizer,
            context_len=args.context_len,
            gen_len=args.gen_len,
            n_trials=args.n_trials,
            warmup=args.warmup,
            dtype=dtype,
        )

        del model
        torch.cuda.empty_cache()

    # ---- Summary (rank 0 only) ----
    if not _is_main_rank():
        if dist.is_initialized():
            dist.barrier()
        return

    print(f"\n{'#' * 60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'#' * 60}")
    print(f"  Context length        : {args.context_len} tokens")
    print(f"  Generation length     : {args.gen_len} tokens")
    if is_distributed:
        print(f"  World size            : {dist.get_world_size()}")
        print(f"  STEM parallel size    : {args.stem_parallel_size}")

    for label, res in results.items():
        _print_results(label, res, args.gen_len)

    if len(results) == 2:
        stem_res = results["STEM Model"]
        base_res = results["Base Model"]
        stem_mean = torch.tensor(stem_res.decode_ms).mean().item()
        base_mean = torch.tensor(base_res.decode_ms).mean().item()
        overhead_pct = (stem_mean - base_mean) / base_mean * 100
        print(f"  STEM decode overhead vs base: {overhead_pct:+.1f}%")

        stem_pf = torch.tensor(stem_res.prefill_ms).mean().item()
        base_pf = torch.tensor(base_res.prefill_ms).mean().item()
        pf_overhead = (stem_pf - base_pf) / base_pf * 100
        print(f"  STEM prefill overhead vs base: {pf_overhead:+.1f}%")
        print()

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
