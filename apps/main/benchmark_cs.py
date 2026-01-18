#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Benchmark throughput/latency for prefilling and decoding using the packed generator.

Examples:
  python apps/main/benchmark.py --ckpt /path/to/consolidated --batch-size 8
  python apps/main/benchmark.py --ckpt /path/to/consolidated --batch-size 8 --compile-prefilling --reduce-generation-overhead
  python apps/main/benchmark.py --ckpt /path/to/consolidated --batch-size 8 --warmup-iters 5 --iters 20 --gen-len 256
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import torch

# Allow running as a script: `python apps/main/benchmark.py ...`
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
    pack_prompts,
    sample_tokens,
)


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_prompt_tokens_exact_len(tokenizer, prompt_len: int) -> List[int]:
    """
    Build a prompt that is exactly `prompt_len` tokens (including BOS if tokenizer adds it).
    We avoid EOS and use a repeatable filler to hit the target length.
    """
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be > 0, got {prompt_len}")

    base = "Benchmarking prompt:"
    toks = tokenizer.encode(base, add_bos=True, add_eos=False)

    filler_ids = tokenizer.encode(" hello", add_bos=False, add_eos=False)
    if not filler_ids:
        filler_ids = tokenizer.encode(" the", add_bos=False, add_eos=False)
    if not filler_ids:
        # Last resort: use a non-special token if we can find one.
        # (Avoid eos_id if present)
        fallback = 1
        if hasattr(tokenizer, "eos_id") and fallback == tokenizer.eos_id:
            fallback = 2
        filler_ids = [fallback]

    while len(toks) < prompt_len:
        toks.extend(filler_ids)

    toks = toks[:prompt_len]
    if hasattr(tokenizer, "eos_id"):
        # Ensure we don't accidentally end with EOS which could affect some model behaviors.
        if toks[-1] == tokenizer.eos_id and len(filler_ids) > 0:
            toks[-1] = filler_ids[0]
    return toks


@torch.inference_mode()
def _run_fixed_steps(
    generator: PackedCausalTransformerGenerator,
    prompt_tokens_batch: List[List[int]],
    gen_len: int,
) -> Tuple[int, int]:
    """
    Run one end-to-end iteration (prefill + fixed-length decode).

    Returns:
      (prompt_tokens_processed, generated_tokens_processed)
    """
    if gen_len <= 0:
        raise ValueError(f"gen_len must be > 0, got {gen_len}")

    packed, lengths = pack_prompts(prompt_tokens_batch)
    packed = packed.to(device=generator.device)
    lengths = lengths.to(device=generator.device)

    # Prefill (writes KV cache)
    prompt_logits = generator.prefill(packed.unsqueeze(0), lengths)

    # Select the last-token logits of each sequence, then sample first generated token
    all_tokens = sample_tokens(prompt_logits, generator.temperature, generator.top_p, generator.top_k)
    last_indices = lengths.cumsum(0) - 1
    current_token = all_tokens[:, last_indices]

    # Decode fixed number of steps (ignore EOS / stop strings for benchmarking consistency)
    for _ in range(1, gen_len):
        next_logits = generator.generate_next_token(current_token)
        current_token = sample_tokens(next_logits.clone(), generator.temperature, generator.top_p, generator.top_k)

    prompt_tok = int(lengths.sum().item())
    gen_tok = int(lengths.numel() * gen_len)
    return prompt_tok, gen_tok


@torch.inference_mode()
def _run_prefill_only(
    generator: PackedCausalTransformerGenerator,
    prompt_tokens_batch: List[List[int]],
) -> int:
    packed, lengths = pack_prompts(prompt_tokens_batch)
    packed = packed.to(device=generator.device)
    lengths = lengths.to(device=generator.device)
    _ = generator.prefill(packed.unsqueeze(0), lengths)
    return int(lengths.sum().item())


@torch.inference_mode()
def _run_decode_only(
    generator: PackedCausalTransformerGenerator,
    prompt_tokens_batch: List[List[int]],
    gen_len: int,
) -> int:
    """
    Decode only: still does one prefill to set up KV cache, then times only decode.
    Returns generated tokens processed.
    """
    packed, lengths = pack_prompts(prompt_tokens_batch)
    packed = packed.to(device=generator.device)
    lengths = lengths.to(device=generator.device)

    prompt_logits = generator.prefill(packed.unsqueeze(0), lengths)
    all_tokens = sample_tokens(prompt_logits, generator.temperature, generator.top_p, generator.top_k)
    last_indices = lengths.cumsum(0) - 1
    current_token = all_tokens[:, last_indices]

    for _ in range(1, gen_len):
        next_logits = generator.generate_next_token(current_token)
        current_token = sample_tokens(next_logits.clone(), generator.temperature, generator.top_p, generator.top_k)

    return int(lengths.numel() * gen_len)


def _format_stats(name: str, seconds: float, tokens: int) -> str:
    tok_s = tokens / seconds if seconds > 0 else float("inf")
    ms = seconds * 1e3
    return f"{name:>12}: {ms:9.3f} ms  |  {tok_s:12.2f} tok/s  |  tokens={tokens}"


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark packed prefill/decode latency & throughput.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to consolidated checkpoint directory.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (#prompts).")
    p.add_argument("--prompt-len", type=int, default=200, help="Prompt length in tokens (approx requested: 200).")
    p.add_argument("--gen-len", type=int, default=128, help="Number of tokens to decode per sequence.")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"], help="Computation dtype.")
    p.add_argument("--device", type=str, default="cuda", help="Device (e.g. cuda, cuda:0).")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 => greedy).")
    p.add_argument("--top-p", type=float, default=None, help="Top-p sampling (only used if temperature > 0).")
    p.add_argument("--top-k", type=int, default=None, help="Top-k sampling (only used if temperature > 0).")
    p.add_argument("--compile-prefilling", action="store_true", help="Enable torch.compile for prefilling.")
    p.add_argument(
        "--reduce-generation-overhead",
        action="store_true",
        help="Enable torch.compile(mode=reduce-overhead) for token-by-token decoding.",
    )
    p.add_argument("--warmup-iters", type=int, default=3, help="Warmup iterations (absorbs compile/caching).")
    p.add_argument("--iters", type=int, default=10, help="Timed iterations.")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Override generator max_tokens. Default auto=(prompt_len+gen_len)*batch_size.",
    )
    args = p.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.prompt_len <= 0:
        raise SystemExit("--prompt-len must be > 0")
    if args.gen_len <= 0:
        raise SystemExit("--gen-len must be > 0")

    # Load model/tokenizer
    model, tokenizer, _ = load_consolidated_model_and_tokenizer(args.ckpt)

    # Build synthetic input prompt of ~200 tokens and replicate to batch size
    prompt_tokens = _build_prompt_tokens_exact_len(tokenizer, args.prompt_len)
    prompts_batch = [list(prompt_tokens) for _ in range(args.batch_size)]

    max_tokens = args.max_tokens if args.max_tokens > 0 else (args.prompt_len + args.gen_len) * args.batch_size

    gen_cfg = PackedCausalTransformerGeneratorArgs(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_gen_len=args.gen_len,
        max_tokens=max_tokens,
        max_prompt_len=args.prompt_len,  # keep it fixed/consistent
        until=[],
        compile_prefilling=args.compile_prefilling,
        reduce_generation_overhead=args.reduce_generation_overhead,
        show_progress=False,
        dtype=args.dtype,
        device=args.device,
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Optional perf knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(
        "Benchmark config:\n"
        f"  ckpt={args.ckpt}\n"
        f"  device={args.device} dtype={args.dtype}\n"
        f"  batch_size={args.batch_size} prompt_len={args.prompt_len} gen_len={args.gen_len}\n"
        f"  max_tokens={max_tokens}\n"
        f"  compile_prefilling={args.compile_prefilling} reduce_generation_overhead={args.reduce_generation_overhead}\n"
        f"  warmup_iters={args.warmup_iters} iters={args.iters}\n"
    )

    # Warmup (includes compilation overhead)
    for _ in range(args.warmup_iters):
        _sync_if_cuda(args.device)
        _ = _run_fixed_steps(generator, prompts_batch, args.gen_len)
    _sync_if_cuda(args.device)

    # Timed: end-to-end
    e2e_tokens = 0
    _sync_if_cuda(args.device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        pt, gt = _run_fixed_steps(generator, prompts_batch, args.gen_len)
        e2e_tokens += pt + gt
    _sync_if_cuda(args.device)
    t1 = time.perf_counter()
    e2e_s = t1 - t0

    # Timed: prefill-only
    prefill_tokens = 0
    _sync_if_cuda(args.device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        prefill_tokens += _run_prefill_only(generator, prompts_batch)
    _sync_if_cuda(args.device)
    t1 = time.perf_counter()
    prefill_s = t1 - t0

    # Timed: decode-only (still does a prefill, but timing focuses on decode)
    decode_tokens = 0
    _sync_if_cuda(args.device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        decode_tokens += _run_decode_only(generator, prompts_batch, args.gen_len)
    _sync_if_cuda(args.device)
    t1 = time.perf_counter()
    decode_s = t1 - t0

    # Report
    print("Results (averaged over iters):")
    print(_format_stats("end_to_end", e2e_s / args.iters, e2e_tokens // args.iters))
    print(_format_stats("prefill", prefill_s / args.iters, prefill_tokens // args.iters))
    print(_format_stats("decode", decode_s / args.iters, decode_tokens // args.iters))


if __name__ == "__main__":
    main()


