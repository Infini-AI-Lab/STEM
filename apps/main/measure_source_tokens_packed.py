#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Per-source token counts using the full training dataloader stack from ``lingua/data.py``
(the same chain as ``build_dataloader`` / ``train.py``):

  init_state → setup_sources → choose_source → tokenize → pack_tokens
  → batch_and_shuffle_prefetched_sequences

Tokens are attributed to a **source at packing time**: each contiguous segment copied into the
pack buffer is labeled with the document source for that segment (so cross-document boundaries
inside a packed ``seq_len`` window are reflected).

This script does **not** modify ``measure_source_tokens.py``. Use ``--max-batches`` to cap work;
the first prefetch fill reads ``prefetch_size * batch_size`` packed sequences (same as training).

With ``logging.wandb`` in the YAML, rank 0 logs a final summary. Use ``--wandb-log-every N`` (``N>0``)
to also ``wandb.log`` cumulative global per-source stats every ``N`` training batches (all ranks
sync via ``all_reduce`` each time).

Example::

    torchrun --standalone --nproc_per_node=8 \\
      apps/main/measure_source_tokens_packed.py \\
      --config apps/main/configs/olmo2_1B_midfine.yaml \\
      --max-batches 100 \\
      --dump-dir logs/measure_packed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

import wandb

from lingua.data import (
    DataArgs,
    PrefetchState,
    TokenizerState,
    batch_and_shuffle_prefetched_sequences,
    choose_source_tagged,
    init_state,
    pack_tokens_with_source_counts,
    setup_sources,
    tokenize_tagged,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    get_device_mesh,
    get_is_master,
    get_world_size,
    setup_env,
    setup_torch_distributed,
)
from lingua.logger import init_logger
from lingua.metrics import LoggingArgs
from lingua.args import dump_config
logger = logging.getLogger(__name__)


@dataclass
class PackedMeasureConfig:
    data: DataArgs = field(default_factory=DataArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    logging: Optional[LoggingArgs] = None
    dump_dir: str = ""


def loose_dataclass(cls, data: dict) -> Any:
    """Merge YAML into ``cls`` without OmegaConf struct rejection on dynamic keys (e.g. ``data.sources``)."""
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, False)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))


def validate_packed_config(cfg: PackedMeasureConfig) -> None:
    assert cfg.data.root_dir, "data.root_dir must be set"
    assert cfg.data.sources, "data.sources must be non-empty"
    if not cfg.dump_dir:
        cfg.dump_dir = "logs/measure_source_tokens_packed"

    for source in cfg.data.sources:
        data_path = os.path.join(cfg.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        cfg.distributed.dp_replicate * cfg.distributed.dp_shard * cfg.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % cfg.distributed.dp_shard == 0
        cfg.distributed.dp_replicate = get_world_size() // cfg.distributed.dp_shard
        assert cfg.distributed.dp_replicate % cfg.distributed.tp_size == 0
        cfg.distributed.dp_replicate = cfg.distributed.dp_replicate // cfg.distributed.tp_size
        logger.warning(
            "Adjusted Data Parallel size to %s",
            cfg.distributed.dp_replicate * cfg.distributed.dp_shard,
        )
        assert (
            cfg.distributed.dp_replicate
            * cfg.distributed.dp_shard
            * cfg.distributed.tp_size
            == get_world_size()
        )
        if cfg.distributed.fsdp_type == "no_shard":
            assert cfg.distributed.dp_shard == 1 and cfg.distributed.dp_replicate == get_world_size()


def resolve_data_parallel(data: DataArgs, dp_rank: int, dp_degree: int) -> tuple[int, int]:
    data_rank = dp_rank
    data_world_size = dp_degree
    if data.node_local:
        lr = os.environ.get("LOCAL_RANK")
        lw = os.environ.get("LOCAL_WORLD_SIZE")
        assert lr is not None and lw is not None, "data.node_local=true requires LOCAL_RANK and LOCAL_WORLD_SIZE"
        data_rank = int(lr)
        data_world_size = int(lw)
        assert data_world_size > 0 and 0 <= data_rank < data_world_size
    return data_rank, data_world_size


def _maybe_init_wandb_packed_early(
    cfg: PackedMeasureConfig,
    *,
    config_path: str,
    max_batches: int,
    yaml_name: Optional[str],
) -> bool:
    """Rank 0 only: start W&B before the batch loop when ``--wandb-log-every`` is enabled."""
    if not get_is_master():
        return False
    if cfg.logging is None or cfg.logging.wandb is None:
        return False
    if wandb.run is not None:
        return False
    wandb.init(
        config={
            "measure_source_tokens_packed": True,
            "config_path": config_path,
            "max_batches": max_batches,
            "yaml_name": yaml_name,
        },
        **asdict(cfg.logging.wandb),
    )
    logger.info("wandb: started run for periodic measure_packed/* logs")
    return True


def wandb_log_packed_global_counts_snapshot(
    device: torch.device,
    order: List[str],
    source_counts: Dict[str, int],
    batches_completed: int,
    wandb_step: int,
) -> None:
    """All ranks: all_reduce cumulative packed token counts; rank 0 logs each source's share of global tokens."""
    vec = torch.tensor([source_counts[s] for s in order], dtype=torch.long, device=device)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    if not get_is_master():
        return
    if wandb.run is None:
        return
    total_global = int(vec.sum().item())
    metrics: Dict[str, Any] = {}
    for i, s in enumerate(order):
        tok = int(vec[i].item())
        obs = tok / total_global if total_global else 0.0
        metrics[f"measure_packed/fraction_observed/{s}"] = obs * 100.0
    wandb.log(metrics, step=wandb_step)
    logger.info(
        "wandb: logged per-source fractions at wandb_step=%s (batches=%s, global_packed_tokens=%s)",
        wandb_step,
        batches_completed,
        total_global,
    )


def log_packed_to_wandb(
    cfg: PackedMeasureConfig,
    out: dict,
    *,
    yaml_name: Optional[str],
    max_batches: int,
    wandb_step: int = 0,
    wandb_started_early: bool = False,
) -> None:
    """Rank 0: log final summary; init if needed; finish if this code or early init started the run."""
    if not get_is_master():
        return
    if cfg.logging is None or cfg.logging.wandb is None:
        return

    _wandb_initialized_here = False
    if wandb.run is None:
        wandb.init(
            config={
                "measure_source_tokens_packed": True,
                "config_path": out["config"],
                "max_batches": max_batches,
                "yaml_name": yaml_name,
                "summary": out,
            },
            **asdict(cfg.logging.wandb),
        )
        _wandb_initialized_here = True

    if wandb.run is not None:
        m: Dict[str, Any] = {}
        for row in out["per_source"]:
            s = row["source"]
            m[f"measure_packed/fraction_observed/{s}"] = row["fraction_observed"]
        wandb.log(m, step=wandb_step)
        logger.info("Logged %s measure_packed keys to wandb (step=%s)", len(m), wandb_step)

    if _wandb_initialized_here or wandb_started_early:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Stop each rank after yielding this many training batches (prefetch pipeline).",
    )
    parser.add_argument(
        "--wandb-log-every",
        type=int,
        default=0,
        help="If >0 and logging.wandb is set, log cumulative global packed-token mixture every N batches "
        "(0 = only the final summary on rank 0).",
    )
    parser.add_argument("--dump-dir", type=str, default=None)
    args_ns = parser.parse_args()

    file_cfg = OmegaConf.load(args_ns.config)
    raw = OmegaConf.to_container(file_cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    logging_raw = raw.get("logging")
    cfg: PackedMeasureConfig = PackedMeasureConfig(
        data=loose_dataclass(DataArgs, raw.get("data") or {}),
        distributed=loose_dataclass(DistributedArgs, raw.get("distributed") or {}),
        env=loose_dataclass(EnvironmentArgs, raw.get("env") or {}),
        logging=loose_dataclass(LoggingArgs, logging_raw) if logging_raw is not None else None,
        dump_dir=str(raw.get("dump_dir") or ""),
    )
    if args_ns.dump_dir:
        cfg.dump_dir = args_ns.dump_dir

    validate_packed_config(cfg)

    dump_dir = cfg.dump_dir
    if get_is_master():
        os.makedirs(dump_dir, exist_ok=True)
    init_logger(Path(dump_dir) / "measure_source_tokens_packed.log")

    setup_env(cfg.env)
    setup_torch_distributed(cfg.distributed)

    try:
        world_mesh = get_device_mesh(cfg.distributed)
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if cfg.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        data_rank, data_world_size = resolve_data_parallel(cfg.data, dp_rank, dp_degree)

        prefetch_state: PrefetchState = init_state(
            root_dir=cfg.data.root_dir,
            sources=cfg.data.sources,
            batch_size=cfg.data.batch_size,
            prefetch_size=cfg.data.prefetch_size,
            seq_len=cfg.data.seq_len,
            n_views=cfg.data.n_views,
            seed=cfg.data.seed,
            rank=data_rank,
            world_size=data_world_size,
            add_bos=cfg.data.add_bos,
            add_eos=cfg.data.add_eos,
            tokenizer_name=cfg.data.tokenizer.name,
            tokenizer_path=cfg.data.tokenizer.path,
        )

        pack_state = prefetch_state["it_state"]
        tokenizer_state: TokenizerState = pack_state["it_state"]
        multi_state = tokenizer_state["it_state"]

        path_to_iter = setup_sources(multi_state)
        source_names = sorted(cfg.data.sources.keys())
        source_counts: Dict[str, int] = {s: 0 for s in source_names}

        tagged = choose_source_tagged(
            source_to_iterator=path_to_iter,
            source_to_state=multi_state["source_to_state"],
            root_dir=multi_state["root_dir"],
            sources=multi_state["sources"],
            rng_state=multi_state["rng_state"],
            source_counts=source_counts if cfg.data.track_packed_source_mixture else None,
        )
        tok_it = tokenize_tagged(
            tagged,
            tokenizer_state["add_bos"],
            tokenizer_state["add_eos"],
            tokenizer_state["name"],
            tokenizer_state["path"],
        )
        packed_it = pack_tokens_with_source_counts(tok_it, pack_state, source_counts)

        batch_it = batch_and_shuffle_prefetched_sequences(
            data_loader=packed_it,
            batch_size=prefetch_state["batch_size"],
            prefetch_size=prefetch_state["prefetch_size"],
            seq_len=pack_state["output_seq_len"],
            n_views=pack_state["n_views"],
            state=prefetch_state,
        )

        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        wandb_log_every = max(0, args_ns.wandb_log_every)
        wandb_started_early = False
        wandb_progress_step = 0
        if (
            wandb_log_every > 0
            and cfg.logging is not None
            and cfg.logging.wandb is not None
        ):
            yn_early = raw.get("name")
            wandb_started_early = _maybe_init_wandb_packed_early(
                cfg,
                config_path=args_ns.config,
                max_batches=args_ns.max_batches,
                yaml_name=yn_early if isinstance(yn_early, str) else None,
            )
        dist.barrier()

        progress_log_every = (
            wandb_log_every
            if wandb_log_every > 0
            else max(1, min(100, max(1, args_ns.max_batches // 10)))
        )
        if get_is_master():
            logger.info(
                "Packed batch loop: max_batches=%s, logging progress every %s batch(es)%s",
                args_ns.max_batches,
                progress_log_every,
                (
                    f", wandb fractions every {wandb_log_every} batch(es)"
                    if wandb_log_every > 0
                    else ""
                ),
            )

        batches = 0
        t_loop = time.perf_counter()
        try:
            for _batch, _st in batch_it:
                batches += 1
                if batches == 1 or batches % progress_log_every == 0:
                    elapsed = time.perf_counter() - t_loop
                    local_tok = sum(source_counts.values())
                    logger.info(
                        "[rank %s data_rank=%s] batches=%s/%s local_packed_tokens=%s elapsed=%.1fs (%.2f batch/s)",
                        dist.get_rank(),
                        data_rank,
                        batches,
                        args_ns.max_batches,
                        local_tok,
                        elapsed,
                        batches / elapsed if elapsed > 0 else 0.0,
                    )
                if (
                    wandb_log_every > 0
                    and batches % wandb_log_every == 0
                    and cfg.logging is not None
                    and cfg.logging.wandb is not None
                ):
                    wandb_log_packed_global_counts_snapshot(
                        device,
                        source_names,
                        source_counts,
                        batches,
                        wandb_progress_step,
                    )
                    wandb_progress_step += 1
                if batches >= args_ns.max_batches:
                    break
        finally:
            for it in path_to_iter.values():
                it.close()

        elapsed_loop = time.perf_counter() - t_loop
        local_packed_total = sum(source_counts.values())
        logger.info(
            "[rank %s data_rank=%s] packed loop done: batches=%s local_packed_tokens=%s in %.1fs (%.2f batch/s)",
            dist.get_rank(),
            data_rank,
            batches,
            local_packed_total,
            elapsed_loop,
            batches / elapsed_loop if elapsed_loop > 0 else 0.0,
        )
        order = source_names
        local_vec = torch.tensor([source_counts[s] for s in order], dtype=torch.long, device=device)
        dist.all_reduce(local_vec, op=dist.ReduceOp.SUM)
        global_counts = {s: int(local_vec[i].item()) for i, s in enumerate(order)}
        total_global = sum(global_counts.values())

        weight_sum = sum(cfg.data.sources.values())
        expected_frac = {s: cfg.data.sources[s] / weight_sum for s in order}
        rows = []
        for s in order:
            obs = global_counts[s] / total_global if total_global else 0.0
            rows.append(
                {
                    "source": s,
                    "tokens": global_counts[s],
                    "fraction_observed": obs,
                    "fraction_config": expected_frac[s],
                }
            )

        out = {
            "config": args_ns.config,
            "mode": "packed_prefetch_pipeline",
            "data_rank": data_rank,
            "data_world_size": data_world_size,
            "batches_this_rank": batches,
            "max_batches_arg": args_ns.max_batches,
            "local_packed_tokens": local_packed_total,
            "global_total_packed_tokens": total_global,
            "per_source": rows,
            "notes": (
                "Counts are tokens attributed during pack_tokens (buffer segments). "
                "Training uses the same packed chunks after batch_and_shuffle."
            ),
        }

        if get_is_master():
            dump_config(cfg, Path(dump_dir) / "packed_measure_config_snapshot.yaml")
            out_path = Path(dump_dir) / "measure_source_tokens_packed.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            logger.info("Wrote %s", out_path)
            print(json.dumps(out, indent=2))
            yn = raw.get("name")
            log_packed_to_wandb(
                cfg,
                out,
                yaml_name=yn if isinstance(yn, str) else None,
                max_batches=args_ns.max_batches,
                wandb_step=wandb_progress_step,
                wandb_started_early=wandb_started_early,
            )

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
