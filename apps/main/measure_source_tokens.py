#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Estimate per-source token counts using the same dataloading path as train.py:

  init_choice_state -> setup_sources -> choose_source (weighted) -> tokenizer

Each time a source is sampled, one JSONL record is tokenized and its length is
added to that source's bucket. This reflects the mixture controlled by
``data.sources`` before packing; ``pack_tokens`` may concatenate fragments from
different sources at chunk boundaries.

Distributed: uses the same ``data_rank`` / ``data_world_size`` rules as
``apps/main/train.py`` (including ``data.node_local`` + LOCAL_RANK /
LOCAL_WORLD_SIZE). Run with torchrun / Slurm like training.

Example::

    torchrun --nproc_per_node=8 apps/main/measure_source_tokens.py \\
        --config apps/main/configs/olmo2_1B_midfine.yaml --max-tokens 500000

Single process (no torchrun)::

    python apps/main/measure_source_tokens.py \\
        --config apps/main/configs/olmo2_1B_midfine.yaml --max-tokens 100000

If the YAML includes ``logging.wandb`` (same as ``train.py``),
rank 0 runs a W&B run. With ``--wandb-log-tokens`` (default: same cadence as ``--progress-log-tokens``),
rank 0 logs global per-source token counts and observed fractions on increasing steps while measuring;
a final ``wandb.log`` is emitted at the last step, then the run is finished.

Progress: periodic INFO logs (``--progress-log-tokens``; default auto from ``--max-tokens``) on every
rank, and a tqdm bar when ``WORLD_SIZE=1`` and stderr is a TTY (disable with ``--no-progress-bar``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Type, TypeVar

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

import wandb
from tqdm import tqdm

from lingua.args import dataclass_from_dict, dump_config
from lingua.data import (
    TRAIN_DATA_FILE_PATTERN,
    DataArgs,
    MultiChoiceState,
    init_choice_state,
    setup_sources,
)
from lingua.metrics import LoggingArgs
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
from lingua.tokenizer import build_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class MeasureConfig:
    """YAML subset consumed by this script (same sections as ``train.py`` uses for data/env/distributed)."""

    data: DataArgs = field(default_factory=DataArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    logging: Optional[LoggingArgs] = None
    dump_dir: str = ""


def validate_measure_config(cfg: MeasureConfig) -> None:
    """Align ``distributed`` with process count and check data dirs (subset of ``validate_train_args``)."""
    assert cfg.data.root_dir, "data.root_dir must be set"
    assert cfg.data.sources, "data.sources must be non-empty"
    if not cfg.dump_dir:
        cfg.dump_dir = "logs/measure_source_tokens"

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


def choose_source_tagged(
    source_to_iterator: Dict[str, Iterator],
    source_to_state: Dict[str, Any],
    root_dir: str,
    sources: Dict[str, float],
    rng_state: Dict[str, Any],
) -> Iterator:
    """Same sampling as lingua.data.choose_source, but yields (content, source_name, state)."""
    n_sources = len(sources)
    possible_sources = list(sources.keys())
    weights = list(sources.values())
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    while True:
        norm_weights = np.array(weights) / np.array(weights).sum()
        source_choice = possible_sources[rng.choice(n_sources, p=norm_weights)]
        seq, state = next(source_to_iterator[source_choice])
        source_to_state = {**source_to_state, source_choice: state}
        multi_choice_state = MultiChoiceState(
            root_dir=root_dir,
            sources=sources,
            source_to_state=source_to_state,
            rng_state=rng.bit_generator.state,
        )
        yield seq, source_choice, multi_choice_state


def resolve_data_parallel(
    data: DataArgs,
    dp_rank: int,
    dp_degree: int,
) -> tuple[int, int]:
    """Match train.py: optional node-local sharding."""
    data_rank = dp_rank
    data_world_size = dp_degree
    if data.node_local:
        local_rank_env = os.environ.get("LOCAL_RANK")
        local_world_env = os.environ.get("LOCAL_WORLD_SIZE")
        assert (
            local_rank_env is not None and local_world_env is not None
        ), "data.node_local=true requires LOCAL_RANK and LOCAL_WORLD_SIZE"
        data_rank = int(local_rank_env)
        data_world_size = int(local_world_env)
        assert data_world_size > 0
        assert 0 <= data_rank < data_world_size
    return data_rank, data_world_size


def _maybe_init_wandb_measure_early(
    measure_cfg: MeasureConfig,
    *,
    config_path: str,
    max_tokens_limit: int,
    yaml_name: Optional[str],
) -> bool:
    """Rank 0 only: start W&B before the loop when we will emit partial logs. Returns True if this call created the run."""
    if not get_is_master():
        return False
    if measure_cfg.logging is None or measure_cfg.logging.wandb is None:
        return False
    if wandb.run is not None:
        return False
    wandb.init(
        config={
            "measure_source_tokens": True,
            "config_path": config_path,
            "max_tokens_per_rank": max_tokens_limit,
            "yaml_name": yaml_name,
        },
        **asdict(measure_cfg.logging.wandb),
    )
    logger.info("wandb: started run for live measure/* logs")
    return True


def wandb_log_global_counts_snapshot(
    measure_cfg: MeasureConfig,
    device: torch.device,
    order: List[str],
    counts: Dict[str, int],
    wandb_step: int,
) -> None:
    """All ranks: all_reduce count vector; rank 0 logs global per-source tokens and fractions."""
    vec = torch.tensor([counts[s] for s in order], dtype=torch.long, device=device)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    if not get_is_master():
        return
    if wandb.run is None:
        return
    total_global = int(vec.sum().item())
    weight_sum = sum(measure_cfg.data.sources.values())
    metrics: Dict[str, Any] = {
        "measure/global_total_tokens": total_global,
        "measure/wandb_step": wandb_step,
    }
    for i, s in enumerate(order):
        tok = int(vec[i].item())
        obs = tok / total_global if total_global else 0.0
        cfg_frac = measure_cfg.data.sources[s] / weight_sum
        # metrics[f"measure/tokens/{s}"] = tok
        metrics[f"measure/fraction_observed/{s}"] = obs * 100.0
        # metrics[f"measure/fraction_config/{s}"] = cfg_frac * 100.0
        # metrics[f"measure/fraction_delta/{s}"] = (obs - cfg_frac) * 100.0
    wandb.log(metrics, step=wandb_step)
    logger.info("wandb: logged global snapshot at step %s (total_tokens=%s)", wandb_step, total_global)


def log_measure_to_wandb(
    measure_cfg: MeasureConfig,
    out: dict,
    *,
    yaml_name: Optional[str],
    max_tokens_limit: int,
    wandb_step: int = 0,
    wandb_started_early: bool = False,
) -> None:
    """Log final mixture metrics on rank 0; init if needed; finish if this code started the run (early or here)."""
    if not get_is_master():
        return
    if measure_cfg.logging is None or measure_cfg.logging.wandb is None:
        return

    _wandb_initialized_here = False
    if wandb.run is None:
        wandb.init(
            config={
                "measure_source_tokens": True,
                "config_path": out["config"],
                "max_tokens_per_rank": max_tokens_limit,
                "yaml_name": yaml_name,
                "summary": out,
            },
            **asdict(measure_cfg.logging.wandb),
        )
        _wandb_initialized_here = True

    if wandb.run is not None:
        metrics: Dict[str, Any] = {
            "measure/global_total_tokens": out["global_total_tokens"],
            "measure/local_tokens_this_rank": out["local_tokens"],
            "measure/local_documents_this_rank": out["local_documents"],
            "measure/data_world_size": out["data_world_size"],
        }
        for row in out["per_source"]:
            s = row["source"]
            metrics[f"measure/tokens/{s}"] = row["tokens"]
            metrics[f"measure/fraction_observed/{s}"] = row["fraction_observed"]
            metrics[f"measure/fraction_config/{s}"] = row["fraction_config"]
            metrics[f"measure/fraction_delta/{s}"] = (
                row["fraction_observed"] - row["fraction_config"]
            )
        wandb.log(metrics, step=wandb_step)
        logger.info("Logged %s measure keys to wandb (step=%s)", len(metrics), wandb_step)

    if _wandb_initialized_here or wandb_started_early:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training YAML (same as train.py: data.sources, tokenizer, seed, ...).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500_000,
        help="Stop each rank after counting this many tokens locally (default 500k).",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap on documents per rank (whichever limit hits first).",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="If set, master writes measure_source_tokens.json here and initializes logger.",
    )
    parser.add_argument(
        "--progress-log-tokens",
        type=int,
        default=-1,
        help="Log local progress every N tokens per rank (-1=auto from --max-tokens, 0=disable).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm when WORLD_SIZE=1 (stderr TTY).",
    )
    parser.add_argument(
        "--wandb-log-tokens",
        type=int,
        default=-1,
        help="W&B: log global per-source fractions every N global tokens (sum across ranks; "
        "-1=use --progress-log-tokens interval, 0=only final log).",
    )
    args_ns = parser.parse_args()

    if args_ns.progress_log_tokens == -1:
        progress_log_every = max(50_000, min(args_ns.max_tokens // 25, 50_000_000))
    elif args_ns.progress_log_tokens == 0:
        progress_log_every = 0
    else:
        progress_log_every = args_ns.progress_log_tokens

    if args_ns.wandb_log_tokens == -1:
        wandb_log_every = progress_log_every
    elif args_ns.wandb_log_tokens == 0:
        wandb_log_every = 0
    else:
        wandb_log_every = args_ns.wandb_log_tokens

    file_cfg = OmegaConf.load(args_ns.config)
    raw = OmegaConf.to_container(file_cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    measure_subset = {
        "data": raw.get("data", {}),
        "distributed": raw.get("distributed", {}),
        "env": raw.get("env", {}),
        "logging": raw.get("logging", {}),
        "dump_dir": raw.get("dump_dir", ""),
    }
    measure_cfg: MeasureConfig = dataclass_from_dict(MeasureConfig, measure_subset, strict=False)

    if args_ns.dump_dir:
        measure_cfg.dump_dir = args_ns.dump_dir
    dump_dir = measure_cfg.dump_dir

    tokenizer = build_tokenizer(measure_cfg.data.tokenizer.name, measure_cfg.data.tokenizer.path)
    validate_measure_config(measure_cfg)

    if get_is_master():
        os.makedirs(dump_dir, exist_ok=True)
    init_logger(Path(dump_dir) / "measure_source_tokens.log")

    setup_env(measure_cfg.env)
    setup_torch_distributed(measure_cfg.distributed)
    try:
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        world_mesh = get_device_mesh(measure_cfg.distributed)

        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if measure_cfg.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        data_rank, data_world_size = resolve_data_parallel(measure_cfg.data, dp_rank, dp_degree)

        ws = dist.get_world_size()
        wandb_started_early = False
        wandb_progress_step = 0
        next_global_log_at = wandb_log_every
        last_wandb_check_local = 0
        check_interval_local = (
            max(25_000, wandb_log_every // max(1, 4 * ws)) if wandb_log_every > 0 else 0
        )

        if (
            wandb_log_every > 0
            and measure_cfg.logging is not None
            and measure_cfg.logging.wandb is not None
        ):
            yaml_name_early = raw.get("name")
            wandb_started_early = _maybe_init_wandb_measure_early(
                measure_cfg,
                config_path=args_ns.config,
                max_tokens_limit=args_ns.max_tokens,
                yaml_name=yaml_name_early if isinstance(yaml_name_early, str) else None,
            )
        dist.barrier()

        multi_state = init_choice_state(
            root_dir=measure_cfg.data.root_dir,
            sources=measure_cfg.data.sources,
            seed=measure_cfg.data.seed,
            rank=data_rank,
            world_size=data_world_size,
            file_pattern=TRAIN_DATA_FILE_PATTERN,
        )
        path_to_iter = setup_sources(multi_state)

        source_names: List[str] = sorted(measure_cfg.data.sources.keys())
        counts: Dict[str, int] = {s: 0 for s in source_names}

        tagged = choose_source_tagged(
            source_to_iterator=path_to_iter,
            source_to_state=multi_state["source_to_state"],
            root_dir=multi_state["root_dir"],
            sources=multi_state["sources"],
            rng_state=multi_state["rng_state"],
        )

        use_pbar = (
            ws == 1
            and not args_ns.no_progress_bar
            and hasattr(sys.stderr, "isatty")
            and sys.stderr.isatty()
        )
        pbar = (
            tqdm(
                total=args_ns.max_tokens,
                unit="tok",
                unit_scale=True,
                desc="measure",
                mininterval=0.3,
            )
            if use_pbar
            else None
        )

        total_local = 0
        n_docs = 0
        last_log_tokens = 0
        t_loop = time.perf_counter()
        try:
            for content, source_name, _ in tagged:
                assert "text" in content or "content" in content
                content_key = "text" if "text" in content else "content"
                text = content[content_key]
                tokens = tokenizer.encode(
                    text,
                    add_bos=measure_cfg.data.add_bos,
                    add_eos=measure_cfg.data.add_eos,
                )
                ntok = len(tokens)
                counts[source_name] += ntok
                total_local += ntok
                n_docs += 1

                if pbar is not None:
                    pbar.update(ntok)
                    pbar.set_postfix(docs=n_docs, last=source_name[:24], refresh=False)

                if progress_log_every > 0 and total_local - last_log_tokens >= progress_log_every:
                    elapsed = time.perf_counter() - t_loop
                    rate = total_local / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "[rank %s data_rank=%s] local_tokens=%s / %s  docs=%s  (%.0f tok/s)",
                        dist.get_rank(),
                        data_rank,
                        total_local,
                        args_ns.max_tokens,
                        n_docs,
                        rate,
                    )
                    last_log_tokens = total_local

                if (
                    wandb_log_every > 0
                    and check_interval_local > 0
                    and measure_cfg.logging is not None
                    and measure_cfg.logging.wandb is not None
                ):
                    if total_local - last_wandb_check_local >= check_interval_local:
                        last_wandb_check_local = total_local
                        tl = torch.tensor([total_local], dtype=torch.long, device=device)
                        dist.all_reduce(tl, op=dist.ReduceOp.SUM)
                        global_total = int(tl.item())
                        if global_total >= next_global_log_at:
                            wandb_log_global_counts_snapshot(
                                measure_cfg,
                                device,
                                source_names,
                                counts,
                                wandb_progress_step,
                            )
                            wandb_progress_step += 1
                            next_global_log_at = (
                                global_total // wandb_log_every + 1
                            ) * wandb_log_every

                if total_local >= args_ns.max_tokens:
                    break
                if args_ns.max_documents is not None and n_docs >= args_ns.max_documents:
                    break
        finally:
            if pbar is not None:
                pbar.close()
            for it in path_to_iter.values():
                it.close()

        elapsed_total = time.perf_counter() - t_loop
        logger.info(
            "[rank %s data_rank=%s] done local_tokens=%s docs=%s in %.1fs (%.0f tok/s)",
            dist.get_rank(),
            data_rank,
            total_local,
            n_docs,
            elapsed_total,
            total_local / elapsed_total if elapsed_total > 0 else 0.0,
        )

        # --- distributed aggregation ---
        order = source_names
        local_vec = torch.tensor([counts[s] for s in order], dtype=torch.long, device=device)
        dist.all_reduce(local_vec, op=dist.ReduceOp.SUM)
        global_counts = {s: int(local_vec[i].item()) for i, s in enumerate(order)}
        total_global = sum(global_counts.values())

        weight_sum = sum(measure_cfg.data.sources.values())
        expected_frac = {s: measure_cfg.data.sources[s] / weight_sum for s in order}

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
            "data_rank": data_rank,
            "data_world_size": data_world_size,
            "local_tokens": total_local,
            "local_documents": n_docs,
            "global_total_tokens": total_global,
            "per_source": rows,
        }

        if get_is_master():
            dump_config(measure_cfg, Path(dump_dir) / "measure_config_snapshot.yaml")
            out_path = Path(dump_dir) / "measure_source_tokens.json"
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            logger.info("Wrote %s", out_path)
            print(json.dumps(out, indent=2))
            yaml_name = raw.get("name")
            log_measure_to_wandb(
                measure_cfg,
                out,
                yaml_name=yaml_name if isinstance(yaml_name, str) else None,
                max_tokens_limit=args_ns.max_tokens,
                wandb_step=wandb_progress_step,
                wandb_started_early=wandb_started_early,
            )

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    # Allow single-process runs without torchrun: train.py-style scripts expect distributed init.
    # When WORLD_SIZE is unset, setup_torch_distributed still runs via launcher defaults.
    main()
