# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Validation CE / perplexity for :mod:`stem_longcat_ngram_train` checkpoints.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

from lingua.args import dataclass_from_dict, dump_config
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    get_global_rank,
    get_is_master,
    get_local_rank,
    get_world_size,
    setup_env,
    setup_torch_distributed,
)
from lingua.logger import init_logger
from lingua.longcat_ngram import LongcatNgramConfig

from apps.main.stem_longcat_ngram_train import ToyLongcatNgramLM, _synthetic_batch

logger = logging.getLogger()


def _causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, pad_id: int
) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=pad_id,
    )


@dataclass
class StemLongcatNgramEvalArgs:
    name: str = "stem_longcat_ngram_eval"
    dump_dir: str = ""
    checkpoint_path: str = ""
    seed: int = 42
    batches: int = 20
    batch_size: int = 4
    seq_len: int = 64

    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def eval_run(args: StemLongcatNgramEvalArgs) -> None:
    setup_env(args.env)
    _ws = get_world_size()
    args.distributed.dp_replicate = _ws
    args.distributed.dp_shard = 1
    args.distributed.tp_size = 1
    args.distributed.fsdp_type = "no_shard"

    setup_torch_distributed(args.distributed)
    rank = get_global_rank()
    world = get_world_size()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if get_is_master():
        os.makedirs(args.dump_dir, exist_ok=True)
        dump_config(args, Path(args.dump_dir) / "eval_config.yaml")
    if dist.is_initialized():
        dist.barrier()

    init_logger(Path(args.dump_dir) / "eval.log")

    ckpt = _load_checkpoint(Path(args.checkpoint_path))
    cfg_dict = ckpt["model_config"]
    model_cfg = dataclass_from_dict(LongcatNgramConfig, cfg_dict)

    model = ToyLongcatNgramLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    pad_id = model_cfg.pad_token_id
    g = torch.Generator(device=device)
    g.manual_seed(args.seed + rank)

    total_n = 0
    total_loss = torch.zeros((), device=device)
    with torch.no_grad():
        for _ in range(args.batches):
            batch = _synthetic_batch(
                args.batch_size,
                args.seq_len,
                model_cfg.vocab_size,
                pad_id,
                device,
                g,
            )
            logits = model(batch)
            loss = _causal_lm_loss(logits, batch, pad_id)
            n = (batch[:, 1:] != pad_id).sum()
            total_loss = total_loss + loss * n
            total_n += int(n.item())

    denom = torch.tensor(float(total_n), device=device)
    if dist.is_initialized() and world > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(denom, op=dist.ReduceOp.SUM)

    mean_ce = (total_loss / denom.clamp(min=1.0)).item()
    ppl = float(torch.exp(torch.tensor(mean_ce)).item())

    if get_is_master():
        logger.info("eval mean_nll %.4f perplexity %.2f (tokens=%d)", mean_ce, ppl, int(denom.item()))


def main() -> None:
    cli = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli.config) if "config" in cli else OmegaConf.create()
    if "config" in cli:
        del cli.config
    default = OmegaConf.structured(StemLongcatNgramEvalArgs())
    cfg = OmegaConf.merge(default, file_cfg, cli)
    args = OmegaConf.to_object(cfg)
    eval_run(args)


if __name__ == "__main__":
    main()
