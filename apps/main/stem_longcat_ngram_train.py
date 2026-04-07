# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Toy causal LM with :class:`lingua.longcat_ngram.NgramEmbedding` for DDP smoke
# tests (no Hugging Face Longcat). See plan: Longcat N-gram embedding Phase 1.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from lingua.args import dump_config
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

logger = logging.getLogger()


class ToyLongcatNgramLM(nn.Module):
    """Token embeddings + n-gram enrichment + linear LM head."""

    def __init__(self, cfg: LongcatNgramConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id
        )
        self.ngram = NgramEmbedding(cfg, self.tok_embeddings)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.ngram(input_ids, ngram_context=None)
        return self.lm_head(h)


@dataclass
class LongcatNgramTrainOptimArgs:
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass
class StemLongcatNgramTrainArgs:
    name: str = "stem_longcat_ngram_train"
    dump_dir: str = ""
    seed: int = 42
    steps: int = 1000
    batch_size: int = 4
    seq_len: int = 64
    grad_acc_steps: int = 1
    log_interval: int = 50
    checkpoint_interval: int = 500

    model: LongcatNgramConfig = field(
        default_factory=lambda: LongcatNgramConfig(
            vocab_size=256,
            hidden_size=32,
            pad_token_id=0,
            eos_token_id=1,
            emb_neighbor_num=3,
            emb_split_num=2,
            ngram_vocab_size_ratio=0.5,
        )
    )
    optim: LongcatNgramTrainOptimArgs = field(
        default_factory=LongcatNgramTrainOptimArgs
    )
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)


def _synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    g: torch.Generator,
) -> torch.Tensor:
    x = torch.randint(
        low=2,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        generator=g,
    )
    x[:, 0] = pad_id
    return x


def _causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, pad_id: int
) -> torch.Tensor:
    # logits (B, S, V), labels (B, S) next-token aligned with logits[:, :-1]
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=pad_id,
    )


def _sync_distributed_mean(loss: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or get_world_size() <= 1:
        return loss
    t = loss.detach().clone()
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def train(args: StemLongcatNgramTrainArgs) -> None:
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
        dump_config(args, Path(args.dump_dir) / "config.yaml")
    if dist.is_initialized():
        dist.barrier()

    init_logger(Path(args.dump_dir) / "train.log")
    torch.manual_seed(args.seed + rank)
    g = torch.Generator(device=device)
    g.manual_seed(args.seed + rank)

    model = ToyLongcatNgramLM(args.model).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        weight_decay=args.optim.weight_decay,
    )
    pad_id = args.model.pad_token_id

    step = 0
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for _ in range(args.grad_acc_steps):
            batch = _synthetic_batch(
                args.batch_size,
                args.seq_len,
                args.model.vocab_size,
                pad_id,
                device,
                g,
            )
            logits = model(batch)
            loss = _causal_lm_loss(logits, batch, pad_id) / args.grad_acc_steps
            loss.backward()
            total_loss = total_loss + loss.detach()

        opt.step()
        step += 1

        avg_loss = _sync_distributed_mean(total_loss)
        if get_is_master() and step % args.log_interval == 0:
            logger.info(
                "step %d loss %.4f ppl %.2f",
                step,
                avg_loss.item(),
                float(torch.exp(avg_loss).item()),
            )

        if (
            get_is_master()
            and args.checkpoint_interval > 0
            and step % args.checkpoint_interval == 0
        ):
            to_save = model.module.state_dict() if world > 1 else model.state_dict()
            ckpt = {
                "model": to_save,
                "step": step,
                "model_config": OmegaConf.to_container(
                    OmegaConf.structured(args.model), resolve=True
                ),
            }
            path = Path(args.dump_dir) / f"checkpoint_{step:06d}.pt"
            torch.save(ckpt, path)
            logger.info("wrote %s", path)

    if get_is_master():
        to_save = model.module.state_dict() if world > 1 else model.state_dict()
        final = {
            "model": to_save,
            "step": step,
            "model_config": OmegaConf.to_container(
                OmegaConf.structured(args.model), resolve=True
            ),
        }
        torch.save(final, Path(args.dump_dir) / "checkpoint_last.pt")
        logger.info("wrote checkpoint_last.pt")


def main() -> None:
    cli = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli.config) if "config" in cli else OmegaConf.create()
    if "config" in cli:
        del cli.config
    default = OmegaConf.structured(StemLongcatNgramTrainArgs())
    cfg = OmegaConf.merge(default, file_cfg, cli)
    args = OmegaConf.to_object(cfg)
    train(args)


if __name__ == "__main__":
    main()
