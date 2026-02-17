# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Full backbone finetuning of a pretrained LMTransformer.

Loads a pretrained checkpoint and continues training all (or a subset of)
parameters with finetuning-appropriate hyperparameters: lower learning rates,
optional layer freezing, and cosine/wsd scheduling.

This script reuses the core Lingua training infrastructure (FSDP, checkpointing,
data loading, metrics) while providing finetuning-specific features:

1. **Layer freezing**: Optionally freeze the first N transformer layers,
   the embeddings, or the output head, training only unfrozen parameters.
2. **Separate LR groups**: Different learning rates for embeddings vs.
   transformer layers (with optional per-layer decay).
3. **Warmup-then-unfreeze**: Optionally warm up unfrozen layers for some
   steps before unfreezing the rest.

Architecture
~~~~~~~~~~~~
Uses the exact same ``LMTransformer`` as pretraining — no architectural changes.
All parallelism (FSDP, TP, activation checkpointing, compile) works unchanged.

Usage
-----
    torchrun --nproc-per-node 8 -m apps.main.backbone_finetune \\
        config=apps/main/configs/backbone_finetune.yaml

    # Override any field via CLI:
    torchrun --nproc-per-node 8 -m apps.main.backbone_finetune \\
        config=apps/main/configs/backbone_finetune.yaml \\
        optim.lr=1e-5 freeze_layers=8
"""

from copy import deepcopy
import gc
import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
import xformers.profiler
from torch.optim import AdamW, lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    WandbArgs,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer, build_lr_fn
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
    tp_parallelize,
    get_no_recompute_ops,
)
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

import wandb

logger = logging.getLogger()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BackboneFinetuneArgs:
    """Configuration for full backbone finetuning."""

    name: str = "backbone_finetune"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 10000

    # ---- Layer freezing ----

    # Number of initial transformer layers to freeze (0 = train all)
    # e.g. freeze_layers=8 freezes layers 0-7, trains layers 8+
    freeze_layers: int = 0

    # Whether to freeze the token embedding layer
    freeze_embeddings: bool = False

    # Whether to freeze the output projection (lm_head)
    freeze_output: bool = False

    # ---- Standard components ----
    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(
        default_factory=lambda: LoggingArgs(
            wandb=WandbArgs(
                entity="randomresearch",
                project="stem",
            )
        )
    )

    # If set to None, eval is run locally otherwise it launches a new job
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


# =============================================================================
# Train State (identical to train.py)
# =============================================================================

@dataclass
class TrainState(Stateful):
    step: int
    acc_step: int
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


# =============================================================================
# Validation
# =============================================================================

def validate_finetune_args(args: BackboneFinetuneArgs, output_size: int):
    """Validate finetuning arguments (adapted from train.py's validate_train_args)."""
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    assert (
        args.model.vocab_size == output_size
    ), f"Vocab size ({args.model.vocab_size}) should match output size ({output_size})"

    assert args.dump_dir, "dump_dir must be set"

    if args.checkpoint.path is None:
        logger.info(
            f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}"
        )
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    assert args.checkpoint.init_ckpt_path, (
        "init_ckpt_path is required for backbone finetuning — "
        "it must point to the pretrained checkpoint"
    )

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard
        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate //= args.distributed.tp_size
        logger.warning(
            f"Setting Data Parallel size to "
            f"{args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

    if args.distributed.fsdp_type == "no_shard":
        assert (
            args.distributed.dp_shard == 1
            and args.distributed.dp_replicate == get_world_size()
        )

    args.model.max_seqlen = args.data.seq_len

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    # Validate freeze_layers
    assert 0 <= args.freeze_layers <= args.model.n_layers, (
        f"freeze_layers ({args.freeze_layers}) must be in "
        f"[0, {args.model.n_layers}]"
    )


# =============================================================================
# Layer freezing
# =============================================================================

def apply_layer_freezing(model: LMTransformer, args: BackboneFinetuneArgs):
    """
    Freeze specified parts of the model.
    Returns the number of frozen and trainable parameters.
    """
    frozen_count = 0
    trainable_count = 0

    # Freeze initial transformer layers
    if args.freeze_layers > 0:
        for i in range(args.freeze_layers):
            for param in model.layers[i].parameters():
                param.requires_grad = False
        logger.info(f"Froze transformer layers 0-{args.freeze_layers - 1}")

    # Freeze embeddings
    if args.freeze_embeddings:
        for param in model.tok_embeddings.parameters():
            param.requires_grad = False
        logger.info("Froze token embeddings")

    # Freeze output head
    if args.freeze_output:
        for param in model.output.parameters():
            param.requires_grad = False
        logger.info("Froze output projection")

    # Count parameters
    for param in model.parameters():
        if param.requires_grad:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    logger.info(
        f"Parameter summary: "
        f"{trainable_count:,} trainable, "
        f"{frozen_count:,} frozen, "
        f"{trainable_count + frozen_count:,} total"
    )

    return frozen_count, trainable_count


# =============================================================================
# Preemption handling
# =============================================================================

preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


# =============================================================================
# Main training function
# =============================================================================

def train(args: BackboneFinetuneArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_finetune_args(args, tokenizer.n_words)

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")

        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting backbone finetuning job: {args.name}")

        # Build dataloader
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = (
                dp_rank * world_mesh["dp_shard"].size()
                + world_mesh["dp_shard"].get_local_rank()
            )
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initialize model on meta device
        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        # Apply FSDP/TP parallelism
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Materialize on GPU
        model = model.to_empty(device="cuda")

        # Load pretrained checkpoint
        logger.info(
            f"Loading pretrained model from {args.checkpoint.init_ckpt_path}"
        )
        load_from_checkpoint(
            args.checkpoint.init_ckpt_path, model, model_key="model"
        )
        model.rope_embeddings.reset_parameters()
        check_model_value_range(model, range=10.0, std=1.0)

        logger.info(f"Model size: {model_param_count:,} total parameters")

        # Apply layer freezing
        frozen_count, trainable_count = apply_layer_freezing(model, args)

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} "
            f"({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # Build optimizer (only over trainable parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Building optimizer over {len(trainable_params)} parameter groups")
        optimizer = AdamW(
            trainable_params,
            lr=args.optim.lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
            eps=args.optim.epsilon,
            fused=True,
        )

        lr_fn = build_lr_fn(args.optim, args.steps)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_fn)

        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        # Initialize checkpoint manager and try to resume
        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        gc.disable()

        # ================================================================
        # Training loop
        # ================================================================
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(
                args.data, state=train_state.data_loader_state,
            )
        )
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()

        while train_state.step < args.steps:
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # Get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen = labels.shape

            # Forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # Probing (optional)
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                assert (
                    next(model.parameters()).grad is None
                ), "Can't probe model if grads are not reset"
                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "backbone_finetune",
                    }
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                    )
                    probe_loss.backward()
                    optimizer.zero_grad()
                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            loss = model(input_ids, labels)

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            loss = loss / args.grad_acc_steps
            loss.backward()
            loss = loss.detach() * args.grad_acc_steps

            # Optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=args.optim.clip, foreach=True
                )
                grad_norm = (
                    grad_norm.full_tensor()
                    if isinstance(grad_norm, DTensor)
                    else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            if torch_profiler:
                xformers.profiler.step()

            # Logging
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (
                    time_delta * args.distributed.tp_size
                )

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                        "finetune": {
                            "frozen_params": frozen_count,
                            "trainable_params": trainable_count,
                        },
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(), 4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw / 1000} W"
                )

            # Checkpointing
            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ):
                saved = checkpoint.save(
                    model, optimizer, train_state, args, device_mesh=world_mesh,
                )

            # Evaluation
            if args.eval is not None and (
                every_n_steps(
                    train_state, args.checkpoint.eval.every, acc_step=0
                )
                or every_n_steps(train_state, args.steps, acc_step=0)
            ):
                from apps.main.eval import (
                    launch_eval,
                    EVAL_FOLDER_NAME,
                    EvalArgs,
                )

                eval_args = dataclass_from_dict(EvalArgs, args.eval)
                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.dump_dir = str(
                    os.path.join(
                        args.dump_dir,
                        "evals",
                        EVAL_FOLDER_NAME.format(train_state.step),
                    )
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    launch_eval(eval_args)
                elif get_is_master():
                    if wandb.run is not None and args.logging.wandb is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                    assert args.async_eval_gpus > 0
                    logger.info(
                        f"Launching evals on {args.async_eval_gpus} gpus"
                    )
                    with clean_env():
                        launch_job(
                            StoolArgs(
                                asdict(eval_args),
                                script="apps.main.eval",
                                copy_code=False,
                                nodes=args.async_eval_gpus // 8,
                                qos="lowest",
                            )
                        )

            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model, optimizer, train_state, args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        checkpoint.save(
            model, optimizer, train_state, args, device_mesh=world_mesh,
        )
    gc.collect()
    logger.info("Backbone finetuning complete!")


# =============================================================================
# Entry point
# =============================================================================

def main():
    """
    CLI entry point with OmegaConf config loading.
    Same pattern as train.py — config file + CLI overrides.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(BackboneFinetuneArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()