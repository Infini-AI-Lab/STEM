# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
LoRA (Low-Rank Adaptation) finetuning of a pretrained LMTransformer.

Loads a pretrained checkpoint, attaches LoRA adapters via forward hooks on
designated linear layers, freezes all base model weights, and trains only the
LoRA parameters. This provides a parameter-efficient finetuning baseline.

Key design: **Hook-based LoRA**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rather than replacing nn.Linear modules (which would break FSDP parameter
registration), we register forward hooks on target Linear layers that add
the low-rank LoRA output to the frozen base output. This mirrors the
approach used in ``stem_adapter_finetune.py`` and is fully compatible with
FSDP, activation checkpointing, and all existing distributed infrastructure.

For each target linear layer ``W``, the hook computes:

    output = W(x) + (alpha / rank) * B(A(dropout(x)))

where ``W`` is frozen, and ``A`` (in -> rank), ``B`` (rank -> out) are
trainable. At initialization, ``B`` is zero so the LoRA contribution is
zero and the model starts from the pretrained state.

Usage
-----
    torchrun --nproc-per-node 8 -m apps.main.lora_finetune \
        config=apps/main/configs/lora_finetune.yaml

    # Override any field via CLI:
    torchrun --nproc-per-node 8 -m apps.main.lora_finetune \
        config=apps/main/configs/lora_finetune.yaml \
        lora_rank=16 lora_alpha=32 lora_lr=3e-4
"""

from copy import deepcopy
import gc
import json
import logging
import math
import os
import re
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import xformers.profiler
from torch.optim import AdamW, lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import (
    CheckpointArgs,
    CheckpointManager,
    load_from_checkpoint,
    FOLDER_NAME,
    RE_FOLDER,
    CONFIG_NAME,
    TRAIN_STATE_NAME,
)
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
    get_num_params,
)
from lingua.optim import OptimArgs, build_lr_fn
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
)
from lingua.stool import StoolArgs, launch_job

import wandb

logger = logging.getLogger()


# =============================================================================
# Hook-Based LoRA Manager
# =============================================================================

LORA_TARGET_MODULES = {
    "attention_q": "attention.wq",
    "attention_k": "attention.wk",
    "attention_v": "attention.wv",
    "attention_o": "attention.wo",
    "ffn_w1": "feed_forward.w1",
    "ffn_w2": "feed_forward.w2",
    "ffn_w3": "feed_forward.w3",
}


class LoRAAdapter(nn.Module):
    """A single LoRA adapter: A (down) and B (up) projection.
    Computes: scaling * B(A(dropout(x)))
    B initialized to zero so initial LoRA contribution is zero."""

    def __init__(self, in_features, out_features, rank, alpha, dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))


class LoRAManager(nn.Module):
    """Manages LoRA adapters via forward hooks (FSDP-safe)."""

    def __init__(self, model, target_modules, rank=8, alpha=16.0, dropout=0.0,
                 lora_layers=None, device=None, dtype=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.rank = rank
        self.alpha = alpha

        target_paths = []
        for key in target_modules:
            if key in LORA_TARGET_MODULES:
                target_paths.append(LORA_TARGET_MODULES[key])
            else:
                raise ValueError(f"Unknown target: {key!r}. Valid: {list(LORA_TARGET_MODULES.keys())}")

        n_layers = len(model.layers)
        layer_indices = lora_layers if lora_layers is not None else list(range(n_layers))

        self.adapters = nn.ModuleDict()
        self._hooks = []

        for layer_idx in layer_indices:
            assert 0 <= layer_idx < n_layers
            layer = model.layers[layer_idx]

            for target_path in target_paths:
                full_path = f"layers.{layer_idx}.{target_path}"
                try:
                    parts = target_path.split(".")
                    target_module = layer
                    for part in parts:
                        target_module = getattr(target_module, part)
                except AttributeError:
                    logger.warning(f"Target {full_path} not found, skipping")
                    continue

                if not isinstance(target_module, nn.Linear):
                    logger.warning(f"{full_path} is not nn.Linear, skipping")
                    continue

                adapter_key = full_path.replace(".", "_")
                adapter = LoRAAdapter(
                    target_module.in_features, target_module.out_features,
                    rank, alpha, dropout, device, dtype,
                )
                self.adapters[adapter_key] = adapter

                hook = target_module.register_forward_hook(self._make_hook(adapter_key))
                self._hooks.append(hook)
                logger.info(f"Attached LoRA at {full_path} ({target_module.in_features}->{target_module.out_features}, r={rank})")

        logger.info(f"LoRA manager: {len(self.adapters)} adapters, {sum(p.numel() for p in self.parameters()):,} params")

    def _make_hook(self, adapter_key):
        def hook_fn(module, input, output):
            x = input[0]
            adapter = self.adapters[adapter_key]
            lora_out = adapter(x.to(dtype=adapter.lora_A.weight.dtype))
            return output + lora_out.to(dtype=output.dtype)
        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_lora_state_dict(self):
        return {name: param.data.cpu() for name, param in self.adapters.named_parameters()}

    def load_lora_state_dict(self, state_dict):
        own_state = dict(self.adapters.named_parameters())
        for name, tensor in state_dict.items():
            if name in own_state:
                own_state[name].data.copy_(tensor.to(own_state[name].device, own_state[name].dtype))

    def merge_into_model(self, model):
        """Merge LoRA weights into base model linear layers."""
        n_merged = 0
        for adapter_key, adapter in self.adapters.items():
            parts = adapter_key.split("_")
            try:
                layer_idx = int(parts[1])
                sub_parts = parts[2:]
                # Reconstruct sub-path: attention_wq -> attention.wq, feed_forward_w1 -> feed_forward.w1
                sub_path = []
                i = 0
                while i < len(sub_parts):
                    if i + 1 < len(sub_parts) and sub_parts[i] == "feed" and sub_parts[i+1] == "forward":
                        sub_path.append("feed_forward")
                        i += 2
                    else:
                        sub_path.append(sub_parts[i])
                        i += 1
                target = model.layers[layer_idx]
                for p in sub_path:
                    target = getattr(target, p)
            except (ValueError, AttributeError, IndexError):
                logger.warning(f"Cannot find base module for {adapter_key}, skipping")
                continue

            if isinstance(target, nn.Linear):
                with torch.no_grad():
                    delta = (adapter.scaling * (adapter.lora_B.weight @ adapter.lora_A.weight)).to(target.weight.dtype, target.weight.device)
                    target.weight.add_(delta)
                    n_merged += 1

        logger.info(f"Merged {n_merged} LoRA adapters")
        self.remove_hooks()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LoRAFinetuneArgs:
    name: str = "lora_finetune"
    dump_dir: str = ""
    seed: int = 42
    steps: int = 10000
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000

    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["attention_q", "attention_v"])
    lora_layers: Optional[List[int]] = None

    lora_lr: float = 3e-4
    lora_weight_decay: float = 0.0
    lora_beta1: float = 0.9
    lora_beta2: float = 0.95
    lora_epsilon: float = 1e-8
    lora_clip: float = 1.0

    scheduler: str = "cosine"
    warmup_steps: int = 200
    lr_min_ratio: float = 0.01

    data: DataArgs = field(default_factory=DataArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


@dataclass
class LoRATrainState(Stateful):
    step: int
    acc_step: int
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self):
        return {"step": self.step, "acc_step": self.acc_step,
                "data_loader_state": self.data_loader_state,
                "scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_lora_checkpoint(lora_manager, optimizer, train_state, args, ckpt_dir):
    if get_is_master():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        torch.save(lora_manager.get_lora_state_dict(), ckpt_dir / "lora_weights.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "lora_optimizer.pt")
        ts_dict = train_state.state_dict()
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(ts_dict, f, default=lambda obj: obj.tolist() if isinstance(obj, torch.Tensor) else obj)
        with open(ckpt_dir / CONFIG_NAME, "w") as f:
            json.dump(OmegaConf.to_container(OmegaConf.structured(args), resolve=True), f)
        logger.info(f"Saved LoRA checkpoint to {ckpt_dir}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_lora_checkpoint(lora_manager, optimizer, train_state, ckpt_dir):
    if not ckpt_dir.exists():
        return False
    lora_path = ckpt_dir / "lora_weights.pt"
    if lora_path.exists():
        lora_manager.load_lora_state_dict(torch.load(lora_path, map_location="cpu", weights_only=True))
        logger.info(f"Loaded LoRA weights from {lora_path}")
    optim_path = ckpt_dir / "lora_optimizer.pt"
    if optim_path.exists():
        try:
            optimizer.load_state_dict(torch.load(optim_path, map_location="cpu", weights_only=True))
        except Exception as e:
            logger.warning(f"Could not load optimizer: {e}")
    ts_path = ckpt_dir / "train_state.json"
    if ts_path.exists():
        with open(ts_path) as f:
            train_state.load_state_dict(json.load(f))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return True


# =============================================================================
# Preemption & utilities
# =============================================================================

preemption_flag = dict(flag=False)

def set_preemption_flag(signum, frame):
    logger.warning(f"Signal {signum}: preemption, checkpointing ASAP")
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

def train(args_dict):
    args = args_dict

    with ExitStack() as context_stack:
        # 1. Setup
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        if args.model.vocab_size < 0:
            args.model.vocab_size = tokenizer.n_words
        assert args.model.vocab_size == tokenizer.n_words

        assert args.dump_dir, "dump_dir must be set"
        if args.checkpoint.path is None:
            args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")
        args.model.max_seqlen = args.data.seq_len
        assert args.checkpoint.init_ckpt_path, "init_ckpt_path required for LoRA finetuning"

        if (args.distributed.dp_replicate * args.distributed.dp_shard * args.distributed.tp_size != get_world_size()):
            assert get_world_size() % args.distributed.dp_shard == 0
            args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard
            assert args.distributed.dp_replicate % args.distributed.tp_size == 0
            args.distributed.dp_replicate //= args.distributed.tp_size

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")

        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting LoRA finetuning: {args.name}")

        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        # 2. Build & load model
        torch.manual_seed(args.seed)
        with torch.device("meta"):
            model = LMTransformer(args.model)

        model_param_count = get_num_params(model)
        saved_compile = args.distributed.compile
        args.distributed.compile = False
        model = parallelize_model(model, world_mesh, args.model, args.distributed,
                                  fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
                                  tp_parallelize=None, no_recompute_ops=get_no_recompute_ops())
        args.distributed.compile = saved_compile
        model = model.to_empty(device="cuda")

        logger.info(f"Loading pretrained model from {args.checkpoint.init_ckpt_path}")
        load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
        model.rope_embeddings.reset_parameters()
        check_model_value_range(model, range=10.0, std=1.0)
        logger.info(f"Model loaded: {model_param_count:,} params")

        # 3. Freeze base & create LoRA
        for param in model.parameters():
            param.requires_grad = False

        device = torch.device("cuda", torch.cuda.current_device())
        lora_manager = LoRAManager(model, args.lora_target_modules, args.lora_rank,
                                   args.lora_alpha, args.lora_dropout, args.lora_layers,
                                   device, torch.float32)
        for param in lora_manager.parameters():
            param.requires_grad = True

        if dp_degree > 1:
            for param in lora_manager.parameters():
                torch.distributed.broadcast(param.data, src=0)
            logger.info("Synced LoRA params across DP ranks")

        lora_param_count = sum(p.numel() for p in lora_manager.parameters())
        logger.info(f"LoRA: {lora_param_count:,} trainable / {model_param_count:,} frozen ({lora_param_count/model_param_count*100:.3f}%)")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")

        # 4. Optimizer
        lora_params = list(lora_manager.parameters())
        optimizer = AdamW(lora_params, lr=args.lora_lr, betas=(args.lora_beta1, args.lora_beta2),
                          weight_decay=args.lora_weight_decay, eps=args.lora_epsilon, fused=False)
        optim_args = OptimArgs(lr=args.lora_lr, scheduler=args.scheduler,
                               warmup=args.warmup_steps, lr_min_ratio=args.lr_min_ratio)
        lr_fn = build_lr_fn(optim_args, args.steps)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_fn)

        # 5. Data & state
        data_loader_state = init_dataloader_state_from_args(args.data, dp_rank, dp_degree)
        train_state = LoRATrainState(step=0, acc_step=0, data_loader_state=data_loader_state, scheduler=scheduler)

        # 6. Resume
        ckpt_base = Path(args.checkpoint.path)
        if get_is_master():
            ckpt_base.mkdir(parents=True, exist_ok=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        existing = sorted([d for d in ckpt_base.iterdir() if d.is_dir() and re.match(RE_FOLDER, d.name)],
                          key=lambda p: int(p.name)) if ckpt_base.exists() else []
        if existing:
            load_lora_checkpoint(lora_manager, optimizer, train_state, existing[-1])

        gc.disable()

        # 7. Training loop
        model.train()
        metric_logger = context_stack.enter_context(MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args))
        data_loader = context_stack.enter_context(build_dataloader_from_args(args.data, state=train_state.data_loader_state))
        torch_profiler = context_stack.enter_context(maybe_run_profiler(args.dump_dir, model, args.profiling))

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        logger.info(f"Training from step {train_state.step} to {args.steps}")

        saved = False
        while train_state.step < args.steps:
            train_state.acc_step = (train_state.acc_step + 1) % args.grad_acc_steps

            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            loss = model(input_ids, labels)
            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            loss.backward()
            loss_for_log = loss.detach() * (args.grad_acc_steps if args.grad_acc_steps > 1 else 1)

            grad_norm = -1.0
            if train_state.acc_step == 0:
                if dp_degree > 1:
                    for param in lora_params:
                        if param.grad is not None:
                            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)

                lora_grads = [p for p in lora_params if p.grad is not None]
                if lora_grads:
                    grad_norm = torch.nn.utils.clip_grad_norm_(lora_grads, max_norm=args.lora_clip, foreach=False).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            if torch_profiler:
                xformers.profiler.step()

            if every_n_steps(train_state, args.logging.freq, acc_step=None if args.logging.acc_freq else 0, acc_freq=args.logging.acc_freq):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)
                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
                total_acc_steps = args.grad_acc_steps * train_state.step + train_state.acc_step
                total_tokens = dp_degree * total_acc_steps * args.data.batch_size * args.data.seq_len

                metrics = flatten_dict({
                    "global_step": train_state.step, "acc_step": train_state.acc_step,
                    "speed": {"wps": wps, "curr_iter_time": curr_iter_time, "data_load_time": data_load_time},
                    "optim": {"grad_norm": grad_norm, "lr": curr_lr, "total_tokens": total_tokens},
                    "memory": gpu_mem_stats._asdict(),
                    "lora": {"rank": args.lora_rank, "trainable_params": lora_param_count},
                }, sep="/")
                to_sync = {"loss/out": loss_for_log.item()}
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(f"step: {train_state.step}  loss: {loss_for_log.item():.4f}  grad: {grad_norm:.2e}  wps: {wps:.2e}  iter: {curr_iter_time:>7}  lr: {curr_lr:.2e}  mem: {gpu_mem_stats.max_active_pct:.0f}%")

            saved = False
            if every_n_steps(train_state, args.checkpoint.dump.every, acc_step=0):
                save_lora_checkpoint(lora_manager, optimizer, train_state, args, ckpt_base / FOLDER_NAME.format(train_state.step))
                saved = True

            if args.eval is not None and every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                from apps.main.eval import launch_eval, EVAL_FOLDER_NAME, EvalArgs
                eval_args = dataclass_from_dict(EvalArgs, args.eval)
                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(ckpt_base / FOLDER_NAME.format(train_state.step))
                eval_args.dump_dir = str(os.path.join(args.dump_dir, "evals", EVAL_FOLDER_NAME.format(train_state.step)))
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    launch_eval(eval_args)
                elif get_is_master():
                    if wandb.run is not None and args.logging.wandb is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                    with clean_env():
                        launch_job(StoolArgs(asdict(eval_args), script="apps.main.eval", copy_code=False, nodes=args.async_eval_gpus // 8, qos="lowest"))

            if preemption_flag["flag"]:
                if not saved:
                    save_lora_checkpoint(lora_manager, optimizer, train_state, args, ckpt_base / FOLDER_NAME.format(train_state.step))
                requeue_slurm_job()
                sys.exit(0)

        if not saved:
            save_lora_checkpoint(lora_manager, optimizer, train_state, args, ckpt_base / FOLDER_NAME.format(train_state.step))

        lora_manager.remove_hooks()

    gc.collect()
    logger.info(f"LoRA finetuning complete! {train_state.step} steps, {lora_param_count:,} LoRA params")


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    default_cfg = OmegaConf.structured(LoRAFinetuneArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    train(cfg)

if __name__ == "__main__":
    main()