# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Adapter-style finetuning: STEM embeddings in parallel with dense FFN up-projection.

Instead of *replacing* the FFN's w3 up-projection with STEM embeddings (as in
standard STEM), this script adds STEM embeddings as a **parallel adapter** that
is blended with the original w3 output via a learnable gating scalar ``alpha``:

    x3_adapted = (1 - sigmoid(alpha)) * w3(x) + sigmoid(alpha) * stem_emb(token_ids)

Key features
------------
1. **Learnable gating (alpha)**: Per-layer scalar, initialized so that
   sigmoid(alpha_init) ≈ 0 (i.e. alpha_init = -5), preserving pretrained
   behavior at initialization. The sigmoid gating ensures smooth, bounded
   interpolation in [0, 1].

2. **Optional CPU-offloaded optimizer**: STEM embedding optimizer states
   (momentum and variance buffers) can be kept on CPU to free GPU memory.
   Weights remain on GPU for the forward/backward pass; only the optimizer
   step is performed on CPU.

3. **Warmup mode**: Optionally freeze the entire pretrained backbone and
   train only the STEM embeddings + gating alphas for a configurable number
   of warmup steps before unfreezing everything.

4. **Multiple initialization strategies**: Xavier uniform, Xavier normal,
   or small-std normal (std=0.02).

Architecture
~~~~~~~~~~~~
For each designated FFN layer ``i``, we register a forward hook on ``w3``
that blends its output with the STEM embedding lookup:

    w3_output = w3(x)                           # original up-projection
    stem_output = stem_emb_i(token_ids)          # embedding lookup
    blended = (1 - gate_i) * w3_output + gate_i * stem_output  # adapter

This hook-based approach avoids modifying any existing model classes and
works seamlessly with FSDP wrapping.

Usage
-----
    torchrun --nproc-per-node 8 -m apps.main.stem_adapter_finetune \\
        config=apps/main/configs/stem_adapter_finetune.yaml

    # Override any config via CLI:
    torchrun --nproc-per-node 8 -m apps.main.stem_adapter_finetune \\
        config=apps/main/configs/stem_adapter_finetune.yaml \\
        warmup_steps=500 stem_init_type=xavier_normal stem_lr=1e-3
"""

import gc
import json
import logging
import math
import os
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, load_from_checkpoint
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
)
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_lr_fn
from lingua.logger import init_logger
from lingua.tokenizer import build_tokenizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.stem_dist_utils import (
    ParallelEmbedding,
    initialize_stem_process_group,
    get_stem_data_parallel_group,
    get_stem_data_parallel_rank,
    get_stem_data_parallel_world_size,
    get_stem_model_parallel_rank,
)
from lingua.stem_checkpoint import (
    save_stem_shards,
    load_stem_shards,
    extract_stem_state_dict,
)

from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token,
)
from apps.main.train import TrainState, validate_train_args, every_n_steps

import wandb

logger = logging.getLogger()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StemAdapterFinetuneArgs:
    """Configuration for STEM adapter finetuning."""

    name: str = "stem_adapter_finetune"
    dump_dir: str = ""
    seed: int = 42

    # Total optimizer steps (warmup_steps + post-warmup steps)
    steps: int = 10000

    # Gradient accumulation
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000

    # ---- Adapter configuration ----

    # Which transformer layers get a parallel STEM embedding adapter
    stem_layers: List[int] = field(
        default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15]
    )

    # STEM embedding initialization: "xavier_uniform", "xavier_normal", "normal_0.02"
    stem_init_type: str = "xavier_uniform"

    # Alpha (gating scalar) initialization value.
    # alpha is passed through sigmoid: sigmoid(alpha_init) ≈ gate value at init.
    # Default: -5.0  =>  sigmoid(-5) ≈ 0.0067  ≈  0  (nearly pure w3 at start)
    alpha_init: float = -5.0

    # ---- Warmup configuration ----

    # Whether to use a warmup phase where the backbone is frozen
    use_warmup: bool = True

    # Number of warmup steps (backbone frozen, only STEM + alphas trained)
    # Only used when use_warmup=True
    warmup_steps: int = 1000

    # ---- Optimizer: STEM embeddings ----

    stem_lr: float = 1e-3
    stem_weight_decay: float = 0.0
    stem_beta1: float = 0.9
    stem_beta2: float = 0.95
    stem_epsilon: float = 1e-8
    stem_clip: float = 1.0

    # ---- Optimizer: alpha parameters ----

    alpha_lr: float = 1e-2
    alpha_weight_decay: float = 0.0

    # ---- Optimizer: backbone (post-warmup) ----

    backbone_lr: float = 2e-5
    backbone_weight_decay: float = 0.033
    backbone_clip: float = 1.0

    # ---- LR schedule (shared across all param groups) ----

    scheduler: str = "cosine"
    warmup_lr_steps: int = 500  # LR warmup steps (distinct from adapter warmup)
    lr_min_ratio: float = 0.01

    # ---- CPU offloading ----

    # Offload STEM embedding optimizer states to CPU.
    # Saves GPU memory (2x model-state savings for AdamW) at the cost of
    # PCIe transfer overhead per optimizer step.
    cpu_offload_stem_optim: bool = False

    # ---- Standard training infrastructure ----

    data: DataArgs = field(default_factory=DataArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)


# =============================================================================
# Train State
# =============================================================================

@dataclass
class AdapterTrainState(Stateful):
    """Tracks training progress for the adapter finetuning loop."""

    step: int
    acc_step: int
    scheduler_states: Dict[str, Any]
    data_loader_state: PackTokensState
    is_warmup: bool  # Whether we are currently in warmup phase

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "scheduler_states": self.scheduler_states,
            "data_loader_state": self.data_loader_state,
            "is_warmup": self.is_warmup,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.scheduler_states = state_dict.get("scheduler_states", {})
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.is_warmup = state_dict.get("is_warmup", False)


# =============================================================================
# CPU-Offloaded AdamW for STEM Embeddings
# =============================================================================

class CPUOffloadedAdamW:
    """
    AdamW optimizer that keeps momentum/variance buffers on CPU to save GPU memory.

    Workflow per step:
        1. Copy gradients from GPU parameters to CPU shadow gradients
        2. Run AdamW update on CPU shadow parameters
        3. Copy updated CPU shadow parameters back to GPU

    This saves ~2x the parameter memory on GPU (AdamW stores momentum + variance),
    at the cost of PCIe transfer latency per optimizer step.

    Note on FLOPs:
        The GPU FLOP cost of STEM embedding forward/backward passes is minimal
        (embedding lookups are memory-bound, not compute-bound). The primary
        benefit of CPU offloading is **GPU memory savings**, not FLOP reduction.
        For training with large vocabularies (128K+) and many STEM layers,
        offloading can save 10-40 GB of GPU memory.
    """

    def __init__(
        self,
        gpu_params: List[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        self.gpu_params = list(gpu_params)
        self.lr = lr

        # Create CPU shadow copies of parameters
        self.cpu_params = []
        for p in self.gpu_params:
            cpu_p = torch.nn.Parameter(
                p.data.detach().clone().to(device="cpu", dtype=torch.float32),
                requires_grad=True,
            )
            self.cpu_params.append(cpu_p)

        # Create CPU optimizer operating on CPU parameters
        self.cpu_optimizer = AdamW(
            self.cpu_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            fused=False,
        )

        # Pinned memory buffers for async GPU→CPU grad transfers
        self._pinned_grads = []
        for p in self.gpu_params:
            self._pinned_grads.append(
                torch.empty_like(p.data, device="cpu", dtype=torch.float32).pin_memory()
            )

    @property
    def param_groups(self):
        return self.cpu_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients on GPU parameters."""
        for p in self.gpu_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        self.cpu_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        """
        1. Copy GPU gradients → CPU (via pinned memory for speed)
        2. AdamW step on CPU
        3. Copy updated CPU weights → GPU
        """
        for i, (gpu_p, cpu_p) in enumerate(zip(self.gpu_params, self.cpu_params)):
            if gpu_p.grad is not None:
                # GPU grad → pinned buffer → CPU param.grad
                self._pinned_grads[i].copy_(gpu_p.grad.data.float(), non_blocking=False)
                cpu_p.grad = self._pinned_grads[i].clone()
            else:
                cpu_p.grad = None

        # Run optimizer on CPU
        self.cpu_optimizer.step()

        # Copy updated weights back to GPU
        for gpu_p, cpu_p in zip(self.gpu_params, self.cpu_params):
            gpu_p.data.copy_(cpu_p.data.to(dtype=gpu_p.dtype), non_blocking=True)

    def state_dict(self):
        return self.cpu_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.cpu_optimizer.load_state_dict(state_dict)
        # After loading, sync CPU params back to GPU
        for gpu_p, cpu_p in zip(self.gpu_params, self.cpu_params):
            gpu_p.data.copy_(cpu_p.data.to(dtype=gpu_p.dtype))


# =============================================================================
# STEM Adapter Manager
# =============================================================================

class StemAdapterManager(nn.Module):
    """
    Manages STEM embedding adapters and alpha gating parameters.

    For each designated layer, registers a forward hook on w3 that blends
    the original w3 output with the STEM embedding lookup:

        blended = (1 - sigmoid(alpha)) * w3(x) + sigmoid(alpha) * stem_emb(token_ids)

    Attributes:
        stem_embeddings (nn.ModuleList): STEM embedding tables per adapter layer.
        alphas (nn.ParameterList): Learnable gating scalars per adapter layer.
    """

    def __init__(
        self,
        model: LMTransformer,
        stem_layer_indices: List[int],
        vocab_size: int,
        stem_embedding_dim: int,
        init_type: str = "xavier_uniform",
        alpha_init: float = -5.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.stem_layer_indices = stem_layer_indices
        self.vocab_size = vocab_size
        self.stem_embedding_dim = stem_embedding_dim
        self.alpha_init = alpha_init

        # Map layer_idx → adapter index
        self._layer_to_idx = {
            layer_idx: i for i, layer_idx in enumerate(stem_layer_indices)
        }

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())

        # Create STEM embeddings
        self.stem_embeddings = nn.ModuleList([
            ParallelEmbedding(vocab_size, stem_embedding_dim, device=device)
            for _ in range(len(stem_layer_indices))
        ])

        # Create learnable alpha gating parameters (one scalar per layer)
        # sigmoid(alpha_init) determines initial gate value
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(alpha_init, device=device, dtype=torch.float32))
            for _ in range(len(stem_layer_indices))
        ])

        # Initialize STEM embeddings
        self._init_stem_embeddings(init_type, device)

        # Hooks will be registered separately via register_hooks()
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._current_input_ids: Optional[torch.Tensor] = None

    def _init_stem_embeddings(self, init_type: str, device: torch.device):
        """Initialize STEM embedding weights."""
        for i, emb in enumerate(self.stem_embeddings):
            weight = emb.weight
            if weight.device.type == "meta":
                logger.warning(f"stem_embeddings[{i}] is on meta device, skipping init")
                continue

            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(weight.data)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(weight.data)
            elif init_type == "normal_0.02":
                nn.init.normal_(weight.data, mean=0.0, std=0.02)
            else:
                raise ValueError(
                    f"Unknown stem_init_type: {init_type!r}. "
                    f"Must be one of: 'xavier_uniform', 'xavier_normal', 'normal_0.02'"
                )

            logger.info(
                f"stem_embeddings[{i}] (layer {self.stem_layer_indices[i]}): "
                f"init={init_type}, norm={weight.norm().item():.4f}, "
                f"shape={tuple(weight.shape)}"
            )

    def set_input_ids(self, input_ids: torch.Tensor):
        """Set the current input token IDs (called before each forward pass)."""
        self._current_input_ids = input_ids

    def _make_adapter_hook(self, adapter_idx: int):
        """Create a forward hook for the w3 linear layer at adapter index."""
        def hook_fn(module, inp, out):
            # out: w3(x) — shape [B, S, hidden_dim]
            input_ids = self._current_input_ids
            if input_ids is None:
                raise RuntimeError(
                    "StemAdapterManager: input_ids not set before forward pass. "
                    "Call set_input_ids() before model.forward()."
                )

            # Get STEM embedding lookup
            stem_out = self.stem_embeddings[adapter_idx](input_ids)  # [B, S, hidden_dim_shard]

            # Ensure dtype compatibility (model may be bf16, embeddings may be fp32)
            stem_out = stem_out.to(dtype=out.dtype)

            # Compute gate: sigmoid(alpha) ∈ [0, 1]
            gate = torch.sigmoid(self.alphas[adapter_idx])

            # Blend: (1 - gate) * w3_out + gate * stem_out
            blended = (1.0 - gate) * out + gate * stem_out

            return blended

        return hook_fn

    def register_hooks(self, model: LMTransformer):
        """Register forward hooks on w3 of each designated layer."""
        self.remove_hooks()  # Clean up any existing hooks

        for adapter_idx, layer_idx in enumerate(self.stem_layer_indices):
            ffn = model.layers[layer_idx].feed_forward
            assert hasattr(ffn, "w3"), (
                f"Layer {layer_idx} FeedForward has no w3 — "
                f"is this a standard LMTransformer?"
            )
            hook = ffn.w3.register_forward_hook(
                self._make_adapter_hook(adapter_idx)
            )
            self._hooks.append(hook)
            logger.info(
                f"Registered adapter hook on layer {layer_idx}.feed_forward.w3 "
                f"(adapter_idx={adapter_idx})"
            )

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_gate_values(self) -> Dict[int, float]:
        """Return current sigmoid(alpha) gate values per layer."""
        with torch.no_grad():
            return {
                layer_idx: torch.sigmoid(self.alphas[i]).item()
                for i, layer_idx in enumerate(self.stem_layer_indices)
            }

    def get_alpha_values(self) -> Dict[int, float]:
        """Return raw alpha values per layer."""
        with torch.no_grad():
            return {
                layer_idx: self.alphas[i].item()
                for i, layer_idx in enumerate(self.stem_layer_indices)
            }


# =============================================================================
# Utility functions
# =============================================================================

def compute_ffn_hidden_dim(
    dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]
) -> int:
    """Compute the FFN hidden dim (same formula used in FeedForward.__init__)."""
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def sync_params_across_dp(params, group_name: str = "adapter"):
    """Broadcast parameter tensors from dp_rank 0 to all STEM DP ranks."""
    if get_stem_data_parallel_world_size() <= 1:
        return
    dp_group = get_stem_data_parallel_group()
    src_rank = torch.distributed.get_global_rank(dp_group, 0)
    for param in params:
        torch.distributed.broadcast(param.data, src=src_rank, group=dp_group)
    logger.info(f"Synchronized {group_name} params across STEM data-parallel ranks")


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_adapter_checkpoint(
    adapter: StemAdapterManager,
    stem_optimizer,
    alpha_optimizer: AdamW,
    backbone_optimizer: Optional[AdamW],
    schedulers: Dict[str, lr_scheduler.LambdaLR],
    train_state: AdapterTrainState,
    args: StemAdapterFinetuneArgs,
    ckpt_dir: Path,
):
    """Save full adapter checkpoint: STEM embeddings, alphas, optimizers, train state."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save STEM embeddings via existing infrastructure
    class _StemHolder(nn.Module):
        def __init__(self, stem_embs):
            super().__init__()
            self.stem_embeddings = stem_embs

    holder = _StemHolder(adapter.stem_embeddings)
    save_stem_shards(
        extract_stem_state_dict(holder),
        ckpt_dir,
        holder,
    )

    # 2. Save alpha parameters + optimizer + schedulers (from DP rank 0 only)
    if get_stem_data_parallel_rank() == 0:
        mp_rank = get_stem_model_parallel_rank()

        # Alpha values
        alpha_dict = {
            f"alpha_{i}": adapter.alphas[i].data.cpu()
            for i in range(len(adapter.alphas))
        }
        torch.save(alpha_dict, ckpt_dir / f"alphas_mp{mp_rank}.pt")

        # Alpha optimizer state
        torch.save(
            alpha_optimizer.state_dict(),
            ckpt_dir / f"alpha_optim_mp{mp_rank}.pt",
        )

        # STEM optimizer state (CPU or GPU depending on offload mode)
        if isinstance(stem_optimizer, CPUOffloadedAdamW):
            torch.save(
                stem_optimizer.state_dict(),
                ckpt_dir / f"stem_optim_cpu_mp{mp_rank}.pt",
            )
        else:
            torch.save(
                stem_optimizer.state_dict(),
                ckpt_dir / f"stem_optim_mp{mp_rank}.pt",
            )

        # Backbone optimizer state (if exists)
        if backbone_optimizer is not None:
            # Note: backbone optimizer may contain DTensors from FSDP
            # Save only on rank 0 as a simple state dict
            try:
                torch.save(
                    backbone_optimizer.state_dict(),
                    ckpt_dir / f"backbone_optim_mp{mp_rank}.pt",
                )
            except Exception as e:
                logger.warning(f"Could not save backbone optimizer: {e}")

        # Scheduler states
        scheduler_states = {
            name: sched.state_dict() for name, sched in schedulers.items()
        }

        # Train state
        ts_dict = train_state.state_dict()
        ts_dict["scheduler_states"] = scheduler_states
        with open(ckpt_dir / f"train_state_mp{mp_rank}.json", "w") as f:
            # Convert tensors to serializable format
            def _serialize(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                return obj
            json.dump(ts_dict, f, default=_serialize)

        logger.info(f"Saved adapter state to {ckpt_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_adapter_checkpoint(
    adapter: StemAdapterManager,
    stem_optimizer,
    alpha_optimizer: AdamW,
    schedulers: Dict[str, lr_scheduler.LambdaLR],
    train_state: AdapterTrainState,
    ckpt_dir: Path,
):
    """Load adapter checkpoint."""
    if not ckpt_dir.exists():
        logger.info(f"No checkpoint at {ckpt_dir}, starting fresh")
        return

    # 1. Load STEM embeddings
    class _StemHolder(nn.Module):
        def __init__(self, stem_embs):
            super().__init__()
            self.stem_embeddings = stem_embs

    holder = _StemHolder(adapter.stem_embeddings)
    try:
        load_stem_shards(holder, ckpt_dir, stem_optimizer=None)
    except Exception as e:
        logger.warning(f"Could not load STEM shards: {e}")

    # 2. Load alphas
    mp_rank = get_stem_model_parallel_rank()
    alpha_path = ckpt_dir / f"alphas_mp{mp_rank}.pt"
    if alpha_path.exists():
        alpha_dict = torch.load(alpha_path, map_location="cpu", weights_only=True)
        for i in range(len(adapter.alphas)):
            key = f"alpha_{i}"
            if key in alpha_dict:
                adapter.alphas[i].data.copy_(alpha_dict[key].to(adapter.alphas[i].device))
        logger.info(f"Loaded alpha values from {alpha_path}")

    # 3. Load alpha optimizer
    alpha_optim_path = ckpt_dir / f"alpha_optim_mp{mp_rank}.pt"
    if alpha_optim_path.exists():
        alpha_optimizer.load_state_dict(
            torch.load(alpha_optim_path, map_location="cpu", weights_only=True)
        )

    # 4. Load STEM optimizer
    if isinstance(stem_optimizer, CPUOffloadedAdamW):
        stem_optim_path = ckpt_dir / f"stem_optim_cpu_mp{mp_rank}.pt"
    else:
        stem_optim_path = ckpt_dir / f"stem_optim_mp{mp_rank}.pt"
    if stem_optim_path.exists():
        stem_optimizer.load_state_dict(
            torch.load(stem_optim_path, map_location="cpu", weights_only=True)
        )

    # 5. Load train state
    ts_path = ckpt_dir / f"train_state_mp{mp_rank}.json"
    if ts_path.exists():
        with open(ts_path, "r") as f:
            ts_dict = json.load(f)
        train_state.load_state_dict(ts_dict)

        # Restore scheduler states
        sched_states = ts_dict.get("scheduler_states", {})
        for name, sched in schedulers.items():
            if name in sched_states:
                sched.load_state_dict(sched_states[name])

        logger.info(f"Loaded train state from {ts_path}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f"Checkpoint loaded from {ckpt_dir}")


# =============================================================================
# Main training function
# =============================================================================

preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning(f"Signal handler called with signal {signum}")
    logger.warning("Preemption! Checkpointing ASAP and exiting.")
    preemption_flag["flag"] = True


def train(args_dict):
    """Main training entry point."""
    args = args_dict if isinstance(args_dict, StemAdapterFinetuneArgs) else args_dict

    with ExitStack() as context_stack:
        # ================================================================
        # 1. Setup: tokenizer, distributed, logging
        # ================================================================
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        if args.model.vocab_size < 0:
            args.model.vocab_size = tokenizer.n_words
        assert args.model.vocab_size == tokenizer.n_words

        # Auto-fix dp_replicate if needed
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
            logger.info(
                f"Auto-set dp_replicate={args.distributed.dp_replicate} "
                f"(dp_shard={args.distributed.dp_shard}, "
                f"tp={args.distributed.tp_size}, ws={get_world_size()})"
            )

        assert args.dump_dir, "dump_dir must be set"
        if args.checkpoint.path is None:
            args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")
        args.model.max_seqlen = args.data.seq_len

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")

        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # DP info
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = (
                dp_rank * world_mesh["dp_shard"].size()
                + world_mesh["dp_shard"].get_local_rank()
            )
            dp_degree *= world_mesh["dp_shard"].size()
        logger.info(f"DP rank: {dp_rank}, DP degree: {dp_degree}")

        # Initialize STEM process groups
        initialize_stem_process_group(args.distributed.stem_parallel_size)
        logger.info(
            f"Initialized STEM process groups "
            f"(parallel_size={args.distributed.stem_parallel_size})"
        )

        # ================================================================
        # 2. Build & load pretrained model (standard LMTransformer with w3)
        # ================================================================
        torch.manual_seed(args.seed)
        logger.info("Building pretrained model (standard LMTransformer)")

        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model built on meta device")

        model_param_count = get_num_params(model)

        # Apply FSDP — DISABLE compilation for hook compatibility
        saved_compile = args.distributed.compile
        args.distributed.compile = False
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=None,
            no_recompute_ops=get_no_recompute_ops(),
        )
        args.distributed.compile = saved_compile

        model = model.to_empty(device="cuda")

        # Load pretrained checkpoint
        assert args.checkpoint.init_ckpt_path, (
            "init_ckpt_path must point to the pretrained checkpoint"
        )
        logger.info(f"Loading pretrained model from {args.checkpoint.init_ckpt_path}")
        load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
        model.rope_embeddings.reset_parameters()

        logger.info(f"Pretrained model loaded. Total params: {model_param_count:,}")

        # Validate stem layer indices
        stem_layers = args.stem_layers
        for idx in stem_layers:
            assert 0 <= idx < len(model.layers), (
                f"stem_layer index {idx} out of range [0, {len(model.layers)})"
            )
            assert hasattr(model.layers[idx].feed_forward, "w3"), (
                f"Layer {idx} has no w3 — is this a standard LMTransformer?"
            )

        # ================================================================
        # 3. Create STEM adapter manager
        # ================================================================
        device = torch.device("cuda", torch.cuda.current_device())
        stem_embedding_dim = compute_ffn_hidden_dim(
            args.model.dim, args.model.multiple_of, args.model.ffn_dim_multiplier
        )
        logger.info(f"STEM embedding dim (FFN hidden dim): {stem_embedding_dim}")

        adapter = StemAdapterManager(
            model=model,
            stem_layer_indices=stem_layers,
            vocab_size=args.model.vocab_size,
            stem_embedding_dim=stem_embedding_dim,
            init_type=args.stem_init_type,
            alpha_init=args.alpha_init,
            device=device,
        )

        # Ensure all adapter parameters require grad
        for param in adapter.parameters():
            param.requires_grad = True

        # Sync adapter params across DP ranks
        sync_params_across_dp(
            list(adapter.stem_embeddings.parameters()),
            group_name="stem_embeddings",
        )
        sync_params_across_dp(
            list(adapter.alphas),
            group_name="alphas",
        )

        # Register hooks on the pretrained model
        adapter.register_hooks(model)

        stem_param_count = sum(p.numel() for p in adapter.stem_embeddings.parameters())
        alpha_param_count = sum(p.numel() for p in adapter.alphas)
        logger.info(
            f"Adapter created: {len(stem_layers)} layers, "
            f"{stem_param_count:,} STEM params (per MP shard), "
            f"{alpha_param_count} alpha params"
        )

        # Log initial gate values
        gate_vals = adapter.get_gate_values()
        logger.info(
            f"Initial gate values (sigmoid(alpha)): "
            + ", ".join(f"L{k}:{v:.4f}" for k, v in gate_vals.items())
        )

        # ================================================================
        # 4. Create optimizers
        # ================================================================

        # Determine total steps for scheduler (excluding warmup phase)
        total_optim_steps = args.steps
        post_warmup_steps = args.steps - args.warmup_steps if args.use_warmup else args.steps

        # --- STEM embedding optimizer ---
        if args.cpu_offload_stem_optim:
            logger.info("Using CPU-offloaded AdamW for STEM embeddings")
            stem_optimizer = CPUOffloadedAdamW(
                gpu_params=list(adapter.stem_embeddings.parameters()),
                lr=args.stem_lr,
                betas=(args.stem_beta1, args.stem_beta2),
                weight_decay=args.stem_weight_decay,
                eps=args.stem_epsilon,
            )
        else:
            stem_optimizer = AdamW(
                adapter.stem_embeddings.parameters(),
                lr=args.stem_lr,
                betas=(args.stem_beta1, args.stem_beta2),
                weight_decay=args.stem_weight_decay,
                eps=args.stem_epsilon,
                fused=False,
            )

        # --- Alpha optimizer ---
        alpha_optimizer = AdamW(
            list(adapter.alphas),
            lr=args.alpha_lr,
            weight_decay=args.alpha_weight_decay,
            fused=False,
        )

        # --- Backbone optimizer (created with initial lr=0 if using warmup) ---
        backbone_params = list(model.parameters())
        backbone_initial_lr = 0.0 if args.use_warmup else args.backbone_lr
        backbone_optimizer = AdamW(
            backbone_params,
            lr=backbone_initial_lr,
            weight_decay=args.backbone_weight_decay,
            fused=True,
        )
        # During warmup, we disable backbone grads entirely (more efficient than lr=0)
        # The optimizer is created now so we can restore from checkpoint uniformly

        # --- LR schedulers ---
        stem_optim_args = OptimArgs(
            lr=args.stem_lr,
            scheduler=args.scheduler,
            warmup=args.warmup_lr_steps,
            lr_min_ratio=args.lr_min_ratio,
        )
        stem_lr_fn = build_lr_fn(stem_optim_args, total_optim_steps)
        stem_scheduler = lr_scheduler.LambdaLR(
            stem_optimizer if not isinstance(stem_optimizer, CPUOffloadedAdamW)
            else stem_optimizer.cpu_optimizer,
            stem_lr_fn,
        )

        alpha_optim_args = OptimArgs(
            lr=args.alpha_lr,
            scheduler=args.scheduler,
            warmup=args.warmup_lr_steps,
            lr_min_ratio=args.lr_min_ratio,
        )
        alpha_lr_fn = build_lr_fn(alpha_optim_args, total_optim_steps)
        alpha_scheduler = lr_scheduler.LambdaLR(alpha_optimizer, alpha_lr_fn)

        backbone_optim_args = OptimArgs(
            lr=args.backbone_lr,
            scheduler=args.scheduler,
            warmup=args.warmup_lr_steps,
            lr_min_ratio=args.lr_min_ratio,
        )
        backbone_lr_fn = build_lr_fn(backbone_optim_args, total_optim_steps)
        backbone_scheduler = lr_scheduler.LambdaLR(backbone_optimizer, backbone_lr_fn)

        schedulers = {
            "stem": stem_scheduler,
            "alpha": alpha_scheduler,
            "backbone": backbone_scheduler,
        }

        # ================================================================
        # 5. Data loader & training state
        # ================================================================
        data_loader_state = init_dataloader_state_from_args(args.data, dp_rank, dp_degree)

        train_state = AdapterTrainState(
            step=0,
            acc_step=0,
            scheduler_states={},
            data_loader_state=data_loader_state,
            is_warmup=args.use_warmup,
        )

        # ================================================================
        # 6. Load existing checkpoint (if any)
        # ================================================================
        ckpt_base = Path(args.checkpoint.path)
        if ckpt_base.exists():
            existing = sorted(
                [
                    d for d in ckpt_base.iterdir()
                    if d.is_dir() and re.match(r"\d{10}", d.name)
                ],
                key=lambda p: int(p.name),
            )
            if existing:
                logger.info(f"Found checkpoints: {[p.name for p in existing]}")
                load_adapter_checkpoint(
                    adapter, stem_optimizer, alpha_optimizer,
                    schedulers, train_state, existing[-1],
                )

        # ================================================================
        # 7. Set up warmup / post-warmup freezing
        # ================================================================
        def freeze_backbone():
            """Freeze all backbone model parameters."""
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            logger.info("Backbone FROZEN (warmup mode)")

        def unfreeze_backbone():
            """Unfreeze backbone model parameters for full finetuning."""
            for param in model.parameters():
                param.requires_grad = True
            model.train()
            # Set backbone optimizer LR to the scheduled value
            for pg in backbone_optimizer.param_groups:
                pg["lr"] = args.backbone_lr
            logger.info("Backbone UNFROZEN (full finetuning mode)")

        # Apply initial freeze state
        if train_state.is_warmup:
            freeze_backbone()
        else:
            unfreeze_backbone()

        gc.disable()

        # ================================================================
        # 8. Training loop
        # ================================================================
        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU: {gpu_memory_monitor.device_name} "
            f"({gpu_memory_monitor.device_capacity_gib:.2f} GiB)"
        )

        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(args.data, state=train_state.data_loader_state)
        )
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()

        logger.info(
            f"Starting training from step {train_state.step} "
            f"(warmup={'ON' if train_state.is_warmup else 'OFF'}, "
            f"warmup_steps={args.warmup_steps if args.use_warmup else 'N/A'}, "
            f"total_steps={args.steps})"
        )

        saved = False
        while train_state.step < args.steps:
            # ---- Check warmup → full finetuning transition ----
            if (
                args.use_warmup
                and train_state.is_warmup
                and train_state.step >= args.warmup_steps
            ):
                logger.info(
                    f"=== WARMUP COMPLETE at step {train_state.step} ==="
                )
                train_state.is_warmup = False
                unfreeze_backbone()

            # ---- Gradient accumulation bookkeeping ----
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # ---- Load batch ----
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            target = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            # ---- Forward pass ----
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # Set input_ids for adapter hooks
            adapter.set_input_ids(input_ids)

            # Forward through model (hooks will blend w3 with STEM embeddings)
            loss = model(input_ids, target=target)

            # Scale for gradient accumulation
            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps

            # ---- Backward pass ----
            loss.backward()

            # Undo scaling for logging
            loss_for_log = loss.detach() * args.grad_acc_steps

            # ---- Optimizer step ----
            grad_norm_backbone = -1.0
            grad_norm_stem = -1.0
            grad_norm_alpha = -1.0

            if train_state.acc_step == 0:
                # All-reduce STEM gradients across STEM DP ranks
                if get_stem_data_parallel_world_size() > 1:
                    dp_group = get_stem_data_parallel_group()
                    for param in adapter.stem_embeddings.parameters():
                        if param.grad is not None:
                            torch.distributed.all_reduce(
                                param.grad,
                                op=torch.distributed.ReduceOp.AVG,
                                group=dp_group,
                            )
                    for param in adapter.alphas:
                        if param.grad is not None:
                            torch.distributed.all_reduce(
                                param.grad,
                                op=torch.distributed.ReduceOp.AVG,
                                group=dp_group,
                            )

                # Clip STEM gradients
                stem_params = [
                    p for p in adapter.stem_embeddings.parameters()
                    if p.grad is not None
                ]
                if stem_params:
                    grad_norm_stem = torch.nn.utils.clip_grad_norm_(
                        stem_params, max_norm=args.stem_clip, foreach=False,
                    ).item()

                # Clip alpha gradients (small params, generous clip)
                alpha_params = [p for p in adapter.alphas if p.grad is not None]
                if alpha_params:
                    grad_norm_alpha = torch.nn.utils.clip_grad_norm_(
                        alpha_params, max_norm=10.0, foreach=False,
                    ).item()

                # Clip backbone gradients (if unfrozen)
                if not train_state.is_warmup:
                    bb_params = [p for p in model.parameters() if p.grad is not None]
                    if bb_params:
                        grad_norm_backbone = torch.nn.utils.clip_grad_norm_(
                            bb_params, max_norm=args.backbone_clip, foreach=True,
                        )
                        grad_norm_backbone = (
                            grad_norm_backbone.full_tensor()
                            if isinstance(grad_norm_backbone, DTensor)
                            else grad_norm_backbone
                        ).item()

                # Step all active optimizers
                stem_optimizer.step()
                alpha_optimizer.step()
                stem_scheduler.step()
                alpha_scheduler.step()

                if not train_state.is_warmup:
                    backbone_optimizer.step()
                    backbone_scheduler.step()

                # Zero gradients
                stem_optimizer.zero_grad()
                alpha_optimizer.zero_grad()
                if not train_state.is_warmup:
                    backbone_optimizer.zero_grad()

                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            if torch_profiler:
                import xformers.profiler
                xformers.profiler.step()

            # ---- Logging ----
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

                # Current LR values
                curr_stem_lr = float(
                    (stem_optimizer.cpu_optimizer if isinstance(stem_optimizer, CPUOffloadedAdamW)
                     else stem_optimizer).param_groups[0]["lr"]
                )
                curr_alpha_lr = float(alpha_optimizer.param_groups[0]["lr"])
                curr_bb_lr = float(backbone_optimizer.param_groups[0]["lr"])

                # Gate values
                gate_vals = adapter.get_gate_values()

                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "is_warmup": int(train_state.is_warmup),
                        "speed": {
                            "wps": wps,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "stem_grad_norm": grad_norm_stem,
                            "alpha_grad_norm": grad_norm_alpha,
                            "backbone_grad_norm": grad_norm_backbone,
                            "stem_lr": curr_stem_lr,
                            "alpha_lr": curr_alpha_lr,
                            "backbone_lr": curr_bb_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {"loss/nll": loss_for_log.item()}
                for layer_idx, gv in gate_vals.items():
                    to_sync[f"gate/layer_{layer_idx}"] = gv
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()

                gate_str = " ".join(
                    f"L{k}:{v:.3f}" for k, v in gate_vals.items()
                )
                alpha_str = " ".join(
                    f"L{k}:{v:.2f}" for k, v in adapter.get_alpha_values().items()
                )
                logger.info(
                    f"step: {train_state.step}"
                    f"  {'WARM' if train_state.is_warmup else 'FULL'}"
                    f"  loss: {loss_for_log.item():.4f}"
                    f"  g_stem: {grad_norm_stem:.2e}"
                    f"  g_alpha: {grad_norm_alpha:.2e}"
                    f"  g_bb: {grad_norm_backbone:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  lr_s: {curr_stem_lr:.2e}"
                    f"  lr_a: {curr_alpha_lr:.2e}"
                    f"  lr_bb: {curr_bb_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"\n    gates: [{gate_str}]"
                    f"\n    alphas: [{alpha_str}]"
                )

            # ---- Checkpointing ----
            saved = False
            if every_n_steps(train_state, args.checkpoint.dump.every, acc_step=0):
                ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                save_adapter_checkpoint(
                    adapter, stem_optimizer, alpha_optimizer,
                    backbone_optimizer, schedulers, train_state,
                    args, ckpt_dir,
                )
                saved = True

            if preemption_flag["flag"]:
                if not saved:
                    ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                    save_adapter_checkpoint(
                        adapter, stem_optimizer, alpha_optimizer,
                        backbone_optimizer, schedulers, train_state,
                        args, ckpt_dir,
                    )
                requeue_slurm_job()
                sys.exit(0)

        # ---- Final save ----
        if not saved:
            ckpt_dir = ckpt_base / f"{train_state.step:010d}"
            save_adapter_checkpoint(
                adapter, stem_optimizer, alpha_optimizer,
                backbone_optimizer, schedulers, train_state,
                args, ckpt_dir,
            )

        # ---- Cleanup ----
        adapter.remove_hooks()

    gc.collect()
    logger.info("Adapter finetuning complete!")

    # Log final gate values
    final_gates = adapter.get_gate_values()
    logger.info(
        "Final gate values: "
        + ", ".join(f"L{k}:{v:.4f}" for k, v in final_gates.items())
    )


# =============================================================================
# Entry point
# =============================================================================

def main():
    """CLI entry point with OmegaConf config loading."""
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(StemAdapterFinetuneArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()