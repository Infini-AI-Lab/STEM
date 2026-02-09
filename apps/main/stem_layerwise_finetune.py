# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Layerwise finetuning of STEM embeddings using MSE loss against the pretrained
FFN of the original model.

Three loss types are supported (set via ``loss_type``):

  up_proj   – MSE(stem_emb(ids), w3(x))
              Matches the raw up-projection output.

  gate_up   – MSE(SiLU(w1(x)) * stem_emb(ids), SiLU(w1(x)) * w3(x))
              Matches the gated intermediate (after element-wise gate).

  down_proj – MSE(w2(SiLU(w1(x)) * stem_emb(ids)), w2(SiLU(w1(x)) * w3(x)))
              Matches the full FFN output (after the down projection).

The approach:
1. Load the pretrained LLaMA model (with standard FeedForward that has w1, w2, w3)
2. Freeze all model parameters
3. Create trainable STEM embeddings (ParallelEmbedding) for each stem layer
4. For each training batch:
   a. Run the frozen forward pass, capturing FFN intermediates via hooks
   b. Compute STEM embedding predictions and MSE loss per the chosen loss type
   c. Backprop to update only the STEM embeddings
"""

import gc
import logging
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf
import torch
import torch.distributed
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
    STEM_SUBDIR_NAME,
)

from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
)
from apps.main.train import TrainState, validate_train_args, every_n_steps

import wandb

logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class LayerwiseFinetuneArgs:
    name: str = "stem_layerwise_finetune"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000

    # Total optimizer steps
    steps: int = 10000

    # Which transformer layers get a STEM embedding
    stem_layers: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15])

    # Optimizer settings for stem embeddings
    stem_lr: float = 1e-3
    stem_weight_decay: float = 0.0
    stem_beta1: float = 0.9
    stem_beta2: float = 0.95
    stem_epsilon: float = 1e-8
    stem_clip: float = 1.0

    # Loss type: "up_proj", "gate_up", or "down_proj"
    #   up_proj  : MSE(stem_emb(ids), w3(x))
    #   gate_up  : MSE(SiLU(w1(x)) * stem_emb(ids), SiLU(w1(x)) * w3(x))
    #   down_proj: MSE(w2(SiLU(w1(x)) * stem_emb(ids)), w2(SiLU(w1(x)) * w3(x)))
    loss_type: str = "up_proj"

    # LR schedule
    stem_scheduler: str = "cosine"
    stem_warmup: int = 500
    stem_lr_min_ratio: float = 0.01

    data: DataArgs = field(default_factory=DataArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)


# ---------------------------------------------------------------------------
# Train state (simplified for MSE-only training)
# ---------------------------------------------------------------------------

@dataclass
class FinetuneTrainState(Stateful):
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
        if "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_ffn_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]) -> int:
    """Compute the FFN hidden dim (same formula used in FeedForward.__init__)."""
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def collect_ffn_intermediates(
    model: LMTransformer,
    input_ids: torch.Tensor,
    stem_layer_indices: List[int],
    loss_type: str = "up_proj",
    target: Optional[torch.Tensor] = None,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
    """
    Run the frozen pretrained model forward and capture FFN intermediates at
    each stem layer via forward hooks.  Optionally compute original NLL.

    Captured intermediates depend on ``loss_type``:
      * ``"up_proj"``  : w3 output only.
      * ``"gate_up"``  : w3 and w1 outputs.
      * ``"down_proj"``: w3, w1 outputs and w2 weight (cloned while unsharded).

    Args:
        model: Frozen pretrained LMTransformer (with standard FeedForward).
        input_ids: Input token IDs [batch, seq_len].
        stem_layer_indices: Layer indices at which to capture intermediates.
        loss_type: One of ``"up_proj"``, ``"gate_up"``, ``"down_proj"``.
        target: Optional target token IDs for NLL computation.

    Returns:
        (intermediates, original_nll)
        - intermediates: ``{layer_idx: {"w3": …, ["w1": …], ["w2_weight": …]}}``.
          All tensors are detached.
        - original_nll: Scalar CE loss (detached) or ``None``.
    """
    intermediates: Dict[int, Dict[str, torch.Tensor]] = {
        idx: {} for idx in stem_layer_indices
    }
    hooks = []

    # -- w3 hooks (always) -------------------------------------------------
    def _w3_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            intermediates[layer_idx]["w3"] = out.detach()
        return hook_fn

    for layer_idx in stem_layer_indices:
        ffn = model.layers[layer_idx].feed_forward
        hooks.append(ffn.w3.register_forward_hook(_w3_hook(layer_idx)))

    # -- w1 hooks (gate_up / down_proj) ------------------------------------
    if loss_type in ("gate_up", "down_proj"):
        def _w1_hook(layer_idx: int):
            def hook_fn(module, inp, out):
                intermediates[layer_idx]["w1"] = out.detach()
            return hook_fn

        for layer_idx in stem_layer_indices:
            ffn = model.layers[layer_idx].feed_forward
            hooks.append(ffn.w1.register_forward_hook(_w1_hook(layer_idx)))

    # -- w2 weight capture (down_proj) -------------------------------------
    #    We use a *pre*-forward hook on w2 so we grab its weight while
    #    FSDP still has it unsharded.
    if loss_type == "down_proj":
        def _w2_pre_hook(layer_idx: int):
            def hook_fn(module, inp):
                weight = module.weight.detach()
                if isinstance(weight, DTensor):
                    weight = weight.full_tensor()
                intermediates[layer_idx]["w2_weight"] = weight.clone()
            return hook_fn

        for layer_idx in stem_layer_indices:
            ffn = model.layers[layer_idx].feed_forward
            hooks.append(ffn.w2.register_forward_pre_hook(_w2_pre_hook(layer_idx)))

    # -- Forward pass (no grad) --------------------------------------------
    with torch.no_grad():
        output = model(input_ids, target=target)

    # -- Cleanup -----------------------------------------------------------
    for h in hooks:
        h.remove()

    original_nll = output.detach() if target is not None else None
    return intermediates, original_nll


def compute_stem_nll(
    model: LMTransformer,
    input_ids: torch.Tensor,
    target: torch.Tensor,
    stem_layer_indices: List[int],
    stem_embeddings: torch.nn.ModuleList,
    layer_to_stem_idx: Dict[int, int],
) -> torch.Tensor:
    """
    Run the frozen model forward with each stem layer's w3 output *replaced*
    by the corresponding STEM embedding lookup, and compute the NLL loss.

    This gives the cross-entropy that the model would achieve if it used
    the learned STEM embeddings instead of the original w3 up-projections.

    Everything runs under ``torch.no_grad()`` – no gradients are computed.

    Args:
        model: Frozen pretrained LMTransformer.
        input_ids: Input token IDs [batch, seq_len].
        target: Target token IDs [batch, seq_len].
        stem_layer_indices: Layer indices where w3 is replaced.
        stem_embeddings: ModuleList of ParallelEmbedding modules.
        layer_to_stem_idx: Mapping from layer index to stem_embeddings index.

    Returns:
        Scalar cross-entropy loss (detached).
    """
    hooks = []

    def _make_replace_hook(stem_idx: int):
        def hook_fn(module, inp, out):
            # Replace w3 output with the STEM embedding lookup,
            # casting to match the model's dtype (e.g. bfloat16)
            return stem_embeddings[stem_idx](input_ids).to(dtype=out.dtype)
        return hook_fn

    for layer_idx in stem_layer_indices:
        stem_idx = layer_to_stem_idx[layer_idx]
        h = model.layers[layer_idx].feed_forward.w3.register_forward_hook(
            _make_replace_hook(stem_idx)
        )
        hooks.append(h)

    with torch.no_grad():
        stem_nll = model(input_ids, target=target)

    for h in hooks:
        h.remove()

    return stem_nll.detach()


def sync_stem_embeddings_across_dp(stem_embeddings: torch.nn.ModuleList):
    """Broadcast stem_embeddings weights from dp_rank 0 to all STEM DP ranks."""
    if get_stem_data_parallel_world_size() <= 1:
        return
    dp_group = get_stem_data_parallel_group()
    src_rank = torch.distributed.get_global_rank(dp_group, 0)
    for param in stem_embeddings.parameters():
        torch.distributed.broadcast(param.data, src=src_rank, group=dp_group)
    logger.info("Synchronized stem_embeddings weights across STEM data-parallel ranks")


# ---------------------------------------------------------------------------
# Checkpoint helpers for stem-only finetuning
# ---------------------------------------------------------------------------

def save_finetune_checkpoint(
    stem_embeddings: torch.nn.ModuleList,
    optimizer: AdamW,
    train_state: FinetuneTrainState,
    args: LayerwiseFinetuneArgs,
    ckpt_dir: Path,
):
    """Save stem embeddings and training state to ckpt_dir."""
    import json

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save stem embeddings (one shard per STEM MP rank, only from DP rank 0)
    stem_model_sd = {}
    for name, param in stem_embeddings.named_parameters():
        stem_model_sd[f"stem_embeddings.{name}"] = param

    # Use the existing save_stem_shards infrastructure
    # We need to create a dummy model-like object for the function
    class _StemHolder(torch.nn.Module):
        def __init__(self, stem_embs):
            super().__init__()
            self.stem_embeddings = stem_embs

    holder = _StemHolder(stem_embeddings)
    save_stem_shards(
        extract_stem_state_dict(holder),
        ckpt_dir,
        holder,
    )

    # Save train state (from DP rank 0 only)
    if get_stem_data_parallel_rank() == 0:
        mp_rank = get_stem_model_parallel_rank()
        ts_path = ckpt_dir / f"train_state_mp{mp_rank}.json"
        with open(ts_path, "w") as f:
            json.dump(train_state.state_dict(), f)
        logger.info(f"Saved train state to {ts_path}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f"Checkpoint saved to {ckpt_dir}")


def load_finetune_checkpoint(
    stem_embeddings: torch.nn.ModuleList,
    optimizer: AdamW,
    train_state: FinetuneTrainState,
    ckpt_dir: Path,
):
    """Load stem embeddings and training state from ckpt_dir."""
    import json

    if not ckpt_dir.exists():
        logger.info(f"No checkpoint found at {ckpt_dir}, starting fresh")
        return

    # Load stem embeddings
    class _StemHolder(torch.nn.Module):
        def __init__(self, stem_embs):
            super().__init__()
            self.stem_embeddings = stem_embs

    holder = _StemHolder(stem_embeddings)
    load_stem_shards(holder, ckpt_dir, stem_optimizer=optimizer)

    # Load train state
    mp_rank = get_stem_model_parallel_rank()
    ts_path = ckpt_dir / f"train_state_mp{mp_rank}.json"
    if ts_path.exists():
        with open(ts_path, "r") as f:
            ts_dict = json.load(f)
        train_state.load_state_dict(ts_dict)
        logger.info(f"Loaded train state from {ts_path}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f"Checkpoint loaded from {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption! Checkpointing ASAP and exiting.")
    preemption_flag["flag"] = True


def train(args: LayerwiseFinetuneArgs):
    with ExitStack() as context_stack:
        # ---- Tokenizer & validation ----
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        if args.model.vocab_size < 0:
            args.model.vocab_size = tokenizer.n_words
        assert args.model.vocab_size == tokenizer.n_words

        # Auto-fix dp_replicate if the mesh doesn't match the world size
        # (same logic as validate_train_args in train.py)
        if (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            != get_world_size()
        ):
            assert get_world_size() % args.distributed.dp_shard == 0
            args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard
            assert args.distributed.dp_replicate % args.distributed.tp_size == 0
            args.distributed.dp_replicate = (
                args.distributed.dp_replicate // args.distributed.tp_size
            )
            logger.info(
                f"Auto-set dp_replicate={args.distributed.dp_replicate} "
                f"(dp_shard={args.distributed.dp_shard}, tp_size={args.distributed.tp_size}, "
                f"world_size={get_world_size()})"
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

        # ---- Data-parallel info ----
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank: {dp_rank}, dp size: {dp_degree}")

        # ---- Initialize STEM process groups for ParallelEmbedding ----
        initialize_stem_process_group(args.distributed.stem_parallel_size)
        logger.info(f"Initialized STEM process groups with parallel size: {args.distributed.stem_parallel_size}")

        # ---- Build the *pretrained* model (standard LMTransformer with w3) ----
        torch.manual_seed(args.seed)
        logger.info("Building pretrained model (standard LMTransformer)")

        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Pretrained model built on meta device")

        model_param_count = get_num_params(model)

        # Apply FSDP (no compilation – we need forward hooks to work)
        saved_compile = args.distributed.compile
        args.distributed.compile = False  # Disable compilation for hook compatibility
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=None,
            no_recompute_ops=get_no_recompute_ops(),
        )
        args.distributed.compile = saved_compile  # Restore

        model = model.to_empty(device="cuda")

        # Load pretrained checkpoint
        assert args.checkpoint.init_ckpt_path, "init_ckpt_path must point to the pretrained checkpoint"
        logger.info(f"Loading pretrained model from {args.checkpoint.init_ckpt_path}")
        load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
        model.rope_embeddings.reset_parameters()  # RoPE is a buffer

        # Freeze the entire pretrained model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info(f"Pretrained model loaded & frozen. Total params: {model_param_count:,}")

        # Verify stem layers exist in the model
        stem_layers = args.stem_layers
        for idx in stem_layers:
            assert 0 <= idx < len(model.layers), (
                f"stem_layer index {idx} out of range [0, {len(model.layers)})"
            )
            assert hasattr(model.layers[idx].feed_forward, "w3"), (
                f"Layer {idx} FeedForward has no w3 – is this a standard LMTransformer?"
            )
        assert args.loss_type in ("up_proj", "gate_up", "down_proj"), (
            f"Invalid loss_type: {args.loss_type!r}. "
            f"Must be one of: up_proj, gate_up, down_proj"
        )
        logger.info(f"STEM layers: {stem_layers}")
        logger.info(f"Loss type: {args.loss_type}")

        # ---- Compute stem embedding dim (= FFN hidden dim) ----
        stem_embedding_dim = compute_ffn_hidden_dim(
            args.model.dim, args.model.multiple_of, args.model.ffn_dim_multiplier
        )
        logger.info(f"STEM embedding dim (FFN hidden dim): {stem_embedding_dim}")

        # ---- Create trainable STEM embeddings ----
        device = torch.device("cuda", torch.cuda.current_device())
        stem_embeddings = torch.nn.ModuleList([
            ParallelEmbedding(args.model.vocab_size, stem_embedding_dim, device=device)
            for _ in range(len(stem_layers))
        ])

        # Map: layer_idx -> index in stem_embeddings list
        layer_to_stem_idx = {layer_idx: i for i, layer_idx in enumerate(stem_layers)}

        # Ensure all STEM embeddings require grad
        for param in stem_embeddings.parameters():
            param.requires_grad = True

        # Sync across DP ranks so every replica starts identically
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(args.seed)
            for emb in stem_embeddings:
                emb.reset_parameters()
        sync_stem_embeddings_across_dp(stem_embeddings)

        stem_param_count = sum(p.numel() for p in stem_embeddings.parameters())
        logger.info(f"STEM embeddings created: {len(stem_layers)} layers, "
                     f"{stem_param_count:,} params per STEM MP shard")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(f"GPU capacity: {gpu_memory_monitor.device_name} "
                     f"({gpu_memory_monitor.device_index}) "
                     f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory")
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # ---- Optimizer & scheduler (stem embeddings only) ----
        optimizer = AdamW(
            stem_embeddings.parameters(),
            lr=args.stem_lr,
            betas=(args.stem_beta1, args.stem_beta2),
            weight_decay=args.stem_weight_decay,
            eps=args.stem_epsilon,
            fused=False,
        )

        optim_args = OptimArgs(
            lr=args.stem_lr,
            scheduler=args.stem_scheduler,
            warmup=args.stem_warmup,
            lr_min_ratio=args.stem_lr_min_ratio,
        )
        lr_fn = build_lr_fn(optim_args, args.steps)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_fn)

        # ---- Data loader ----
        data_loader_state = init_dataloader_state_from_args(args.data, dp_rank, dp_degree)

        train_state = FinetuneTrainState(
            step=0,
            acc_step=0,
            scheduler=scheduler,
            data_loader_state=data_loader_state,
        )

        # ---- Load existing finetune checkpoint (if any) ----
        ckpt_base = Path(args.checkpoint.path)
        # Scan for latest checkpoint
        if ckpt_base.exists():
            import re
            existing = sorted(
                [d for d in ckpt_base.iterdir() if d.is_dir() and re.match(r"\d{10}", d.name)],
                key=lambda p: int(p.name),
            )
            if existing:
                logger.info(f"Found existing checkpoints: {[p.name for p in existing]}")
                load_finetune_checkpoint(stem_embeddings, optimizer, train_state, existing[-1])

        gc.disable()

        # ---- Training loop ----
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

        logger.info(f"Starting training from step {train_state.step}")

        saved = False
        while train_state.step < args.steps:
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # ---- Get batch ----
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            target = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen = input_ids.shape

            # ---- Forward: collect FFN intermediates and original NLL ----
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            intermediates, original_nll = collect_ffn_intermediates(
                model, input_ids, stem_layers,
                loss_type=args.loss_type, target=target,
            )

            # ---- Compute NLL with stem embeddings replacing w3 (no grad) ----
            stem_nll = compute_stem_nll(
                model, input_ids, target, stem_layers,
                stem_embeddings, layer_to_stem_idx,
            )

            # ---- Compute MSE loss across all stem layers ----
            total_loss = torch.tensor(0.0, device="cuda")
            per_layer_losses = {}

            for layer_idx in stem_layers:
                stem_idx = layer_to_stem_idx[layer_idx]
                data = intermediates[layer_idx]
                stem_out = stem_embeddings[stem_idx](input_ids)  # [B, S, hidden_dim]

                if args.loss_type == "up_proj":
                    # MSE between stem embedding and w3 output
                    layer_loss = F.mse_loss(
                        stem_out.float(), data["w3"].float()
                    )

                elif args.loss_type == "gate_up":
                    # MSE after applying the gate: SiLU(w1(x)) * {stem_emb, w3}
                    gate = F.silu(data["w1"].float())  # detached
                    pred_gated = gate * stem_out.float()
                    tgt_gated = gate * data["w3"].float()
                    layer_loss = F.mse_loss(pred_gated, tgt_gated)

                elif args.loss_type == "down_proj":
                    # MSE at the FFN output: w2(SiLU(w1(x)) * {stem_emb, w3})
                    gate = F.silu(data["w1"].float())  # detached
                    w2_weight = data["w2_weight"].float()  # detached
                    pred_down = F.linear(gate * stem_out.float(), w2_weight)
                    tgt_down = F.linear(
                        gate * data["w3"].float(), w2_weight
                    ).detach()
                    layer_loss = F.mse_loss(pred_down, tgt_down)

                else:
                    raise ValueError(f"Unknown loss_type: {args.loss_type}")

                total_loss = total_loss + layer_loss
                per_layer_losses[layer_idx] = layer_loss.detach().item()

            # Average across layers
            total_loss = total_loss / len(stem_layers)

            # Scale for gradient accumulation
            if args.grad_acc_steps > 1:
                total_loss = total_loss / args.grad_acc_steps

            # Backward (only stem_embeddings have requires_grad=True)
            total_loss.backward()

            # Undo scaling for logging
            total_loss_for_log = total_loss.detach() * args.grad_acc_steps

            # ---- Optimizer step ----
            grad_norm = -1.0
            if train_state.acc_step == 0:
                # All-reduce stem gradients across STEM data-parallel ranks
                if get_stem_data_parallel_world_size() > 1:
                    dp_group = get_stem_data_parallel_group()
                    for param in stem_embeddings.parameters():
                        if param.grad is not None:
                            torch.distributed.all_reduce(
                                param.grad,
                                op=torch.distributed.ReduceOp.AVG,
                                group=dp_group,
                            )

                # Clip gradients
                stem_params = [p for p in stem_embeddings.parameters() if p.grad is not None]
                if stem_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        stem_params, max_norm=args.stem_clip, foreach=False,
                    ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            if torch_profiler:
                import xformers.profiler
                xformers.profiler.step()

            # ---- Logging ----
            if every_n_steps(
                train_state, args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)
                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = args.grad_acc_steps * train_state.step + train_state.acc_step
                tokens_per_gpu = total_acc_steps * args.data.batch_size * args.data.seq_len
                total_tokens = dp_degree * tokens_per_gpu

                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "stem_grad_norm": grad_norm,
                            "stem_lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {
                    "loss/mse_total": total_loss_for_log.item(),
                    "loss/nll_original": original_nll.item(),
                    "loss/nll_stem": stem_nll.item(),
                }
                for layer_idx, ll in per_layer_losses.items():
                    to_sync[f"loss/mse_layer_{layer_idx}"] = ll
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()

                layer_losses_str = "  ".join(
                    f"L{idx}:{per_layer_losses.get(idx, 0):.4f}" for idx in stem_layers
                )
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  mse: {total_loss_for_log.item():.6f}"
                    f"  nll_orig: {original_nll.item():.4f}"
                    f"  nll_stem: {stem_nll.item():.4f}"
                    f"  grad: {grad_norm:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  [{layer_losses_str}]"
                )

            # ---- Checkpointing ----
            saved = False
            if every_n_steps(train_state, args.checkpoint.dump.every, acc_step=0):
                ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                save_finetune_checkpoint(
                    stem_embeddings, optimizer, train_state, args, ckpt_dir,
                )
                saved = True

            if preemption_flag["flag"]:
                if not saved:
                    ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                    save_finetune_checkpoint(
                        stem_embeddings, optimizer, train_state, args, ckpt_dir,
                    )
                requeue_slurm_job()
                sys.exit(0)

        # ---- Final save ----
        if not saved:
            ckpt_dir = ckpt_base / f"{train_state.step:010d}"
            save_finetune_checkpoint(
                stem_embeddings, optimizer, train_state, args, ckpt_dir,
            )

    gc.collect()
    logger.info("Layerwise finetuning complete!")


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(LayerwiseFinetuneArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()