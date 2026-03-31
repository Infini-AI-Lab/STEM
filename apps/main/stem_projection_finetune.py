# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Projection-based learning of STEM embeddings.

Instead of directly finetuning per-token STEM embedding tables (vocab_size x
hidden_dim per layer), this script learns a compact linear projection
(dim -> hidden_dim) that maps token embeddings to the STEM up-projection
space.  The same standard LMTransformer architecture from
apps/main/transformer.py is used, but the up-projection (w3) at stem layers
takes token embeddings as input instead of the layer's hidden state.

Three loss modes are supported (set via ``loss_type``):

  nll  -- End-to-end NLL: hooks replace w3(hidden_state) with proj(tok_emb)
          at stem layers.  The full model forward runs with gradients flowing
          through the frozen model, and cross-entropy loss updates the
          projection weights.  More expensive (full backward pass through
          the frozen layers) but trains with the actual language modelling
          objective.

  mse  -- Layerwise down-proj distillation: MSE between original and modified FFN
          down-projection outputs:
          MSE(w2(SiLU(w1(x)) * proj(tok_emb)), w2(SiLU(w1(x)) * w3(x)))
          Cheaper (only backprops through the projection), but the objective
          is a proxy for the actual language modelling loss.

  mse_up  -- Layerwise up-proj distillation: MSE between original FFN up-proj
             output and projection output:
             MSE(w3(x), proj(tok_emb))
             This mode only needs to hook/store w3 outputs (plus token
             embeddings), so it avoids collecting w1/w2 intermediates.

After training, STEM embeddings can be derived by pre-computing:
  stem_emb[token_id] = projection(tok_embeddings.weight[token_id])
for all tokens in the vocabulary.  The script saves derived STEM embedding
shards at each checkpoint for direct use by the STEM training pipeline.

The approach:
1. Load the pretrained LLaMA model (standard LMTransformer with w1, w2, w3)
2. Freeze all model parameters
3. Create trainable projection modules nn.Linear(dim, hidden_dim) per stem
   layer (optionally initialized from pretrained w3 weights)
4. For each training batch:
   a. Capture token embeddings from the frozen model
   b. Compute loss (NLL or MSE) using projections in place of w3
   c. Backprop to update only the projection weights
5. At checkpoint time, derive and save STEM embedding tables
"""

import gc
import logging
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
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
    initialize_stem_process_group,
    get_stem_data_parallel_group,
    get_stem_data_parallel_rank,
    get_stem_data_parallel_world_size,
    get_stem_model_parallel_rank,
    get_stem_model_parallel_world_size,
)
from lingua.stem_checkpoint import (
    STEM_SUBDIR_NAME,
    STEM_MODEL_FILE_TEMPLATE,
)

from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    build_fsdp_grouping_plan as llama_build_fsdp_grouping_plan,
    get_no_recompute_ops as llama_get_no_recompute_ops,
)
from apps.main.qwen3 import (
    Qwen3LMTransformerArgs,
    Qwen3LMTransformer,
    build_fsdp_grouping_plan as qwen3_build_fsdp_grouping_plan,
    get_no_recompute_ops as qwen3_get_no_recompute_ops,
)
from apps.main.olmo3 import (
    OLMo3LMTransformerArgs,
    OLMo3LMTransformer,
    build_fsdp_grouping_plan as olmo3_build_fsdp_grouping_plan,
    get_no_recompute_ops as olmo3_get_no_recompute_ops,
)
from apps.main.train import every_n_steps

# ---------------------------------------------------------------------------
# Model registry: model_type -> (model_cls, args_cls,
#                                 build_fsdp_grouping_plan,
#                                 get_no_recompute_ops)
# ---------------------------------------------------------------------------
PROJ_MODEL_REGISTRY = {
    "llama": (
        LMTransformer, LMTransformerArgs,
        llama_build_fsdp_grouping_plan, llama_get_no_recompute_ops,
    ),
    "qwen3": (
        Qwen3LMTransformer, Qwen3LMTransformerArgs,
        qwen3_build_fsdp_grouping_plan, qwen3_get_no_recompute_ops,
    ),
    "olmo3": (
        OLMo3LMTransformer, OLMo3LMTransformerArgs,
        olmo3_build_fsdp_grouping_plan, olmo3_get_no_recompute_ops,
    ),
}

import wandb

logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class ProjectionFinetuneArgs:
    name: str = "stem_projection_finetune"
    dump_dir: str = ""

    seed: int = 42

    # Model type: "llama", "qwen3", or "olmo3"
    model_type: str = "llama"

    # Number of gradient accumulation steps
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000

    # Total optimizer steps
    steps: int = 10000

    # Which transformer layers get a projection (replaces w3 input)
    stem_layers: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15])

    # Loss type: "nll", "mse", or "mse_up"
    #   nll: End-to-end NLL with projection replacing w3 output at stem layers.
    #        NOTE: this requires a full backward pass through the frozen model,
    #        so it is more memory- and compute-intensive than MSE mode.
    #   mse: Layerwise MSE on FFN down-projection outputs
    #        (w2(SiLU(w1) * proj(tok_emb)) vs w2(SiLU(w1) * w3)).
    #   mse_up: Layerwise MSE on FFN up-projection outputs
    #        (proj(tok_emb) vs w3). Only requires hooking w3.
    loss_type: str = "mse"

    # Whether to initialize projections from pretrained w3 weights.
    # If True, projections start identical to the original w3 weights, so the
    # model begins in a state close to the pretrained model.
    init_from_w3: bool = True
    # Collect token-wise dense up-projection means μ_v^(l) for stage-2 reparam.
    collect_token_means: bool = True
    # Chunk size used while reducing token stats across ranks.
    token_stat_reduce_chunk_size: int = 2048
    # Optional post-pass: further fit projections A^(l) to collected μ_v^(l).
    mean_fit_steps: int = 0
    mean_fit_batch_size: int = 4096
    mean_fit_lr: float = 5e-4

    # Optimizer settings for projections
    proj_lr: float = 1e-3
    proj_weight_decay: float = 0.0
    proj_beta1: float = 0.9
    proj_beta2: float = 0.95
    proj_epsilon: float = 1e-8
    proj_clip: float = 1.0

    # LR schedule
    proj_scheduler: str = "cosine"
    proj_warmup: int = 500
    proj_lr_min_ratio: float = 0.01
    # Periodic projection-replaced eval (0 disables).
    eval_max_steps: int = 0
    # Use a different RNG stream for eval dataloader state.
    eval_seed_offset: int = 1

    data: DataArgs = field(default_factory=DataArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)


# ---------------------------------------------------------------------------
# Train state
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


@dataclass
class TokenMeanStats:
    """
    Running token-wise dense FFN up-projection statistics for stage-1 export.
    """

    token_counts: torch.Tensor
    token_sums: Dict[int, torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def capture_token_embedding_weight(model: torch.nn.Module) -> torch.Tensor:
    """
    Capture full token embedding weight [vocab, dim] in an FSDP-safe way.
    """
    tok_emb_weight_cache: Dict[str, torch.Tensor] = {}

    def _tok_emb_pre_hook(module, args):
        weight = module.weight.detach()
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        tok_emb_weight_cache["weight"] = weight.clone()

    hook = model.tok_embeddings.register_forward_pre_hook(_tok_emb_pre_hook)
    device = torch.device("cuda", torch.cuda.current_device())
    dummy_ids = torch.zeros(1, 32, dtype=torch.long, device=device)
    with torch.no_grad():
        model(dummy_ids)
    hook.remove()
    return tok_emb_weight_cache["weight"]


def initialize_token_mean_stats(
    vocab_size: int,
    hidden_dim: int,
    stem_layers: List[int],
) -> TokenMeanStats:
    token_counts = torch.zeros(vocab_size, dtype=torch.int64, device="cpu")
    token_sums = {
        layer_idx: torch.zeros(vocab_size, hidden_dim, dtype=torch.float32, device="cpu")
        for layer_idx in stem_layers
    }
    return TokenMeanStats(token_counts=token_counts, token_sums=token_sums)


def update_token_mean_stats(
    stats: TokenMeanStats,
    input_ids: torch.Tensor,
    intermediates: Dict[int, Dict[str, torch.Tensor]],
    stem_layers: List[int],
):
    """
    Update running sums/counts from one batch:
      sum[v] += w3(x)_pos for positions with token id v
      count[v] += #positions with token id v
    """
    flat_tokens = input_ids.reshape(-1)
    unique_tokens, inverse, counts = torch.unique(
        flat_tokens, return_inverse=True, return_counts=True
    )
    unique_tokens_cpu = unique_tokens.to(device="cpu")
    counts_cpu = counts.to(device="cpu", dtype=torch.int64)
    stats.token_counts.index_add_(0, unique_tokens_cpu, counts_cpu)

    for layer_idx in stem_layers:
        w3_flat = intermediates[layer_idx]["w3"].reshape(-1, intermediates[layer_idx]["w3"].shape[-1]).float()
        per_token_sum = torch.zeros(
            unique_tokens.shape[0],
            w3_flat.shape[-1],
            dtype=torch.float32,
            device=w3_flat.device,
        )
        per_token_sum.index_add_(0, inverse, w3_flat)
        stats.token_sums[layer_idx].index_add_(
            0,
            unique_tokens_cpu,
            per_token_sum.to(device="cpu"),
        )


def reduce_token_mean_stats_across_ranks(
    stats: TokenMeanStats,
    stem_layers: List[int],
    chunk_size: int,
):
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
        return

    device = torch.device("cuda", torch.cuda.current_device())
    vocab_size = stats.token_counts.shape[0]

    # Counts
    for c_start in range(0, vocab_size, chunk_size):
        c_end = min(c_start + chunk_size, vocab_size)
        chunk = stats.token_counts[c_start:c_end].to(device=device)
        torch.distributed.all_reduce(chunk, op=torch.distributed.ReduceOp.SUM)
        stats.token_counts[c_start:c_end].copy_(chunk.cpu())

    # Layer-wise sums
    for layer_idx in stem_layers:
        layer_sum = stats.token_sums[layer_idx]
        for c_start in range(0, vocab_size, chunk_size):
            c_end = min(c_start + chunk_size, vocab_size)
            chunk = layer_sum[c_start:c_end].to(device=device)
            torch.distributed.all_reduce(chunk, op=torch.distributed.ReduceOp.SUM)
            layer_sum[c_start:c_end].copy_(chunk.cpu())


def compute_token_means_from_stats(
    stats: TokenMeanStats,
    stem_layers: List[int],
) -> Dict[int, torch.Tensor]:
    means: Dict[int, torch.Tensor] = {}
    denom = stats.token_counts.clamp_min(1).to(dtype=torch.float32).unsqueeze(-1)
    for layer_idx in stem_layers:
        means[layer_idx] = stats.token_sums[layer_idx] / denom
    return means


def fit_projections_to_token_means(
    model: torch.nn.Module,
    projections: torch.nn.ModuleList,
    layer_to_proj_idx: Dict[int, int],
    stem_layers: List[int],
    token_means: Dict[int, torch.Tensor],
    token_counts: torch.Tensor,
    args: ProjectionFinetuneArgs,
):
    if args.mean_fit_steps <= 0:
        return

    observed = torch.nonzero(token_counts > 0, as_tuple=False).squeeze(-1)
    if observed.numel() == 0:
        logger.warning("No observed tokens in token mean stats; skipping mean-fit phase")
        return

    logger.info(
        f"Running post-fit of projections to token means for {args.mean_fit_steps} steps "
        f"on {observed.numel()} observed tokens"
    )
    tok_emb_weight = capture_token_embedding_weight(model).cpu()
    fit_optimizer = AdamW(
        projections.parameters(),
        lr=args.mean_fit_lr,
        betas=(args.proj_beta1, args.proj_beta2),
        weight_decay=args.proj_weight_decay,
        eps=args.proj_epsilon,
        fused=False,
    )
    device = torch.device("cuda", torch.cuda.current_device())

    for fit_step in range(args.mean_fit_steps):
        sample_size = min(args.mean_fit_batch_size, observed.numel())
        sampled = observed[torch.randint(0, observed.numel(), (sample_size,))]
        emb = tok_emb_weight[sampled].to(device=device).float()

        loss = torch.tensor(0.0, device=device)
        for layer_idx in stem_layers:
            proj_idx = layer_to_proj_idx[layer_idx]
            target = token_means[layer_idx][sampled].to(device=device).float()
            pred = projections[proj_idx](emb)
            loss = loss + F.mse_loss(pred, target)
        loss = loss / len(stem_layers)

        fit_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in projections.parameters() if p.grad is not None],
            max_norm=args.proj_clip,
            foreach=False,
        )
        fit_optimizer.step()

        if fit_step % 50 == 0 or fit_step + 1 == args.mean_fit_steps:
            logger.info(
                f"[mean-fit] step={fit_step + 1}/{args.mean_fit_steps} "
                f"mse={loss.detach().item():.6f}"
            )


def capture_w3_weights(
    model: torch.nn.Module,
    stem_layer_indices: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Run a dummy forward pass to capture unsharded w3 weights from FSDP.

    Returns:
        {layer_idx: w3_weight_tensor [hidden_dim, dim]}
    """
    w3_weights: Dict[int, torch.Tensor] = {}
    hooks = []

    def _w3_pre_hook(layer_idx: int):
        def hook_fn(module, inp):
            weight = module.weight.detach()
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            w3_weights[layer_idx] = weight.clone()
        return hook_fn

    for layer_idx in stem_layer_indices:
        ffn = model.layers[layer_idx].feed_forward
        hooks.append(ffn.w3.register_forward_pre_hook(_w3_pre_hook(layer_idx)))

    device = torch.device("cuda", torch.cuda.current_device())
    dummy_ids = torch.zeros(1, 32, dtype=torch.long, device=device)
    with torch.no_grad():
        model(dummy_ids)

    for h in hooks:
        h.remove()

    return w3_weights


def collect_intermediates_and_tok_emb(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    stem_layer_indices: List[int],
    target: Optional[torch.Tensor] = None,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], torch.Tensor, Optional[torch.Tensor]]:
    """
    Run the frozen pretrained model forward and capture FFN intermediates
    at each stem layer plus the token embeddings.

    Captures w1 output, w3 output, and w2 weight at each stem layer
    (for the ``down_proj`` MSE computation), as well as the token
    embeddings from ``model.tok_embeddings``.

    Args:
        model: Frozen pretrained LMTransformer (with standard FeedForward).
        input_ids: Input token IDs [batch, seq_len].
        stem_layer_indices: Layer indices at which to capture intermediates.
        target: Optional target token IDs for NLL computation.

    Returns:
        (intermediates, tok_emb, original_nll)
        - intermediates: ``{layer_idx: {"w3": ..., "w1": ..., "w2_weight": ...}}``.
        - tok_emb: Detached token embeddings [batch, seq_len, dim].
        - original_nll: Scalar CE loss (detached) or ``None``.
    """
    intermediates: Dict[int, Dict[str, torch.Tensor]] = {
        idx: {} for idx in stem_layer_indices
    }
    tok_emb_cache: Dict[str, torch.Tensor] = {}
    hooks = []

    # -- Capture token embeddings ------------------------------------------
    def _tok_emb_hook(module, args, output):
        tok_emb_cache["tok_emb"] = output.detach()

    hooks.append(model.tok_embeddings.register_forward_hook(_tok_emb_hook))

    # -- w3 hooks ----------------------------------------------------------
    def _w3_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            intermediates[layer_idx]["w3"] = out.detach()
        return hook_fn

    for layer_idx in stem_layer_indices:
        ffn = model.layers[layer_idx].feed_forward
        hooks.append(ffn.w3.register_forward_hook(_w3_hook(layer_idx)))

    # -- w1 hooks ----------------------------------------------------------
    def _w1_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            intermediates[layer_idx]["w1"] = out.detach()
        return hook_fn

    for layer_idx in stem_layer_indices:
        ffn = model.layers[layer_idx].feed_forward
        hooks.append(ffn.w1.register_forward_hook(_w1_hook(layer_idx)))

    # -- w2 weight capture (pre-forward so FSDP has it unsharded) ----------
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
    tok_emb = tok_emb_cache["tok_emb"]
    return intermediates, tok_emb, original_nll


def collect_w3_and_tok_emb(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    stem_layer_indices: List[int],
    target: Optional[torch.Tensor] = None,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], torch.Tensor, Optional[torch.Tensor]]:
    """
    Run frozen forward and capture only w3 outputs + token embeddings.
    Used by ``loss_type == "mse_up"``.
    """
    intermediates: Dict[int, Dict[str, torch.Tensor]] = {idx: {} for idx in stem_layer_indices}
    tok_emb_cache: Dict[str, torch.Tensor] = {}
    hooks = []

    def _tok_emb_hook(module, args, output):
        tok_emb_cache["tok_emb"] = output.detach()

    hooks.append(model.tok_embeddings.register_forward_hook(_tok_emb_hook))

    def _w3_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            intermediates[layer_idx]["w3"] = out.detach()
        return hook_fn

    for layer_idx in stem_layer_indices:
        ffn = model.layers[layer_idx].feed_forward
        hooks.append(ffn.w3.register_forward_hook(_w3_hook(layer_idx)))

    with torch.no_grad():
        output = model(input_ids, target=target)

    for h in hooks:
        h.remove()

    original_nll = output.detach() if target is not None else None
    tok_emb = tok_emb_cache["tok_emb"]
    return intermediates, tok_emb, original_nll


def compute_projection_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target: torch.Tensor,
    stem_layer_indices: List[int],
    projections: torch.nn.ModuleList,
    layer_to_proj_idx: Dict[int, int],
) -> torch.Tensor:
    """
    Run the frozen model forward with w3 output at each stem layer *replaced*
    by ``proj(tok_emb)``, and compute the NLL loss.  No gradients are computed.

    Args:
        model: Frozen pretrained LMTransformer.
        input_ids: Input token IDs [batch, seq_len].
        target: Target token IDs [batch, seq_len].
        stem_layer_indices: Layer indices where w3 output is replaced.
        projections: ModuleList of nn.Linear projection modules.
        layer_to_proj_idx: Mapping from layer index to projections index.

    Returns:
        Scalar cross-entropy loss (detached).
    """
    tok_emb_cache: Dict[str, torch.Tensor] = {}
    hooks = []

    def _tok_emb_hook(module, args, output):
        tok_emb_cache["tok_emb"] = output.detach()

    hooks.append(model.tok_embeddings.register_forward_hook(_tok_emb_hook))

    def _make_replace_hook(proj_idx: int):
        def hook_fn(module, inp, out):
            tok_emb = tok_emb_cache["tok_emb"].float()
            return projections[proj_idx](tok_emb).to(dtype=out.dtype)
        return hook_fn

    for layer_idx in stem_layer_indices:
        proj_idx = layer_to_proj_idx[layer_idx]
        h = model.layers[layer_idx].feed_forward.w3.register_forward_hook(
            _make_replace_hook(proj_idx)
        )
        hooks.append(h)

    with torch.no_grad():
        proj_nll = model(input_ids, target=target)

    for h in hooks:
        h.remove()

    return proj_nll.detach()


def run_projection_eval(
    model: torch.nn.Module,
    projections: torch.nn.ModuleList,
    layer_to_proj_idx: Dict[int, int],
    stem_layers: List[int],
    eval_loader,
    eval_data_loader_state: PackTokensState,
    eval_max_steps: int,
) -> Tuple[Dict[str, float], PackTokensState]:
    """
    Evaluate projection-replaced validation loss on a small eval stream.
    """
    was_training = model.training
    model.eval()

    total_orig = 0.0
    total_proj = 0.0
    nsteps = 0

    with torch.no_grad():
        for _ in range(eval_max_steps):
            batch, eval_data_loader_state = next(eval_loader)
            batch = torch.tensor(batch, dtype=torch.long)
            input_ids = batch[:, :, 0].cuda()
            target = batch[:, :, 1].cuda()

            original_nll = model(input_ids, target=target).detach()
            proj_nll = compute_projection_nll(
                model=model,
                input_ids=input_ids,
                target=target,
                stem_layer_indices=stem_layers,
                projections=projections,
                layer_to_proj_idx=layer_to_proj_idx,
            )
            total_orig += float(original_nll.item())
            total_proj += float(proj_nll.item())
            nsteps += 1

    if was_training:
        model.train()

    if nsteps == 0:
        return {"eval/nll_original": 0.0, "eval/nll_projection": 0.0, "eval/nll_gap": 0.0}, eval_data_loader_state

    mean_orig = total_orig / nsteps
    mean_proj = total_proj / nsteps
    return {
        "eval/nll_original": mean_orig,
        "eval/nll_projection": mean_proj,
        "eval/nll_gap": mean_proj - mean_orig,
    }, eval_data_loader_state


def sync_projections_across_dp(projections: torch.nn.ModuleList):
    """Broadcast projection weights from rank 0 to all ranks."""
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return
    for param in projections.parameters():
        torch.distributed.broadcast(param.data, src=0)
    logger.info("Synchronized projection weights across all ranks")


# ---------------------------------------------------------------------------
# Derived STEM embeddings
# ---------------------------------------------------------------------------

def derive_and_save_stem_embeddings(
    model: torch.nn.Module,
    projections: torch.nn.ModuleList,
    stem_layers: List[int],
    layer_to_proj_idx: Dict[int, int],
    ckpt_dir: Path,
    hidden_dim: int,
):
    """
    Derive STEM embedding tables from trained projections and save them
    in the STEM shard format for compatibility with the STEM pipeline.

      stem_emb[token_id] = projection(tok_embeddings.weight[token_id])

    The tables are sharded along hidden_dim per STEM MP rank.
    """
    tok_emb_weight = capture_token_embedding_weight(model)  # [vocab_size, dim]
    vocab_size = tok_emb_weight.shape[0]
    device = torch.device("cuda", torch.cuda.current_device())

    mp_rank = get_stem_model_parallel_rank()
    mp_size = get_stem_model_parallel_world_size()
    dp_rank = get_stem_data_parallel_rank()

    partition_size = hidden_dim // mp_size
    start = mp_rank * partition_size
    end = start + partition_size

    # Create stem_shards directory
    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    if get_is_master() and not stem_dir.exists():
        stem_dir.mkdir(parents=True, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Only save one replica per MP shard
    if dp_rank != 0:
        return

    stem_model_sd: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for i, layer_idx in enumerate(stem_layers):
            proj_idx = layer_to_proj_idx[layer_idx]
            proj = projections[proj_idx]

            # Process in chunks to avoid GPU OOM on large vocabularies
            chunk_size = 4096
            shard_chunks = []
            for c_start in range(0, vocab_size, chunk_size):
                c_end = min(c_start + chunk_size, vocab_size)
                chunk_emb = tok_emb_weight[c_start:c_end].to(device).float()
                full_chunk = proj(chunk_emb)  # [chunk, hidden_dim]
                shard_chunks.append(full_chunk[:, start:end].cpu())
            stem_shard = torch.cat(shard_chunks, dim=0)  # [vocab_size, partition_size]
            stem_model_sd[f"stem_embeddings.{i}.weight"] = stem_shard

    model_shard_path = stem_dir / STEM_MODEL_FILE_TEMPLATE.format(mp_rank=mp_rank)
    torch.save(stem_model_sd, model_shard_path)
    logger.info(
        f"Saved derived STEM embedding shard for mp_rank={mp_rank} "
        f"to {model_shard_path}"
    )


def save_stage1_reparam_package(
    projections: torch.nn.ModuleList,
    args: ProjectionFinetuneArgs,
    ckpt_dir: Path,
    stem_layers: List[int],
    token_stats: TokenMeanStats,
):
    token_means = compute_token_means_from_stats(token_stats, stem_layers)
    package = {
        "format_version": "stem_reparam_stage1_v1",
        "model_type": args.model_type,
        "stem_layers": stem_layers,
        "vocab_size": args.model.vocab_size,
        "model_dim": args.model.dim,
        "hidden_dim": compute_ffn_hidden_dim(
            args.model.dim, args.model.multiple_of, args.model.ffn_dim_multiplier
        ),
        "token_counts": token_stats.token_counts.clone().cpu(),
        "token_means": {
            str(layer_idx): token_means[layer_idx].to(dtype=torch.float16).cpu()
            for layer_idx in stem_layers
        },
        "projection_state_dict": {
            name: param.detach().cpu()
            for name, param in projections.named_parameters()
        },
    }
    out_path = ckpt_dir / "stage1_reparam.pt"
    torch.save(package, out_path)
    logger.info(f"Saved stage-1 reparam package to {out_path}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_projection_checkpoint(
    projections: torch.nn.ModuleList,
    optimizer: AdamW,
    train_state: FinetuneTrainState,
    args: ProjectionFinetuneArgs,
    ckpt_dir: Path,
    model: Optional[torch.nn.Module] = None,
    token_stats: Optional[TokenMeanStats] = None,
):
    """Save projection weights, derived STEM embeddings, and training state."""
    import json

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stem_layers = args.stem_layers
    if token_stats is not None:
        reduce_token_mean_stats_across_ranks(
            token_stats, stem_layers, args.token_stat_reduce_chunk_size
        )

    # Projections are replicated -- save from rank 0 only
    if torch.distributed.get_rank() == 0:
        proj_sd = {
            name: param.detach().cpu()
            for name, param in projections.named_parameters()
        }
        torch.save(proj_sd, ckpt_dir / "projections.pt")
        logger.info(f"Saved projection weights to {ckpt_dir / 'projections.pt'}")

        torch.save(optimizer.state_dict(), ckpt_dir / "proj_optimizer.pt")

        ts_path = ckpt_dir / "train_state.json"
        with open(ts_path, "w") as f:
            json.dump(train_state.state_dict(), f)
        logger.info(f"Saved train state to {ts_path}")

        if token_stats is not None:
            stats_path = ckpt_dir / "token_stats.pt"
            torch.save(
                {
                    "token_counts": token_stats.token_counts.clone().cpu(),
                    "token_sums": {
                        str(layer_idx): token_stats.token_sums[layer_idx].clone().cpu()
                        for layer_idx in stem_layers
                    },
                },
                stats_path,
            )
            logger.info(f"Saved token mean stats to {stats_path}")
            save_stage1_reparam_package(
                projections=projections,
                args=args,
                ckpt_dir=ckpt_dir,
                stem_layers=stem_layers,
                token_stats=token_stats,
            )

    # Derive and save STEM embedding tables in the shard format
    if model is not None:
        hidden_dim = compute_ffn_hidden_dim(
            args.model.dim, args.model.multiple_of, args.model.ffn_dim_multiplier
        )
        layer_to_proj_idx = {
            layer_idx: i for i, layer_idx in enumerate(stem_layers)
        }
        derive_and_save_stem_embeddings(
            model, projections, stem_layers, layer_to_proj_idx,
            ckpt_dir, hidden_dim,
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f"Checkpoint saved to {ckpt_dir}")


def load_projection_checkpoint(
    projections: torch.nn.ModuleList,
    optimizer: AdamW,
    train_state: FinetuneTrainState,
    ckpt_dir: Path,
):
    """Load projection weights and training state from ckpt_dir."""
    import json

    if not ckpt_dir.exists():
        logger.info(f"No checkpoint found at {ckpt_dir}, starting fresh")
        return

    device = torch.device("cuda", torch.cuda.current_device())

    # Load projection weights
    proj_path = ckpt_dir / "projections.pt"
    if proj_path.exists():
        proj_sd = torch.load(proj_path, map_location=device)
        with torch.no_grad():
            for name, param in projections.named_parameters():
                if name in proj_sd:
                    param.copy_(proj_sd[name])
        logger.info(f"Loaded projection weights from {proj_path}")

    # Load optimizer state
    opt_path = ckpt_dir / "proj_optimizer.pt"
    if opt_path.exists():
        opt_sd = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(opt_sd)
        logger.info(f"Loaded optimizer state from {opt_path}")

    # Load train state
    ts_path = ckpt_dir / "train_state.json"
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


def train(args: ProjectionFinetuneArgs):
    with ExitStack() as context_stack:
        # ---- Tokenizer & validation ----
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        if args.model.vocab_size < 0:
            args.model.vocab_size = tokenizer.n_words
        assert args.model.vocab_size == tokenizer.n_words

        # Auto-fix dp_replicate if the mesh doesn't match the world size
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

        # ---- Initialize STEM process groups (for saving derived embeddings) ----
        initialize_stem_process_group(args.distributed.stem_parallel_size)
        logger.info(
            f"Initialized STEM process groups with parallel size: "
            f"{args.distributed.stem_parallel_size}"
        )

        # ---- Resolve model class & helpers from the registry ----
        if args.model_type not in PROJ_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{args.model_type}'. "
                f"Available: {list(PROJ_MODEL_REGISTRY.keys())}"
            )
        (
            model_cls, model_args_cls,
            _build_fsdp_grouping_plan, _get_no_recompute_ops,
        ) = PROJ_MODEL_REGISTRY[args.model_type]
        logger.info(f"Using model type: {args.model_type} ({model_cls.__name__})")

        # ---- Build the pretrained model ----
        torch.manual_seed(args.seed)
        logger.info(f"Building pretrained model ({model_cls.__name__})")

        with torch.device("meta"):
            model = model_cls(args.model)
        logger.info("Pretrained model built on meta device")

        model_param_count = get_num_params(model)

        # Apply FSDP (no compilation -- we need forward hooks to work)
        saved_compile = args.distributed.compile
        args.distributed.compile = False
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=_build_fsdp_grouping_plan(args.model),
            tp_parallelize=None,
            no_recompute_ops=_get_no_recompute_ops(),
        )
        args.distributed.compile = saved_compile

        model = model.to_empty(device="cuda")

        # Load pretrained checkpoint
        assert args.checkpoint.init_ckpt_path, (
            "init_ckpt_path must point to the pretrained checkpoint"
        )
        logger.info(f"Loading pretrained model from {args.checkpoint.init_ckpt_path}")
        load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
        model.rope_embeddings.reset_parameters()  # RoPE is a buffer

        # Freeze the entire pretrained model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info(
            f"Pretrained model loaded & frozen. Total params: {model_param_count:,}"
        )

        # Verify stem layers exist and have w3
        stem_layers = args.stem_layers
        for idx in stem_layers:
            assert 0 <= idx < len(model.layers), (
                f"stem_layer index {idx} out of range [0, {len(model.layers)})"
            )
            assert hasattr(model.layers[idx].feed_forward, "w3"), (
                f"Layer {idx} FeedForward has no w3 -- model type '{args.model_type}' may not be compatible"
            )
        assert args.loss_type in ("nll", "mse", "mse_up"), (
            f"Invalid loss_type: {args.loss_type!r}. Must be 'nll', 'mse', or 'mse_up'"
        )
        assert args.distributed.tp_size == 1, (
            "Projection finetuning currently requires tp_size=1"
        )
        logger.info(f"STEM layers: {stem_layers}")
        logger.info(f"Loss type: {args.loss_type}")

        # ---- Compute projection dimensions ----
        hidden_dim = compute_ffn_hidden_dim(
            args.model.dim, args.model.multiple_of, args.model.ffn_dim_multiplier
        )
        logger.info(
            f"Projection: dim={args.model.dim} -> hidden_dim={hidden_dim}"
        )
        token_stats = None
        if args.collect_token_means:
            token_stats = initialize_token_mean_stats(
                vocab_size=args.model.vocab_size,
                hidden_dim=hidden_dim,
                stem_layers=stem_layers,
            )
            logger.info("Enabled token mean collection for stage-1 reparameterization")

        # ---- Create trainable projections ----
        device = torch.device("cuda", torch.cuda.current_device())
        projections = torch.nn.ModuleList([
            torch.nn.Linear(args.model.dim, hidden_dim, bias=False, device=device)
            for _ in range(len(stem_layers))
        ])

        # Map: layer_idx -> index in projections list
        layer_to_proj_idx = {
            layer_idx: i for i, layer_idx in enumerate(stem_layers)
        }

        # ---- Optionally initialize from pretrained w3 weights ----
        if args.init_from_w3:
            logger.info("Capturing w3 weights for projection initialization...")
            w3_weights = capture_w3_weights(model, stem_layers)
            for layer_idx in stem_layers:
                proj_idx = layer_to_proj_idx[layer_idx]
                with torch.no_grad():
                    projections[proj_idx].weight.copy_(
                        w3_weights[layer_idx].to(device=device)
                    )
            del w3_weights
            logger.info("Projections initialized from pretrained w3 weights")
        else:
            for proj in projections:
                torch.nn.init.xavier_normal_(proj.weight)
            logger.info("Projections initialized randomly (xavier_normal)")

        # Ensure all projections require grad
        for param in projections.parameters():
            param.requires_grad = True

        # Sync across ranks so every replica starts identically
        sync_projections_across_dp(projections)

        proj_param_count = sum(p.numel() for p in projections.parameters())
        logger.info(
            f"Projections created: {len(stem_layers)} layers, "
            f"{proj_param_count:,} total params "
            f"(vs ~{args.model.vocab_size * hidden_dim * len(stem_layers):,} "
            f"for full STEM embedding tables)"
        )

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} "
            f"({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # ---- Optimizer & scheduler (projections only) ----
        optimizer = AdamW(
            projections.parameters(),
            lr=args.proj_lr,
            betas=(args.proj_beta1, args.proj_beta2),
            weight_decay=args.proj_weight_decay,
            eps=args.proj_epsilon,
            fused=False,
        )

        optim_args = OptimArgs(
            lr=args.proj_lr,
            scheduler=args.proj_scheduler,
            warmup=args.proj_warmup,
            lr_min_ratio=args.proj_lr_min_ratio,
        )
        lr_fn = build_lr_fn(optim_args, args.steps)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_fn)

        # ---- Data loader ----
        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = FinetuneTrainState(
            step=0,
            acc_step=0,
            scheduler=scheduler,
            data_loader_state=data_loader_state,
        )

        # ---- Load existing finetune checkpoint (if any) ----
        ckpt_base = Path(args.checkpoint.path)
        if ckpt_base.exists():
            import re
            existing = sorted(
                [
                    d for d in ckpt_base.iterdir()
                    if d.is_dir() and re.match(r"\d{10}", d.name)
                ],
                key=lambda p: int(p.name),
            )
            if existing:
                logger.info(
                    f"Found existing checkpoints: {[p.name for p in existing]}"
                )
                load_projection_checkpoint(
                    projections, optimizer, train_state, existing[-1]
                )

        gc.disable()

        # ---- Training loop ----
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(
                args.data, state=train_state.data_loader_state
            )
        )
        eval_data_loader = None
        eval_data_loader_state = None
        if args.eval_max_steps > 0:
            eval_data_args = replace(
                args.data,
                seed=args.data.seed + args.eval_seed_offset,
                load_async=False,
            )
            eval_data_loader_state = init_dataloader_state_from_args(
                eval_data_args, dp_rank, dp_degree
            )
            eval_data_loader = context_stack.enter_context(
                build_dataloader_from_args(
                    eval_data_args, state=eval_data_loader_state
                )
            )
            logger.info(
                f"Enabled periodic projection eval: eval_max_steps={args.eval_max_steps}, "
                f"every={args.checkpoint.eval.every}"
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

            # ---- Forward ----
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            if args.loss_type == "mse":
                # ===========================================================
                # MSE mode: layerwise distillation
                # ===========================================================
                # 1. Collect intermediates from frozen model
                intermediates, tok_emb, original_nll = (
                    collect_intermediates_and_tok_emb(
                        model, input_ids, stem_layers, target=target,
                    )
                )
                if token_stats is not None:
                    update_token_mean_stats(token_stats, input_ids, intermediates, stem_layers)

                # 2. Compute projection NLL (for logging, no grad)
                proj_nll = compute_projection_nll(
                    model, input_ids, target, stem_layers,
                    projections, layer_to_proj_idx,
                )

                # 3. Compute MSE loss across all stem layers
                total_loss = torch.tensor(0.0, device="cuda")
                per_layer_losses: Dict[int, float] = {}

                for layer_idx in stem_layers:
                    proj_idx = layer_to_proj_idx[layer_idx]
                    data = intermediates[layer_idx]

                    # Projection output: proj(tok_emb)
                    proj_out = projections[proj_idx](tok_emb.float())  # [B, S, hidden_dim]

                    # MSE at FFN output: w2(SiLU(w1(x)) * {proj(tok_emb), w3(x)})
                    gate = F.silu(data["w1"].float())  # detached
                    w2_weight = data["w2_weight"].float()  # detached
                    pred_down = F.linear(gate * proj_out.float(), w2_weight)
                    tgt_down = F.linear(
                        gate * data["w3"].float(), w2_weight
                    ).detach()
                    layer_loss = F.mse_loss(pred_down, tgt_down)

                    total_loss = total_loss + layer_loss
                    per_layer_losses[layer_idx] = layer_loss.detach().item()

                # Average across layers
                total_loss = total_loss / len(stem_layers)
                total_loss_for_log = total_loss.detach()

                # Scale for gradient accumulation
                if args.grad_acc_steps > 1:
                    total_loss = total_loss / args.grad_acc_steps

                # Backward (only projections have requires_grad=True)
                total_loss.backward()

            elif args.loss_type == "mse_up":
                # ===========================================================
                # MSE_UP mode: layerwise up-projection distillation
                # ===========================================================
                intermediates, tok_emb, original_nll = collect_w3_and_tok_emb(
                    model, input_ids, stem_layers, target=target
                )
                if token_stats is not None:
                    update_token_mean_stats(token_stats, input_ids, intermediates, stem_layers)

                proj_nll = compute_projection_nll(
                    model, input_ids, target, stem_layers,
                    projections, layer_to_proj_idx,
                )

                total_loss = torch.tensor(0.0, device="cuda")
                per_layer_losses: Dict[int, float] = {}
                for layer_idx in stem_layers:
                    proj_idx = layer_to_proj_idx[layer_idx]
                    proj_out = projections[proj_idx](tok_emb.float())
                    tgt_up = intermediates[layer_idx]["w3"].float().detach()
                    layer_loss = F.mse_loss(proj_out, tgt_up)
                    total_loss = total_loss + layer_loss
                    per_layer_losses[layer_idx] = layer_loss.detach().item()

                total_loss = total_loss / len(stem_layers)
                total_loss_for_log = total_loss.detach()

                if args.grad_acc_steps > 1:
                    total_loss = total_loss / args.grad_acc_steps
                total_loss.backward()

            elif args.loss_type == "nll":
                # ===========================================================
                # NLL mode: end-to-end with projection replacing w3
                # ===========================================================
                # 1. Compute original NLL (for logging, no grad)
                with torch.no_grad():
                    original_nll = model(input_ids, target=target).detach()
                if token_stats is not None:
                    stats_intermediates, _, _ = collect_intermediates_and_tok_emb(
                        model, input_ids, stem_layers, target=None
                    )
                    update_token_mean_stats(token_stats, input_ids, stats_intermediates, stem_layers)

                # 2. Forward with projections replacing w3 (WITH gradients
                #    flowing through the frozen model to the projections)
                tok_emb_cache: Dict[str, torch.Tensor] = {}
                hooks = []

                def _tok_emb_hook_nll(module, args_hook, output):
                    tok_emb_cache["tok_emb"] = output.detach()
                    return output  # pass through unchanged

                hooks.append(
                    model.tok_embeddings.register_forward_hook(_tok_emb_hook_nll)
                )

                for layer_idx in stem_layers:
                    pidx = layer_to_proj_idx[layer_idx]

                    def _make_w3_hook(proj_idx):
                        def hook_fn(module, args_hook, output):
                            tok_emb = tok_emb_cache["tok_emb"].float()
                            return projections[proj_idx](tok_emb).to(
                                dtype=output.dtype
                            )
                        return hook_fn

                    hooks.append(
                        model.layers[layer_idx].feed_forward.w3
                        .register_forward_hook(_make_w3_hook(pidx))
                    )

                # Forward with gradients -- only projections accumulate grads
                nll_loss = model(input_ids, target=target)

                for h in hooks:
                    h.remove()

                proj_nll = nll_loss.detach()
                total_loss_for_log = nll_loss.detach()
                per_layer_losses = {}  # no per-layer MSE in NLL mode

                if args.grad_acc_steps > 1:
                    nll_loss = nll_loss / args.grad_acc_steps

                nll_loss.backward()

            # ---- Optimizer step ----
            grad_norm = -1.0
            if train_state.acc_step == 0:
                # All-reduce projection gradients across all ranks
                # (projections are replicated, not sharded)
                if (
                    torch.distributed.is_initialized()
                    and torch.distributed.get_world_size() > 1
                ):
                    for param in projections.parameters():
                        if param.grad is not None:
                            torch.distributed.all_reduce(
                                param.grad,
                                op=torch.distributed.ReduceOp.AVG,
                            )

                # Clip gradients
                proj_params = [
                    p for p in projections.parameters() if p.grad is not None
                ]
                if proj_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        proj_params, max_norm=args.proj_clip, foreach=False,
                    ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(
                start_timer.elapsed_time(end_timer) * 1e-3, 4
            )

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
                wps = nwords_since_last_log / (
                    time_delta * args.distributed.tp_size
                )
                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step
                    + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
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
                            "proj_grad_norm": grad_norm,
                            "proj_lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync: Dict[str, float] = {
                    f"loss/{args.loss_type}_total": total_loss_for_log.item(),
                    "loss/nll_original": original_nll.item(),
                    "loss/nll_projection": proj_nll.item(),
                }
                for layer_idx, ll in per_layer_losses.items():
                    to_sync[f"loss/mse_layer_{layer_idx}"] = ll
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()

                layer_losses_str = (
                    "  ".join(
                        f"L{idx}:{per_layer_losses.get(idx, 0):.4f}"
                        for idx in stem_layers
                    )
                    if per_layer_losses
                    else ""
                )

                log_msg = (
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  {args.loss_type}: {total_loss_for_log.item():.6f}"
                    f"  nll_orig: {original_nll.item():.4f}"
                    f"  nll_proj: {proj_nll.item():.4f}"
                    f"  grad: {grad_norm:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                )
                if layer_losses_str:
                    log_msg += f"  [{layer_losses_str}]"
                logger.info(log_msg)

            # ---- Checkpointing ----
            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ):
                ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                save_projection_checkpoint(
                    projections, optimizer, train_state, args,
                    ckpt_dir, model=model, token_stats=token_stats,
                )
                saved = True

            if (
                eval_data_loader is not None
                and every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0)
            ):
                eval_metrics, eval_data_loader_state = run_projection_eval(
                    model=model,
                    projections=projections,
                    layer_to_proj_idx=layer_to_proj_idx,
                    stem_layers=stem_layers,
                    eval_loader=eval_data_loader,
                    eval_data_loader_state=eval_data_loader_state,
                    eval_max_steps=args.eval_max_steps,
                )
                eval_metrics = dist_mean_dict(eval_metrics)
                eval_metrics["global_step"] = train_state.step
                if get_is_master():
                    metric_logger.log(eval_metrics)
                logger.info(
                    f"[eval] step={train_state.step} "
                    f"nll_orig={eval_metrics['eval/nll_original']:.4f} "
                    f"nll_proj={eval_metrics['eval/nll_projection']:.4f} "
                    f"gap={eval_metrics['eval/nll_gap']:.4f}"
                )

            if preemption_flag["flag"]:
                if not saved:
                    ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                    save_projection_checkpoint(
                        projections, optimizer, train_state, args,
                        ckpt_dir, model=model, token_stats=token_stats,
                    )
                requeue_slurm_job()
                sys.exit(0)

        if token_stats is not None and args.mean_fit_steps > 0:
            reduce_token_mean_stats_across_ranks(
                token_stats, stem_layers, args.token_stat_reduce_chunk_size
            )
            token_means = compute_token_means_from_stats(token_stats, stem_layers)
            fit_projections_to_token_means(
                model=model,
                projections=projections,
                layer_to_proj_idx=layer_to_proj_idx,
                stem_layers=stem_layers,
                token_means=token_means,
                token_counts=token_stats.token_counts,
                args=args,
            )

        # ---- Final save ----
        if not saved:
            ckpt_dir = ckpt_base / f"{train_state.step:010d}"
            save_projection_checkpoint(
                projections, optimizer, train_state, args,
                ckpt_dir, model=model, token_stats=token_stats,
            )

    gc.collect()
    logger.info("Projection finetuning complete!")


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(ProjectionFinetuneArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()

