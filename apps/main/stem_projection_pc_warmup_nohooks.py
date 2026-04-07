import gc
import json
import logging
import os
import re
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn as nn

from lingua.args import dump_config, flatten_dict
from lingua.checkpoint import load_from_checkpoint
from lingua.data import PackTokensState, build_dataloader_from_args, init_dataloader_state_from_args
from lingua.distributed import (
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    parallelize_model,
    requeue_slurm_job,
    setup_env,
    setup_torch_distributed,
    init_signal_handler,
)
from lingua.logger import init_logger
from lingua.metrics import GPUMemoryMonitor, MetricLogger, get_num_params
from lingua.profiling import maybe_run_profiler
from lingua.stem_checkpoint import STEM_MODEL_FILE_TEMPLATE, STEM_SUBDIR_NAME
from lingua.tokenizer import build_tokenizer

from apps.main.olmo3 import (
    OLMo3LMTransformer,
    OLMo3LMTransformerArgs,
    build_fsdp_grouping_plan as olmo3_build_fsdp_grouping_plan,
    get_no_recompute_ops as olmo3_get_no_recompute_ops,
)
from apps.main.qwen3 import (
    Qwen3LMTransformer,
    Qwen3LMTransformerArgs,
    build_fsdp_grouping_plan as qwen3_build_fsdp_grouping_plan,
    get_no_recompute_ops as qwen3_get_no_recompute_ops,
)
from apps.main.train import TrainArgs, every_n_steps, validate_train_args
from apps.main.transformer import (
    LMTransformer,
    LMTransformerArgs,
    build_fsdp_grouping_plan as llama_build_fsdp_grouping_plan,
    get_no_recompute_ops as llama_get_no_recompute_ops,
)

logger = logging.getLogger()

PROJ_MODEL_REGISTRY = {
    "llama": (
        LMTransformer,
        LMTransformerArgs,
        llama_build_fsdp_grouping_plan,
        llama_get_no_recompute_ops,
    ),
    "qwen3": (
        Qwen3LMTransformer,
        Qwen3LMTransformerArgs,
        qwen3_build_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops,
    ),
    "olmo3": (
        OLMo3LMTransformer,
        OLMo3LMTransformerArgs,
        olmo3_build_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops,
    ),
}


def _resolve_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n == "bf16":
        return torch.bfloat16
    if n == "fp16":
        return torch.float16
    if n == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name '{name}', expected one of: bf16, fp16, fp32")


def _compute_vocab_shard(vocab_size: int, shard_size: int, shard_rank: int) -> Tuple[int, int, int]:
    if shard_size <= 0:
        shard_size = vocab_size
    n_shards = (vocab_size + shard_size - 1) // shard_size
    if n_shards <= 0:
        raise ValueError(f"Invalid shard setup: vocab_size={vocab_size}, shard_size={shard_size}")
    if shard_rank < 0:
        shard_rank = 0
    if not (0 <= shard_rank < n_shards):
        raise ValueError(
            f"vocab_shard_rank={shard_rank} out of range for n_shards={n_shards} (shard_size={shard_size})"
        )
    start = shard_rank * shard_size
    end = min(start + shard_size, vocab_size)
    return start, end, n_shards


@dataclass
class ProjectionPCWarmupArgs(TrainArgs):
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    stem_layers: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15])

    collect_dtype: str = "bf16"
    stats_dtype: str = "fp32"

    vocab_shard_size: int = 0
    vocab_shard_rank: int = -1

    token_stat_reduce_chunk_size: int = 4096
    min_count_for_pc_update: int = 32
    max_token_count_collect: int = 50000
    oja_eps: float = 1e-3

    save_mean_stats: bool = True
    save_pc_stats: bool = True
    save_as_stem_shards: bool = True

    eval_max_steps: int = 0


@dataclass
class PCWarmupState:
    step: int
    data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "data_loader_state": self.data_loader_state,
        }

    def load_state_dict(self, state_dict: Dict[str, object]):
        self.step = int(state_dict["step"])
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])


@dataclass
class BatchUpdateContext:
    row_indices: torch.Tensor
    local_token_ids: torch.Tensor


class TokenPCStats(nn.Module):
    """
    Token-wise running stats for a single vocab shard.

    Buffers:
    - token_counts: [V_local] int64
    - token_means_{layer}: [V_local, hidden_dim] stats_dtype
    - token_pc_{layer}: [V_local, hidden_dim] stats_dtype, row-wise unit vectors
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        stem_layers: List[int],
        device: torch.device,
        shard_start: int,
        shard_end: int,
        stats_dtype: torch.dtype,
        min_count_for_pc_update: int,
        max_token_count_collect: int,
        oja_eps: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.stem_layers = list(stem_layers)
        self.shard_start = int(shard_start)
        self.shard_end = int(shard_end)
        self.shard_vocab_size = int(shard_end - shard_start)
        self.stats_dtype = stats_dtype
        self.min_count_for_pc_update = int(min_count_for_pc_update)
        self.max_token_count_collect = int(max_token_count_collect)
        self.oja_eps = float(oja_eps)

        if self.shard_vocab_size <= 0:
            raise ValueError(
                f"Invalid shard size from [{self.shard_start}, {self.shard_end}) for vocab={self.vocab_size}"
            )

        self.register_buffer(
            "token_counts",
            torch.zeros(self.shard_vocab_size, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "token_applied_counts",
            torch.zeros(self.shard_vocab_size, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "token_pending_counts",
            torch.zeros(self.shard_vocab_size, dtype=torch.int64, device=device),
        )
        for layer_idx in self.stem_layers:
            self.register_buffer(
                f"token_means_{layer_idx}",
                torch.zeros(self.shard_vocab_size, hidden_dim, dtype=stats_dtype, device=device),
            )
            init_pc = torch.zeros(self.shard_vocab_size, hidden_dim, dtype=stats_dtype, device=device)
            init_pc[:, 0] = 1.0
            self.register_buffer(f"token_pc_{layer_idx}", init_pc)
            self.register_buffer(
                f"buffer_sum_{layer_idx}",
                torch.zeros(self.shard_vocab_size, hidden_dim, dtype=stats_dtype, device=device),
            )
            self.register_buffer(
                f"buffer_proj_sum_{layer_idx}",
                torch.zeros(self.shard_vocab_size, dtype=stats_dtype, device=device),
            )
            self.register_buffer(
                f"buffer_weighted_sum_{layer_idx}",
                torch.zeros(self.shard_vocab_size, hidden_dim, dtype=stats_dtype, device=device),
            )

    def layer_mean(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"token_means_{layer_idx}")

    def layer_pc(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"token_pc_{layer_idx}")

    def buffer_sum(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"buffer_sum_{layer_idx}")

    def buffer_proj_sum(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"buffer_proj_sum_{layer_idx}")

    def buffer_weighted_sum(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"buffer_weighted_sum_{layer_idx}")

    @torch.no_grad()
    def begin_batch(self, global_token_ids: torch.Tensor) -> Optional[BatchUpdateContext]:
        local_row_mask = (global_token_ids >= self.shard_start) & (global_token_ids < self.shard_end)
        if not local_row_mask.any():
            return None

        row_indices = torch.nonzero(local_row_mask, as_tuple=False).squeeze(1)
        local_token_ids = global_token_ids[row_indices] - self.shard_start

        if self.max_token_count_collect > 0:
            keep = self.token_counts[local_token_ids] < self.max_token_count_collect
            if not keep.any():
                return None
            row_indices = row_indices[keep]
            local_token_ids = local_token_ids[keep]

        if local_token_ids.numel() == 0:
            return None

        self.token_counts.index_add_(
            0,
            local_token_ids,
            torch.ones_like(local_token_ids, dtype=torch.int64),
        )
        self.token_pending_counts.index_add_(
            0,
            local_token_ids,
            torch.ones_like(local_token_ids, dtype=torch.int64),
        )

        return BatchUpdateContext(
            row_indices=row_indices,
            local_token_ids=local_token_ids,
        )

    @torch.no_grad()
    def _flush_selected_tokens_for_layer(
        self,
        layer_idx: int,
        token_ids: torch.Tensor,
        new_applied: torch.Tensor,
    ):
        if token_ids.numel() == 0:
            return
        n = self.token_pending_counts[token_ids]
        n_f = n.to(dtype=self.stats_dtype).unsqueeze(1)

        bsum = self.buffer_sum(layer_idx)[token_ids]
        bproj = self.buffer_proj_sum(layer_idx)[token_ids]
        bweighted = self.buffer_weighted_sum(layer_idx)[token_ids]

        mean_table = self.layer_mean(layer_idx)
        pc_table = self.layer_pc(layer_idx)
        old_means = mean_table[token_ids]
        old_pcs = pc_table[token_ids]
        batch_means = bsum / n_f.clamp_min(1.0)
        alpha = (n.to(dtype=self.stats_dtype) / new_applied.clamp_min(1).to(dtype=self.stats_dtype)).unsqueeze(1)
        new_means = old_means + alpha * (batch_means - old_means)
        mean_table[token_ids] = new_means

        # g = Σ (x-μ)((x-μ)·v), computed from buffered sufficient statistics.
        mu_dot_v = (new_means * old_pcs).sum(dim=1)
        g = (
            bweighted
            - new_means * bproj.unsqueeze(1)
            - bsum * mu_dot_v.unsqueeze(1)
            + n_f * new_means * mu_dot_v.unsqueeze(1)
        )
        candidate = g + (self.oja_eps * old_pcs)
        norms = candidate.norm(dim=1, keepdim=True)
        valid = norms.squeeze(1) > 0
        if valid.any():
            pc_table[token_ids[valid]] = candidate[valid] / norms[valid].clamp_min(1e-12)

    @torch.no_grad()
    def maybe_flush(self, force: bool = False):
        if force:
            flush_mask = self.token_pending_counts > 0
        else:
            flush_mask = self.token_pending_counts >= self.min_count_for_pc_update
        if not flush_mask.any():
            return

        token_ids = torch.nonzero(flush_mask, as_tuple=False).squeeze(1)
        old_applied = self.token_applied_counts[token_ids]
        new_applied = old_applied + self.token_pending_counts[token_ids]
        for layer_idx in self.stem_layers:
            self._flush_selected_tokens_for_layer(
                layer_idx=layer_idx,
                token_ids=token_ids,
                new_applied=new_applied,
            )
            self.buffer_sum(layer_idx)[token_ids].zero_()
            self.buffer_proj_sum(layer_idx)[token_ids].zero_()
            self.buffer_weighted_sum(layer_idx)[token_ids].zero_()
        self.token_applied_counts[token_ids] = new_applied
        self.token_pending_counts[token_ids] = 0

    @torch.no_grad()
    def update_layer_from_batch(
        self,
        layer_idx: int,
        w3_flat: torch.Tensor,
        ctx: BatchUpdateContext,
    ):
        x_local = w3_flat[ctx.row_indices]
        local_token_ids = ctx.local_token_ids
        finite_mask = torch.isfinite(x_local).all(dim=1)
        if not finite_mask.all():
            if not finite_mask.any():
                return
            x_local = x_local[finite_mask]
            local_token_ids = local_token_ids[finite_mask]

        if x_local.numel() == 0:
            return

        x_local = x_local.to(dtype=self.stats_dtype)

        row_pcs = self.layer_pc(layer_idx)[local_token_ids]
        proj = (x_local * row_pcs).sum(dim=1)
        weighted = x_local * proj.unsqueeze(1)

        self.buffer_sum(layer_idx).index_add_(0, local_token_ids, x_local)
        self.buffer_proj_sum(layer_idx).index_add_(0, local_token_ids, proj)
        self.buffer_weighted_sum(layer_idx).index_add_(0, local_token_ids, weighted)


class CaptureFeedForward(nn.Module):
    """
    Hook-free FFN wrapper that captures the w3 activation in a module attribute.
    """

    def __init__(self, base_ffn: nn.Module):
        super().__init__()
        self.w1 = base_ffn.w1
        self.w2 = base_ffn.w2
        self.w3 = base_ffn.w3
        self.dim = getattr(base_ffn, "dim", None)
        self.hidden_dim = getattr(base_ffn, "hidden_dim", None)
        self.last_w3: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        self.last_w3 = x3
        out = self.w2(torch.nn.functional.silu(x1) * x3)
        return out

    def reset_capture(self):
        self.last_w3 = None

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


def install_ffn_captures(model: torch.nn.Module, stem_layer_indices: List[int]) -> Dict[int, CaptureFeedForward]:
    captured: Dict[int, CaptureFeedForward] = {}
    for layer_idx in stem_layer_indices:
        layer = model.layers[layer_idx]
        if isinstance(layer.feed_forward, CaptureFeedForward):
            captured[layer_idx] = layer.feed_forward
            continue
        wrapped = CaptureFeedForward(layer.feed_forward)
        layer.feed_forward = wrapped
        captured[layer_idx] = wrapped
    return captured


@torch.no_grad()
def collect_w3_outputs(
    model: torch.nn.Module,
    capture_ffns: Dict[int, CaptureFeedForward],
    input_ids: torch.Tensor,
    stem_layer_indices: List[int],
    collect_dtype: torch.dtype,
    target: Optional[torch.Tensor] = None,
) -> Tuple[Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    intermediates: Dict[int, torch.Tensor] = {}
    for layer_idx in stem_layer_indices:
        capture_ffns[layer_idx].reset_capture()

    with torch.inference_mode():
        output = model(input_ids, target=target)

    for layer_idx in stem_layer_indices:
        w3 = capture_ffns[layer_idx].last_w3
        if w3 is None:
            raise RuntimeError(f"Missing captured w3 output for layer={layer_idx}")
        intermediates[layer_idx] = w3.detach().to(dtype=collect_dtype)

    detached = output.detach() if target is not None and isinstance(output, torch.Tensor) else None
    return intermediates, detached


def reduce_pc_stats_across_dp(stats: TokenPCStats, chunk_size: int):
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
        return

    vocab_local = stats.token_counts.shape[0]

    for start in range(0, vocab_local, chunk_size):
        end = min(start + chunk_size, vocab_local)

        local_count_chunk = stats.token_counts[start:end].contiguous()
        global_count_chunk = local_count_chunk.clone()
        torch.distributed.all_reduce(global_count_chunk, op=torch.distributed.ReduceOp.SUM)

        local_count_f = local_count_chunk.to(dtype=stats.stats_dtype).unsqueeze(1)
        for layer_idx in stats.stem_layers:
            mean_chunk = stats.layer_mean(layer_idx)[start:end].contiguous()
            pc_chunk = stats.layer_pc(layer_idx)[start:end].contiguous()

            weighted_mean = mean_chunk * local_count_f
            weighted_pc = pc_chunk * local_count_f

            torch.distributed.all_reduce(weighted_mean, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(weighted_pc, op=torch.distributed.ReduceOp.SUM)

            denom = global_count_chunk.to(dtype=stats.stats_dtype).unsqueeze(1).clamp_min(1.0)
            merged_mean = weighted_mean / denom

            merged_pc = weighted_pc
            nonzero = global_count_chunk > 0
            merged_pc[~nonzero] = 0
            if nonzero.any():
                norms = merged_pc[nonzero].norm(dim=1, keepdim=True)
                valid = norms.squeeze(1) > 0
                if valid.any():
                    merged_pc_nonzero = merged_pc[nonzero]
                    merged_pc_nonzero[valid] = merged_pc_nonzero[valid] / norms[valid].clamp_min(1e-12)
                    merged_pc[nonzero] = merged_pc_nonzero

            stats.layer_mean(layer_idx)[start:end].copy_(merged_mean)
            stats.layer_pc(layer_idx)[start:end].copy_(merged_pc)

        stats.token_counts[start:end].copy_(global_count_chunk)


def log_token_count_coverage_summary(stats: TokenPCStats, min_count_for_pc_update: int):
    counts = stats.token_counts.detach().to(dtype=torch.int64).clone()
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)

    local_vocab = int(counts.numel())
    covered = int((counts >= min_count_for_pc_update).sum().item())
    seen_once = int((counts > 0).sum().item())
    max_count = int(counts.max().item()) if local_vocab > 0 else 0
    pct_covered = (100.0 * covered / local_vocab) if local_vocab > 0 else 0.0
    pct_seen_once = (100.0 * seen_once / local_vocab) if local_vocab > 0 else 0.0

    if get_is_master():
        logger.info(
            "Token coverage summary (local shard): "
            f"range=[{stats.shard_start}, {stats.shard_end}), "
            f"count>0 {seen_once}/{local_vocab} ({pct_seen_once:.2f}%), "
            f"count>={min_count_for_pc_update} {covered}/{local_vocab} ({pct_covered:.2f}%), "
            f"max_count={max_count}"
        )


def _save_local_pc_stats(stats: TokenPCStats, train_state: PCWarmupState, ckpt_dir: Path):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    payload = {
        "metadata": {
            "vocab_size": stats.vocab_size,
            "hidden_dim": stats.hidden_dim,
            "stem_layers": stats.stem_layers,
            "shard_start": stats.shard_start,
            "shard_end": stats.shard_end,
            "min_count_for_pc_update": stats.min_count_for_pc_update,
            "max_token_count_collect": stats.max_token_count_collect,
            "oja_eps": stats.oja_eps,
            "step": train_state.step,
            "rank": rank,
        },
        "token_counts": stats.token_counts.detach().cpu(),
        "token_applied_counts": stats.token_applied_counts.detach().cpu(),
        "token_pending_counts": stats.token_pending_counts.detach().cpu(),
        "token_means": {str(li): stats.layer_mean(li).detach().cpu() for li in stats.stem_layers},
        "token_pc": {str(li): stats.layer_pc(li).detach().cpu() for li in stats.stem_layers},
        "buffer_sum": {str(li): stats.buffer_sum(li).detach().cpu() for li in stats.stem_layers},
        "buffer_proj_sum": {str(li): stats.buffer_proj_sum(li).detach().cpu() for li in stats.stem_layers},
        "buffer_weighted_sum": {str(li): stats.buffer_weighted_sum(li).detach().cpu() for li in stats.stem_layers},
    }
    if world_size == 1:
        torch.save(payload, ckpt_dir / "pc_stats.pt")
    else:
        torch.save(payload, ckpt_dir / f"pc_stats_rank{rank:02d}.pt")


def _load_local_pc_stats(stats: TokenPCStats, ckpt_dir: Path):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = ckpt_dir / f"pc_stats_rank{rank:05d}.pt"
    if not path.exists():
        logger.warning(f"Local stats file not found for rank={rank}: {path}")
        return

    payload = torch.load(path, map_location="cpu")
    meta = payload["metadata"]
    if int(meta["shard_start"]) != stats.shard_start or int(meta["shard_end"]) != stats.shard_end:
        raise ValueError(
            f"Checkpoint shard range [{meta['shard_start']}, {meta['shard_end']}) does not match "
            f"current range [{stats.shard_start}, {stats.shard_end})"
        )

    stats.token_counts.copy_(payload["token_counts"].to(device=stats.token_counts.device, dtype=torch.int64))
    if "token_applied_counts" in payload:
        stats.token_applied_counts.copy_(
            payload["token_applied_counts"].to(device=stats.token_counts.device, dtype=torch.int64)
        )
    else:
        stats.token_applied_counts.zero_()
    if "token_pending_counts" in payload:
        stats.token_pending_counts.copy_(
            payload["token_pending_counts"].to(device=stats.token_counts.device, dtype=torch.int64)
        )
    elif "buffer_counts" in payload:
        # Backward compatibility with earlier per-layer buffer counts.
        stats.token_pending_counts.copy_(
            payload["buffer_counts"][str(stats.stem_layers[0])].to(device=stats.token_counts.device, dtype=torch.int64)
        )
    else:
        stats.token_pending_counts.zero_()
    for li in stats.stem_layers:
        stats.layer_mean(li).copy_(payload["token_means"][str(li)].to(device=stats.token_counts.device, dtype=stats.stats_dtype))
        stats.layer_pc(li).copy_(payload["token_pc"][str(li)].to(device=stats.token_counts.device, dtype=stats.stats_dtype))
        if "buffer_sum" in payload:
            stats.buffer_sum(li).copy_(payload["buffer_sum"][str(li)].to(device=stats.token_counts.device, dtype=stats.stats_dtype))
            stats.buffer_proj_sum(li).copy_(payload["buffer_proj_sum"][str(li)].to(device=stats.token_counts.device, dtype=stats.stats_dtype))
            stats.buffer_weighted_sum(li).copy_(
                payload["buffer_weighted_sum"][str(li)].to(device=stats.token_counts.device, dtype=stats.stats_dtype)
            )
        else:
            stats.buffer_sum(li).zero_()
            stats.buffer_proj_sum(li).zero_()
            stats.buffer_weighted_sum(li).zero_()


@torch.no_grad()
def save_pc_as_stem_shards(
    ckpt_dir: Path,
    stats: TokenPCStats,
    stem_layers: List[int],
    stem_parallel_size: int,
    save_mean_stats: bool,
    save_pc_stats: bool,
):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    stem_dir.mkdir(parents=True, exist_ok=True)

    if stats.hidden_dim % stem_parallel_size != 0:
        raise ValueError(
            f"hidden_dim={stats.hidden_dim} must be divisible by stem_parallel_size={stem_parallel_size}"
        )

    shard_dim = stats.hidden_dim // stem_parallel_size
    shard_start = stats.shard_start
    shard_end = stats.shard_end
    vocab_size = stats.vocab_size

    if save_pc_stats:
        for mp_rank in range(stem_parallel_size):
            col_start = mp_rank * shard_dim
            col_end = col_start + shard_dim
            sd: Dict[str, torch.Tensor] = {}
            for i, layer_idx in enumerate(stem_layers):
                local_pc = stats.layer_pc(layer_idx)[:, col_start:col_end].detach().cpu().to(dtype=torch.bfloat16)
                full = torch.zeros(vocab_size, shard_dim, dtype=torch.bfloat16)
                full[shard_start:shard_end] = local_pc
                sd[f"stem_embeddings.{i}.weight"] = full
            shard_path = stem_dir / STEM_MODEL_FILE_TEMPLATE.format(mp_rank=mp_rank)
            torch.save(sd, shard_path)

    if save_mean_stats:
        aux = {
            "metadata": {
                "vocab_size": vocab_size,
                "hidden_dim": stats.hidden_dim,
                "stem_layers": stem_layers,
                "shard_start": shard_start,
                "shard_end": shard_end,
            },
            "token_counts": stats.token_counts.detach().cpu(),
            "token_means": {str(li): stats.layer_mean(li).detach().cpu() for li in stem_layers},
        }
        torch.save(aux, ckpt_dir / "pc_aux_stats.pt")


def cleanup_old_checkpoints(ckpt_base: Path, keep: int):
    if keep <= 0:
        return
    if not ckpt_base.exists():
        return

    ckpt_dirs = sorted(
        [d for d in ckpt_base.iterdir() if d.is_dir() and re.match(r"\d{10}", d.name)],
        key=lambda p: int(p.name),
    )
    if len(ckpt_dirs) <= keep:
        return

    for old_dir in ckpt_dirs[:-keep]:
        shutil.rmtree(old_dir, ignore_errors=True)
        logger.info(f"Removed old checkpoint directory: {old_dir}")


def save_pc_checkpoint(
    stats: TokenPCStats,
    train_state: PCWarmupState,
    args: ProjectionPCWarmupArgs,
    ckpt_dir: Path,
    final_save: bool,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if final_save:
        # Ensure rare tokens with non-empty buffers are also incorporated before export.
        stats.maybe_flush(force=True)

    _save_local_pc_stats(stats, train_state, ckpt_dir)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(train_state.state_dict(), f)

    if final_save:
        reduce_pc_stats_across_dp(stats, args.token_stat_reduce_chunk_size)
        if args.save_as_stem_shards:
            save_pc_as_stem_shards(
                ckpt_dir=ckpt_dir,
                stats=stats,
                stem_layers=args.stem_layers,
                stem_parallel_size=args.distributed.stem_parallel_size,
                save_mean_stats=args.save_mean_stats,
                save_pc_stats=args.save_pc_stats,
            )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        cleanup_old_checkpoints(ckpt_dir.parent, args.checkpoint.dump.keep)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    logger.info(f"Checkpoint saved to {ckpt_dir}")


def load_pc_checkpoint(stats: TokenPCStats, train_state: PCWarmupState, ckpt_dir: Path):
    if not ckpt_dir.exists():
        return

    ts_path = ckpt_dir / "train_state.json"
    if ts_path.exists():
        with open(ts_path, "r") as f:
            train_state.load_state_dict(json.load(f))

    _load_local_pc_stats(stats, ckpt_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


preemption_flag = {"flag": False}


def set_preemption_flag(signum, frame):
    logger.warning(f"Signal handler called with signal {signum}")
    logger.warning("Preemption! Checkpointing ASAP and exiting.")
    preemption_flag["flag"] = True


def train(args: ProjectionPCWarmupArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(args, tokenizer.n_words)

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)

        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        if args.model_type not in PROJ_MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{args.model_type}'. Available: {list(PROJ_MODEL_REGISTRY.keys())}")
        model_cls, _model_args_cls, build_fsdp, no_recompute = PROJ_MODEL_REGISTRY[args.model_type]

        torch.manual_seed(args.seed)
        with torch.device("meta"):
            model = model_cls(args.model)
        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp(args.model),
            tp_parallelize=None,
            no_recompute_ops=no_recompute(),
        )
        model = model.to_empty(device="cuda")

        assert args.checkpoint.init_ckpt_path, "init_ckpt_path must point to pretrained checkpoint"
        load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
        model.rope_embeddings.reset_parameters()
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info(f"Pretrained model loaded & frozen. Total params: {model_param_count:,}")

        stem_layers = args.stem_layers
        for idx in stem_layers:
            assert 0 <= idx < len(model.layers), f"stem_layer index {idx} out of range"
        assert args.distributed.tp_size == 1, "PC warmup requires tp_size=1"
        capture_ffns = install_ffn_captures(model, stem_layers)

        hidden_dim = model.layers[stem_layers[0]].feed_forward.w3.weight.shape[0]
        for layer_idx in stem_layers[1:]:
            hd = model.layers[layer_idx].feed_forward.w3.weight.shape[0]
            if hd != hidden_dim:
                raise ValueError(f"Inconsistent hidden dim across stem layers: {hidden_dim} vs {hd}")

        shard_start, shard_end, n_shards = _compute_vocab_shard(
            vocab_size=args.model.vocab_size,
            shard_size=args.vocab_shard_size,
            shard_rank=args.vocab_shard_rank,
        )
        logger.info(
            f"Using vocab shard rank={args.vocab_shard_rank if args.vocab_shard_rank >= 0 else 0}/"
            f"{n_shards - 1}: [{shard_start}, {shard_end})"
        )

        device = torch.device("cuda", torch.cuda.current_device())
        stats_dtype = _resolve_dtype(args.stats_dtype)
        collect_dtype = _resolve_dtype(args.collect_dtype)

        token_stats = TokenPCStats(
            vocab_size=args.model.vocab_size,
            hidden_dim=hidden_dim,
            stem_layers=stem_layers,
            device=device,
            shard_start=shard_start,
            shard_end=shard_end,
            stats_dtype=stats_dtype,
            min_count_for_pc_update=args.min_count_for_pc_update,
            max_token_count_collect=args.max_token_count_collect,
            oja_eps=args.oja_eps,
        )

        data_loader_state = init_dataloader_state_from_args(args.data, dp_rank, dp_degree)
        train_state = PCWarmupState(step=0, data_loader_state=data_loader_state)

        ckpt_base = Path(args.checkpoint.path)
        if ckpt_base.exists():
            existing = sorted(
                [d for d in ckpt_base.iterdir() if d.is_dir() and re.match(r"\d{10}", d.name)],
                key=lambda p: int(p.name),
            )
            if existing:
                load_pc_checkpoint(token_stats, train_state, existing[-1])
                logger.info(f"Resumed from checkpoint: {existing[-1]}")

        metric_logger = context_stack.enter_context(MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args))
        data_loader = context_stack.enter_context(build_dataloader_from_args(args.data, state=train_state.data_loader_state))

        torch_profiler = context_stack.enter_context(maybe_run_profiler(args.dump_dir, model, args.profiling))
        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        nwords_since_last_log = 0
        time_last_log = timer()
        gc.disable()
        gc.collect()

        while train_state.step < args.steps:
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)
            input_ids = batch[:, :, 0].cuda()
            target = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            intermediates, _ = collect_w3_outputs(
                model,
                capture_ffns,
                input_ids,
                stem_layers,
                collect_dtype=collect_dtype,
            )

            flat_input_ids = input_ids.reshape(-1)
            ctx = token_stats.begin_batch(flat_input_ids)
            if ctx is not None:
                for layer_idx in stem_layers:
                    token_stats.update_layer_from_batch(
                        layer_idx=layer_idx,
                        w3_flat=intermediates[layer_idx].reshape(-1, hidden_dim),
                        ctx=ctx,
                    )
                token_stats.maybe_flush(force=False)

            train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            if torch_profiler:
                import xformers.profiler
                xformers.profiler.step()

            if every_n_steps(train_state, args.logging.freq):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)
                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
                total_tokens = dp_degree * train_state.step * args.data.batch_size * args.data.seq_len
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "speed": {"wps": wps, "curr_iter_time": curr_iter_time, "data_load_time": data_load_time},
                        "optim": {"total_tokens": total_tokens},
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )
                if get_is_master():
                    metric_logger.log(metrics)
                    logger.info(
                        f"step: {train_state.step}"
                        f"  wps: {wps:.2e}"
                        f"  iter: {curr_iter_time:>7}"
                        f"  data: {data_load_time:>5}"
                        f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                        f"  pow: {gpu_mem_stats.power_draw/1000:.1f} W"
                    )
                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()

            if every_n_steps(train_state, args.checkpoint.dump.every):
                ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                save_pc_checkpoint(
                    stats=token_stats,
                    train_state=train_state,
                    args=args,
                    ckpt_dir=ckpt_dir,
                    final_save=False,
                )

            if preemption_flag["flag"]:
                ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                save_pc_checkpoint(
                    stats=token_stats,
                    train_state=train_state,
                    args=args,
                    ckpt_dir=ckpt_dir,
                    final_save=True,
                )
                requeue_slurm_job()
                sys.exit(0)

        ckpt_dir = ckpt_base / f"{train_state.step:010d}"
        save_pc_checkpoint(
            stats=token_stats,
            train_state=train_state,
            args=args,
            ckpt_dir=ckpt_dir,
            final_save=True,
        )

    log_token_count_coverage_summary(token_stats, args.min_count_for_pc_update)

    gc.collect()
    logger.info("Projection PC warmup (no hooks) complete.")


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    default_cfg = OmegaConf.structured(ProjectionPCWarmupArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
