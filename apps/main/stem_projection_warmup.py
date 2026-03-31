import gc
import json
import logging
import os
import re
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

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
from lingua.optim import build_optimizer
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
from apps.main.train import TrainArgs, TrainState, every_n_steps, validate_train_args
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


@dataclass
class ProjectionWarmupArgs(TrainArgs):
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    stem_layers: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15])
    collect_token_means: bool = True
    token_stat_reduce_chunk_size: int = 4096
    max_token_count_collect: int = 50000
    proj_clip: float = 1.0
    min_token_count_for_residual: int = 100
    eval_max_steps: int = 0
    eval_seed_offset: int = 1


class TokenMeanStats(nn.Module):
    """
    Full-table token stats on GPU.
    - token_counts: [vocab] int64
    - token_sums_l: [vocab, hidden_dim] bf16 sums of w3 outputs per layer
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        stem_layers: List[int],
        device: torch.device,
        max_token_count_collect: int = 50000,
    ):
        super().__init__()
        self.stem_layers = list(stem_layers)
        self.max_token_count_collect = max_token_count_collect
        self._cap_mask: Optional[torch.Tensor] = None
        self.register_buffer("token_counts", torch.zeros(vocab_size, dtype=torch.int64, device=device))
        for layer_idx in stem_layers:
            self.register_buffer(
                f"token_sums_{layer_idx}",
                torch.zeros(vocab_size, hidden_dim, dtype=torch.bfloat16, device=device),
            )

    def layer_sum(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"token_sums_{layer_idx}")

    @torch.no_grad()
    def update(
        self,
        flat_input_ids: torch.Tensor,
        w3_flat: torch.Tensor,
        layer_idx: int,
        add_counts: bool,
    ):
        if self.max_token_count_collect > 0:
            if add_counts:
                self._cap_mask = self.token_counts[flat_input_ids] < self.max_token_count_collect
            mask = self._cap_mask
            if mask is not None and not mask.all():
                flat_input_ids = flat_input_ids[mask]
                w3_flat = w3_flat[mask]
        if add_counts:
            self.token_counts.index_add_(
                0,
                flat_input_ids,
                torch.ones_like(flat_input_ids, dtype=torch.int64),
            )
        self.layer_sum(layer_idx).index_add_(0, flat_input_ids, w3_flat.to(dtype=torch.bfloat16))


def compute_ffn_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]) -> int:
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def sync_projections_across_dp(projections: torch.nn.ModuleList):
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
        return
    for param in projections.parameters():
        torch.distributed.broadcast(param.data, src=0)


def collect_w3_and_tok_emb(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    stem_layer_indices: List[int],
    target: Optional[torch.Tensor] = None,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], torch.Tensor, Optional[torch.Tensor]]:
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
        hooks.append(model.layers[layer_idx].feed_forward.w3.register_forward_hook(_w3_hook(layer_idx)))

    with torch.no_grad():
        output = model(input_ids, target=target)

    for h in hooks:
        h.remove()
    return intermediates, tok_emb_cache["tok_emb"], output.detach() if target is not None else None


def compute_projection_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target: torch.Tensor,
    stem_layer_indices: List[int],
    projections: torch.nn.ModuleList,
    layer_to_proj_idx: Dict[int, int],
) -> torch.Tensor:
    tok_emb_cache: Dict[str, torch.Tensor] = {}
    hooks = []

    def _tok_emb_hook(module, args, output):
        tok_emb_cache["tok_emb"] = output.detach()

    hooks.append(model.tok_embeddings.register_forward_hook(_tok_emb_hook))

    def _make_replace_hook(proj_idx: int):
        def hook_fn(module, inp, out):
            return projections[proj_idx](tok_emb_cache["tok_emb"].float()).to(dtype=out.dtype)
        return hook_fn

    for layer_idx in stem_layer_indices:
        hooks.append(
            model.layers[layer_idx].feed_forward.w3.register_forward_hook(
                _make_replace_hook(layer_to_proj_idx[layer_idx])
            )
        )

    with torch.no_grad():
        proj_nll = model(input_ids, target=target)

    for h in hooks:
        h.remove()
    return proj_nll.detach()


def reduce_token_mean_stats_across_dp(stats: TokenMeanStats, chunk_size: int):
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
        return

    vocab_size = stats.token_counts.shape[0]
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        count_chunk = stats.token_counts[start:end].contiguous()
        torch.distributed.all_reduce(count_chunk, op=torch.distributed.ReduceOp.SUM)
        stats.token_counts[start:end].copy_(count_chunk)

    for layer_idx in stats.stem_layers:
        layer_sum = stats.layer_sum(layer_idx)
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            sum_chunk = layer_sum[start:end].contiguous()
            torch.distributed.all_reduce(sum_chunk, op=torch.distributed.ReduceOp.SUM)
            layer_sum[start:end].copy_(sum_chunk)


def log_token_count_coverage_summary(
    stats: TokenMeanStats,
    min_token_count_for_residual: int,
):
    counts = stats.token_counts.detach().to(dtype=torch.int64).clone()
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)

    vocab_size = int(counts.numel())
    covered = int((counts >= min_token_count_for_residual).sum().item())
    seen_once = int((counts > 0).sum().item())
    max_count = int(counts.max().item()) if vocab_size > 0 else 0
    pct_covered = (100.0 * covered / vocab_size) if vocab_size > 0 else 0.0
    pct_seen_once = (100.0 * seen_once / vocab_size) if vocab_size > 0 else 0.0

    if get_is_master():
        logger.info(
            "Token coverage summary: "
            f"count>0 {seen_once}/{vocab_size} ({pct_seen_once:.2f}%), "
            f"count>={min_token_count_for_residual} {covered}/{vocab_size} ({pct_covered:.2f}%), "
            f"max_count={max_count}"
        )


def save_residuals_as_stem_shards(
    ckpt_dir: Path,
    stats: TokenMeanStats,
    token_embedding_weight: torch.Tensor,
    projections: torch.nn.ModuleList,
    layer_to_proj_idx: Dict[int, int],
    stem_layers: List[int],
    min_token_count_for_residual: int,
    projection_chunk_size: int,
):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    stem_dir.mkdir(parents=True, exist_ok=True)

    counts = stats.token_counts.clamp_min(1).to(dtype=torch.float32).unsqueeze(-1)
    valid_mask = (stats.token_counts >= min_token_count_for_residual).unsqueeze(-1)
    token_embedding_weight = token_embedding_weight.detach().to(device=counts.device, dtype=torch.float32)
    model_sd: Dict[str, torch.Tensor] = {}
    for i, layer_idx in enumerate(stem_layers):
        mean_w3 = stats.layer_sum(layer_idx).to(dtype=torch.float32) / counts
        proj_idx = layer_to_proj_idx[layer_idx]
        projected = torch.empty_like(mean_w3, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, token_embedding_weight.shape[0], projection_chunk_size):
                end = min(start + projection_chunk_size, token_embedding_weight.shape[0])
                projected[start:end] = projections[proj_idx](token_embedding_weight[start:end]).to(dtype=torch.float32)
        residual_means = mean_w3 - projected
        residual_means = torch.where(valid_mask, residual_means, torch.zeros_like(residual_means))
        model_sd[f"stem_embeddings.{i}.weight"] = residual_means.to(dtype=torch.bfloat16).cpu()

    # No MP sharding: store full table in mp0 shard file.
    shard_path = stem_dir / STEM_MODEL_FILE_TEMPLATE.format(mp_rank=0)
    torch.save(model_sd, shard_path)
    logger.info(
        f"Saved full residual STEM tables to {shard_path} "
        f"(mean_w3 - projection(tok_emb), zeroed where token_count < {min_token_count_for_residual})"
    )


def cleanup_old_checkpoints(ckpt_base: Path, keep: int):
    """
    Remove older numbered checkpoint directories, keeping only the latest `keep`.
    """
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


def save_projection_checkpoint(
    projections: torch.nn.ModuleList,
    layer_to_proj_idx: Dict[int, int],
    token_embedding_weight: torch.Tensor,
    train_state: TrainState,
    args: ProjectionWarmupArgs,
    ckpt_dir: Path,
    token_stats: Optional[TokenMeanStats],
    save_stem_shards: bool = False,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if token_stats is not None:
        reduce_token_mean_stats_across_dp(token_stats, args.token_stat_reduce_chunk_size)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(
            {name: param.detach().cpu() for name, param in projections.named_parameters()},
            ckpt_dir / "projections.pt",
        )
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(train_state.state_dict(), f)

    if token_stats is not None:
        if save_stem_shards:
            save_residuals_as_stem_shards(
                ckpt_dir,
                token_stats,
                token_embedding_weight,
                projections,
                layer_to_proj_idx,
                args.stem_layers,
                args.min_token_count_for_residual,
                args.token_stat_reduce_chunk_size,
            )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        cleanup_old_checkpoints(ckpt_dir.parent, args.checkpoint.dump.keep)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    logger.info(f"Checkpoint saved to {ckpt_dir}")


def load_projection_checkpoint(
    projections: torch.nn.ModuleList,
    train_state: TrainState,
    ckpt_dir: Path,
):
    if not ckpt_dir.exists():
        return

    device = torch.device("cuda", torch.cuda.current_device())
    proj_path = ckpt_dir / "projections.pt"
    if proj_path.exists():
        proj_sd = torch.load(proj_path, map_location=device)
        with torch.no_grad():
            for name, param in projections.named_parameters():
                if name in proj_sd:
                    param.copy_(proj_sd[name])

    ts_path = ckpt_dir / "train_state.json"
    if ts_path.exists():
        with open(ts_path, "r") as f:
            train_state.load_state_dict(json.load(f))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def run_projection_eval(
    model: torch.nn.Module,
    projections: torch.nn.ModuleList,
    layer_to_proj_idx: Dict[int, int],
    stem_layers: List[int],
    eval_loader,
    eval_data_loader_state: PackTokensState,
    eval_max_steps: int,
) -> Tuple[Dict[str, float], PackTokensState]:
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
            proj_nll = compute_projection_nll(model, input_ids, target, stem_layers, projections, layer_to_proj_idx)
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


preemption_flag = {"flag": False}


def set_preemption_flag(signum, frame):
    logger.warning(f"Signal handler called with signal {signum}")
    logger.warning("Preemption! Checkpointing ASAP and exiting.")
    preemption_flag["flag"] = True


def train(args: ProjectionWarmupArgs):
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

        saved_compile = args.distributed.compile
        args.distributed.compile = False
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp(args.model),
            tp_parallelize=None,
            no_recompute_ops=no_recompute(),
        )
        args.distributed.compile = saved_compile
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
        assert args.distributed.tp_size == 1, "Projection warmup requires tp_size=1"

        hidden_dim = compute_ffn_hidden_dim(args.model.dim, args.model.multiple_of, args.model.ffn_dim_multiplier)
        device = torch.device("cuda", torch.cuda.current_device())
        projections = torch.nn.ModuleList(
            [torch.nn.Linear(args.model.dim, hidden_dim, bias=False, device=device) for _ in stem_layers]
        )
        for proj in projections:
            torch.nn.init.xavier_normal_(proj.weight)
        for param in projections.parameters():
            param.requires_grad = True
        sync_projections_across_dp(projections)
        layer_to_proj_idx = {layer_idx: i for i, layer_idx in enumerate(stem_layers)}

        token_stats = None
        if args.collect_token_means:
            token_stats = TokenMeanStats(
                args.model.vocab_size,
                hidden_dim,
                stem_layers,
                device=device,
            )

        optimizer, scheduler = build_optimizer(projections, args.optim, args.steps)

        data_loader_state = init_dataloader_state_from_args(args.data, dp_rank, dp_degree)
        train_state = TrainState(step=0, acc_step=0, data_loader_state=data_loader_state, scheduler=scheduler)

        ckpt_base = Path(args.checkpoint.path)
        if ckpt_base.exists():
            existing = sorted(
                [d for d in ckpt_base.iterdir() if d.is_dir() and re.match(r"\d{10}", d.name)],
                key=lambda p: int(p.name),
            )
            if existing:
                load_projection_checkpoint(projections, train_state, existing[-1])

        metric_logger = context_stack.enter_context(MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args))
        data_loader = context_stack.enter_context(build_dataloader_from_args(args.data, state=train_state.data_loader_state))
        eval_data_loader = None
        eval_data_loader_state = None
        if args.eval_max_steps > 0:
            eval_data_args = replace(args.data, seed=args.data.seed + args.eval_seed_offset, load_async=False)
            eval_data_loader_state = init_dataloader_state_from_args(eval_data_args, dp_rank, dp_degree)
            eval_data_loader = context_stack.enter_context(
                build_dataloader_from_args(eval_data_args, state=eval_data_loader_state)
            )

        torch_profiler = context_stack.enter_context(maybe_run_profiler(args.dump_dir, model, args.profiling))
        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        nwords_since_last_log = 0
        time_last_log = timer()
        gc.disable()
        gc.collect()
        saved = False

        while train_state.step < args.steps:
            train_state.acc_step = (train_state.acc_step + 1) % args.grad_acc_steps
            curr_lr = float(optimizer.param_groups[0]["lr"])
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

            intermediates, tok_emb, original_nll = collect_w3_and_tok_emb(model, input_ids, stem_layers, target=target)

            proj_nll = compute_projection_nll(model, input_ids, target, stem_layers, projections, layer_to_proj_idx)

            total_loss = torch.tensor(0.0, device=device)
            per_layer_losses: Dict[int, float] = {}
            tokenwise_up_proj_out: Dict[int, torch.Tensor] = {}
            for layer_idx in stem_layers:
                proj_idx = layer_to_proj_idx[layer_idx]
                proj_out = projections[proj_idx](tok_emb.float())
                tgt_up = intermediates[layer_idx]["w3"].float().detach()
                layer_loss = F.mse_loss(proj_out, tgt_up)
                total_loss = total_loss + layer_loss
                per_layer_losses[layer_idx] = float(layer_loss.detach().item())
                if token_stats is not None:
                    tokenwise_up_proj_out[layer_idx] = tgt_up
            total_loss = total_loss / len(stem_layers)
            total_loss_for_log = total_loss.detach()

            if args.grad_acc_steps > 1:
                total_loss = total_loss / args.grad_acc_steps
            total_loss.backward()

            grad_norm = -1.0
            if train_state.acc_step == 0:
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    for param in projections.parameters():
                        if param.grad is not None:
                            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)
                proj_params = [p for p in projections.parameters() if p.grad is not None]
                if proj_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        proj_params, max_norm=args.proj_clip, foreach=False
                    ).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            if token_stats is not None:
                flat_input_ids = input_ids.reshape(-1)
                for li, layer_idx in enumerate(stem_layers):
                    token_stats.update(
                        flat_input_ids,
                        tokenwise_up_proj_out[layer_idx].reshape(-1, hidden_dim).bfloat16(),
                        layer_idx,
                        add_counts=(li == 0),
                    )

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            if torch_profiler:
                import xformers.profiler
                xformers.profiler.step()

            if every_n_steps(train_state, args.logging.freq, acc_step=None if args.logging.acc_freq else 0, acc_freq=args.logging.acc_freq):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)
                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
                total_acc_steps = args.grad_acc_steps * train_state.step + train_state.acc_step
                total_tokens = dp_degree * total_acc_steps * args.data.batch_size * args.data.seq_len
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {"wps": wps, "curr_iter_time": curr_iter_time, "data_load_time": data_load_time},
                        "optim": {"proj_grad_norm": grad_norm, "proj_lr": curr_lr, "total_tokens": total_tokens},
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )
                to_sync: Dict[str, float] = {
                    "loss/mse_up_total": total_loss_for_log.item(),
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

            saved = False
            if every_n_steps(train_state, args.checkpoint.dump.every, acc_step=0):
                ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                save_projection_checkpoint(
                    projections,
                    layer_to_proj_idx,
                    model.tok_embeddings.weight,
                    train_state,
                    args,
                    ckpt_dir,
                    token_stats,
                    save_stem_shards=False,
                )
                saved = True

            if eval_data_loader is not None and every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
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

            if preemption_flag["flag"]:
                ckpt_dir = ckpt_base / f"{train_state.step:010d}"
                save_projection_checkpoint(
                    projections,
                    layer_to_proj_idx,
                    model.tok_embeddings.weight,
                    train_state,
                    args,
                    ckpt_dir,
                    token_stats,
                    save_stem_shards=True,
                )
                requeue_slurm_job()
                sys.exit(0)

        ckpt_dir = ckpt_base / f"{train_state.step:010d}"
        save_projection_checkpoint(
            projections,
            layer_to_proj_idx,
            model.tok_embeddings.weight,
            train_state,
            args,
            ckpt_dir,
            token_stats,
            save_stem_shards=True,
        )

    if token_stats is not None:
        log_token_count_coverage_summary(
            token_stats,
            args.min_token_count_for_residual,
        )

    gc.collect()
    logger.info("Projection warmup complete.")


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    default_cfg = OmegaConf.structured(ProjectionWarmupArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
                
            