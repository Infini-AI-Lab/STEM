from copy import deepcopy
import gc
import logging
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Optional

import xformers.profiler
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.optim import AdamW, lr_scheduler

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.stem_checkpoint import StemCheckpointManager, load_from_checkpoint
from lingua.data import (
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from lingua.distributed import (
    check_model_value_range,
    clean_env,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    init_signal_handler,
    parallelize_model,
    requeue_slurm_job,
    setup_env,
    setup_torch_distributed,
)
from lingua.logger import init_logger
from lingua.metrics import GPUMemoryMonitor, MetricLogger, get_num_params
from lingua.optim import build_lr_fn
from lingua.probe import AutoProbeD
from lingua.profiling import maybe_run_profiler
from lingua.stool import StoolArgs, launch_job
from lingua.tokenizer import build_tokenizer

from apps.main.stem import (
    LMTransformer,
    OLMo3LMTransformer,
    Qwen3LMTransformer,
    StemLMTransformer,
    StemLMTransformerArgs,
    STEM_MODEL_REGISTRY,
)
from apps.main.stem_train import sync_stem_embeddings_across_dp
from apps.main.train import TrainState, TrainArgs, every_n_steps, validate_train_args
from lingua.stem_dist_utils import (
    get_stem_data_parallel_group,
    get_stem_data_parallel_world_size,
)
from lingua.transformer import cross_entropy

logger = logging.getLogger()


def compute_ffn_hidden_dim(dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]) -> int:
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class ReparamLMTransformer(LMTransformer):
    """LMTransformer variant that can pass tok_emb to stem callback."""

    def __init__(self, args: StemLMTransformerArgs):
        super().__init__(args)
        hidden_dim = compute_ffn_hidden_dim(args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.projections = nn.ModuleList(
            [nn.Linear(args.dim, hidden_dim, bias=False) for _ in range(len(self.stem_layers))]
        )
        self._layer_to_stem_idx = {
            layer_idx: stem_idx for stem_idx, layer_idx in enumerate(self.stem_layers)
        }

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
        stem_embeddings_fn: Optional[Callable[..., torch.Tensor]] = None,
    ):
        _, seqlen = token_values.shape

        tok_emb = self.tok_embeddings(token_values)
        h = tok_emb

        mask = (
            mask
            if mask is not None
            else self._create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            if i in self.stem_layers:
                if stem_embeddings_fn is not None:
                    stem_idx = self._layer_to_stem_idx[i]
                    residual = stem_embeddings_fn(i, token_values)
                    proj = self.projections[stem_idx](tok_emb).to(dtype=residual.dtype)
                    y = residual + proj
                    h = layer(h, freq_cis, y=y, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
                else:
                    h = layer(h, freq_cis, y=None, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
            else:
                h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        return logits


class Qwen3ReparamLMTransformer(Qwen3LMTransformer):
    def __init__(self, args: StemLMTransformerArgs):
        super().__init__(args)
        hidden_dim = compute_ffn_hidden_dim(args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.projections = nn.ModuleList(
            [nn.Linear(args.dim, hidden_dim, bias=False) for _ in range(len(self.stem_layers))]
        )
        self._layer_to_stem_idx = {
            layer_idx: stem_idx for stem_idx, layer_idx in enumerate(self.stem_layers)
        }

    forward = ReparamLMTransformer.forward


class OLMo3ReparamLMTransformer(OLMo3LMTransformer):
    def __init__(self, args: StemLMTransformerArgs):
        super().__init__(args)
        hidden_dim = compute_ffn_hidden_dim(args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.projections = nn.ModuleList(
            [nn.Linear(args.dim, hidden_dim, bias=False) for _ in range(len(self.stem_layers))]
        )
        self._layer_to_stem_idx = {
            layer_idx: stem_idx for stem_idx, layer_idx in enumerate(self.stem_layers)
        }

    forward = ReparamLMTransformer.forward


class ReparamStemLMTransformer(StemLMTransformer):
    """
    STEM model with an additive projection branch:
      y_l(token) = residual_stem_l(token) + A_l(tok_embedding(token))
    """

    _lm_transformer_cls = ReparamLMTransformer

    # StemLMTransformer.forward is sufficient: it provides residual stem values,
    # and ReparamLMTransformer combines them with projection outputs.

class Qwen3ReparamStemLMTransformer(ReparamStemLMTransformer):
    _lm_transformer_cls = Qwen3ReparamLMTransformer


class OLMo3ReparamStemLMTransformer(ReparamStemLMTransformer):
    _lm_transformer_cls = OLMo3ReparamLMTransformer


@dataclass
class StemReparamTrainArgs(TrainArgs):
    model: StemLMTransformerArgs = field(default_factory=StemLMTransformerArgs)

    stem_lr: Optional[float] = None
    stem_weight_decay: Optional[float] = None
    stem_warmup: Optional[int] = None
    stem_scheduler: Optional[str] = None
    stem_lr_min_ratio: Optional[float] = None

    proj_lr: Optional[float] = None
    proj_weight_decay: Optional[float] = None
    proj_warmup: Optional[int] = None
    proj_scheduler: Optional[str] = None
    proj_lr_min_ratio: Optional[float] = None
    proj_clip: float = 1.0


REPARAM_MODEL_REGISTRY = {
    "llama": ReparamStemLMTransformer,
    "qwen3": Qwen3ReparamStemLMTransformer,
    "olmo3": OLMo3ReparamStemLMTransformer,
}


preemption_flag = {"flag": False}


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def sync_non_fsdp_grads(model: ReparamStemLMTransformer):
    if get_stem_data_parallel_world_size() <= 1:
        return
    dp_group = get_stem_data_parallel_group()
    for p in model.stem_embeddings.parameters():
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, group=dp_group)


def _register_reparam_models_for_eval():
    from apps.main import stem as stem_module
    from apps.main.stem_eval import STEM_MODEL_REGISTRY as STEM_EVAL_REG

    # Reuse FSDP/no-recompute/flops helpers from base STEM registry.
    for base_key, reparam_cls in REPARAM_MODEL_REGISTRY.items():
        stem_args_cls = STEM_MODEL_REGISTRY[base_key][1]
        build_plan = STEM_MODEL_REGISTRY[base_key][2]
        no_recompute = STEM_MODEL_REGISTRY[base_key][3]
        flop_fn = STEM_MODEL_REGISTRY[base_key][4]
        reparam_key = f"{base_key}_reparam"
        entry = (reparam_cls, stem_args_cls, build_plan, no_recompute, flop_fn)
        stem_module.STEM_MODEL_REGISTRY[reparam_key] = entry
        STEM_EVAL_REG[reparam_key] = entry


def train(args: StemReparamTrainArgs):
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
        logger.info(f"Starting job: {args.name}")

        # Dataloader rank/degree
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()
        logger.info(f"Running on dp rank: {dp_rank}, dp size: {dp_degree}")

        from lingua.stem_dist_utils import initialize_stem_process_group
        initialize_stem_process_group(args.distributed.stem_parallel_size)

        if args.model_type not in REPARAM_MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{args.model_type}'. Available: {list(REPARAM_MODEL_REGISTRY.keys())}")
        if args.model_type not in STEM_MODEL_REGISTRY:
            raise ValueError(f"Base model_type '{args.model_type}' missing from STEM registry")
        reparam_model_cls = REPARAM_MODEL_REGISTRY[args.model_type]
        build_plan = STEM_MODEL_REGISTRY[args.model_type][2]
        get_no_recompute_ops = STEM_MODEL_REGISTRY[args.model_type][3]
        get_num_flop_per_token = STEM_MODEL_REGISTRY[args.model_type][4]

        torch.manual_seed(args.seed)
        with torch.device("meta"):
            model = reparam_model_cls(args.model)
        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=list(build_plan(args.model))
            + [(f"lm_transformer.projections.{i}", True) for i in range(len(model.lm_transformer.projections))],
            tp_parallelize=None,
            no_recompute_ops=get_no_recompute_ops(),
        )
        model = model.to_empty(device="cuda")

        # Enforced init: load backbone+projection params via distcp and residual stem tables via stem_shards.
        if not args.checkpoint.init_ckpt_path:
            raise ValueError(
                "checkpoint.init_ckpt_path is required for stem_reparam_train "
                "(must contain distcp model with projections + stem_shards)."
            )
        if args.checkpoint.continue_training_from_init:
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, optimizer=None, model_key="model")
        else:
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
        model.rope_embeddings.reset_parameters()
        check_model_value_range(model, range=10.0, std=1.0)

        hidden_dim = compute_ffn_hidden_dim(args.model.dim, args.model.multiple_of, args.model.ffn_dim_multiplier)
        logger.info(f"Model size: {model_param_count:,}; hidden_dim={hidden_dim}")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # Build optimizers
        proj_lr = args.proj_lr if args.proj_lr is not None else args.optim.lr
        proj_wd = args.proj_weight_decay if args.proj_weight_decay is not None else args.optim.weight_decay
        proj_param_ids = set(id(p) for p in model.lm_transformer.projections.parameters())
        lm_only_params = [p for p in model.lm_transformer.parameters() if id(p) not in proj_param_ids]
        lm_optimizer = AdamW(
            [
                {
                    "params": lm_only_params,
                    "lr": args.optim.lr,
                    "weight_decay": args.optim.weight_decay,
                },
                {
                    "params": list(model.lm_transformer.projections.parameters()),
                    "lr": proj_lr,
                    "weight_decay": proj_wd,
                },
            ],
            betas=(args.optim.beta1, args.optim.beta2),
            eps=args.optim.epsilon,
            fused=True,
        )
        stem_lr = args.stem_lr if args.stem_lr is not None else args.optim.lr
        stem_wd = args.stem_weight_decay if args.stem_weight_decay is not None else args.optim.weight_decay
        stem_optimizer = AdamW(
            model.stem_embeddings.parameters(),
            lr=stem_lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=stem_wd,
            eps=args.optim.epsilon,
            fused=False,
        )
        logger.info(
            f"Optimizers: lm_lr={args.optim.lr}, stem_lr={stem_lr}, "
            f"proj_lr={proj_lr}"
        )

        # Build schedulers
        proj_sched_args = replace(
            args.optim,
            warmup=args.proj_warmup if args.proj_warmup is not None else args.optim.warmup,
            scheduler=args.proj_scheduler if args.proj_scheduler is not None else args.optim.scheduler,
            lr_min_ratio=args.proj_lr_min_ratio if args.proj_lr_min_ratio is not None else args.optim.lr_min_ratio,
        )
        lm_scheduler = lr_scheduler.LambdaLR(
            lm_optimizer,
            [
                build_lr_fn(args.optim, args.steps),
                build_lr_fn(proj_sched_args, args.steps),
            ],
        )
        stem_sched_args = replace(
            args.optim,
            warmup=args.stem_warmup if args.stem_warmup is not None else args.optim.warmup,
            scheduler=args.stem_scheduler if args.stem_scheduler is not None else args.optim.scheduler,
            lr_min_ratio=args.stem_lr_min_ratio if args.stem_lr_min_ratio is not None else args.optim.lr_min_ratio,
        )
        stem_scheduler = lr_scheduler.LambdaLR(stem_optimizer, build_lr_fn(stem_sched_args, args.steps))
        optimizer = {"lm": lm_optimizer, "stem": stem_optimizer}
        scheduler = {"lm": lm_scheduler, "stem": stem_scheduler}

        # Data loader state
        data_rank = dp_rank
        data_world_size = dp_degree
        if args.data.node_local:
            local_rank_env = os.environ.get("LOCAL_RANK")
            local_world_env = os.environ.get("LOCAL_WORLD_SIZE")
            assert local_rank_env is not None and local_world_env is not None
            data_rank = int(local_rank_env)
            data_world_size = int(local_world_env)
        data_loader_state = init_dataloader_state_from_args(args.data, data_rank, data_world_size)
        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        # Checkpoint manager
        checkpoint = StemCheckpointManager.instantiate_and_make_dir(args.checkpoint, train_stage=None)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        stage_start_step = train_state.step
        if args.stage_steps is None:
            target_step = args.steps
        else:
            target_step = min(args.steps, stage_start_step + args.stage_steps)
            logger.info(
                f"Stage-limited: start={stage_start_step}, stage_steps={args.stage_steps}, target={target_step}"
            )

        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl" if (dp_rank % 128 == 0) else None),
            )

        gc.disable()
        model.train()
        metric_logger = context_stack.enter_context(MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args))
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(args.data, state=train_state.data_loader_state)
        )
        torch_profiler = context_stack.enter_context(maybe_run_profiler(args.dump_dir, model, args.profiling))

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        saved = False
        while train_state.step < target_step:
            train_state.acc_step = (train_state.acc_step + 1) % args.grad_acc_steps
            curr_lr = float(optimizer["lm"].param_groups[0]["lr"])
            curr_stem_lr = float(optimizer["stem"].param_groups[0]["lr"])
            curr_proj_lr = float(optimizer["lm"].param_groups[1]["lr"])

            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)
            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()
            bsz, seqlen = labels.shape

            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                assert next(model.parameters()).grad is None
                with probe:
                    probe.metadata = {"it": train_state.step, "global_step": train_state.step, "loop": "reparam"}
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = model(input_ids[:probe_bsz, :probe_seq], labels[:probe_bsz, :probe_seq])
                    probe_loss.backward()
                    optimizer["lm"].zero_grad()
                    optimizer["stem"].zero_grad()
                assert next(model.parameters()).grad is None

            loss = model(input_ids, labels)
            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)
            loss = loss / args.grad_acc_steps
            loss.backward()
            loss = loss.detach() * args.grad_acc_steps

            grad_norm = -1.0
            stem_grad_norm = -1.0
            proj_grad_norm = -1.0
            if train_state.acc_step == 0:
                lm_params = [p for p in model.lm_transformer.parameters() if p.grad is not None]
                if lm_params:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        lm_params, max_norm=args.optim.clip, foreach=True
                    )
                    grad_norm = (
                        grad_norm_t.full_tensor() if isinstance(grad_norm_t, DTensor) else grad_norm_t
                    ).item()

                stem_params = [p for p in model.stem_embeddings.parameters() if p.grad is not None]
                if stem_params:
                    stem_grad_norm = torch.nn.utils.clip_grad_norm_(
                        stem_params, max_norm=args.optim.clip, foreach=False
                    ).item()

                proj_params = [p for p in model.lm_transformer.projections.parameters() if p.grad is not None]
                if proj_params:
                    proj_grad_norm = torch.nn.utils.clip_grad_norm_(
                        proj_params, max_norm=args.proj_clip, foreach=False
                    ).item()

                sync_non_fsdp_grads(model)

                optimizer["lm"].step()
                optimizer["stem"].step()
                scheduler["lm"].step()
                scheduler["stem"].step()
                optimizer["lm"].zero_grad()
                optimizer["stem"].zero_grad()
                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            if torch_profiler:
                xformers.profiler.step()

            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)
                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
                total_acc_steps = args.grad_acc_steps * train_state.step + train_state.acc_step
                total_tokens = dp_degree * total_acc_steps * args.data.batch_size * args.data.seq_len
                flops = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                optim_dict = {
                    "grad_norm": grad_norm,
                    "stem_grad_norm": stem_grad_norm,
                    "proj_grad_norm": proj_grad_norm,
                    "lr": curr_lr,
                    "stem_lr": curr_stem_lr,
                    "proj_lr": curr_proj_lr,
                    "total_tokens": total_tokens,
                }
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": flops,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": optim_dict,
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )
                metrics.update(dist_mean_dict({"loss/out": loss.item()}))
                if get_is_master():
                    metric_logger.log(metrics)
                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step} acc: {train_state.acc_step} loss: {loss.item():.4f} "
                    f"lr: {curr_lr:.2e} stem_lr: {curr_stem_lr:.2e} proj_lr: {curr_proj_lr:.2e} "
                    f"grad: {grad_norm:.2e} stem_grad: {stem_grad_norm:.2e} proj_grad: {proj_grad_norm:.2e} "
                    f"wps: {wps:.2e} iter: {curr_iter_time} data: {data_load_time} "
                    f"mem: {gpu_mem_stats.max_active_pct:.0f}%"
                )

            saved = False
            if every_n_steps(train_state, args.checkpoint.dump.every, acc_step=0) or every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ):
                saved = checkpoint.save(model, optimizer, train_state, args, device_mesh=world_mesh)

            if args.eval is not None and (
                every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0)
                or every_n_steps(train_state, target_step, acc_step=0)
            ):
                from apps.main.stem_eval import launch_stem_eval, EVAL_FOLDER_NAME, StemEvalArgs

                _register_reparam_models_for_eval()
                eval_args = dataclass_from_dict(StemEvalArgs, args.eval)
                eval_args.model_type = f"{args.model_type}_reparam"
                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.stem_parallel_size = args.distributed.stem_parallel_size
                eval_args.dump_dir = str(
                    os.path.join(args.dump_dir, "evals", EVAL_FOLDER_NAME.format(train_state.step))
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    launch_stem_eval(eval_args)
                elif get_is_master():
                    if getattr(args.logging, "wandb", None) is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                    assert args.async_eval_gpus > 0
                    logger.info(f"Launching evals on {args.async_eval_gpus} gpus")
                    with clean_env():
                        launch_job(
                            StoolArgs(
                                asdict(eval_args),
                                script="apps.main.stem_eval",
                                copy_code=False,
                                nodes=args.async_eval_gpus // 8,
                                qos="lowest",
                            )
                        )

            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(model, optimizer, train_state, args, device_mesh=world_mesh)
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        checkpoint.save(model, optimizer, train_state, args, device_mesh=world_mesh)
    gc.collect()


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(StemReparamTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    train(cfg)


if __name__ == "__main__":
    main()