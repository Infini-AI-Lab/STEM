# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Training script for Selective-State-Space-STEM models.

Usage::

    torchrun --nproc-per-node 8 -m apps.main.stem_sss_train \\
        config=apps/main/configs/stem_sss_llama3_1B_midfine.yaml
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
from typing import Any, Dict, List, Optional


import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
import xformers.profiler
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.stem_checkpoint import StemCheckpointManager, load_from_checkpoint
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
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import build_optimizer
from lingua.logger import init_logger
from lingua.tokenizer import build_tokenizer
from lingua.profiling import maybe_run_profiler
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

from apps.main.train import TrainArgs, TrainState, validate_train_args, every_n_steps
from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_sss import (
    SSSStemLMTransformerArgs,
    SELECTIVE_IIR_STEM_MODEL_REGISTRY,
)

# Register selective-IIR model types so that launch_stem_eval can resolve them.
STEM_MODEL_REGISTRY.update(SELECTIVE_IIR_STEM_MODEL_REGISTRY)

from lingua.stem_dist_utils import (
    initialize_stem_process_group,
    get_stem_data_parallel_group,
    get_stem_data_parallel_rank,
    get_stem_data_parallel_world_size,
)


SELECTIVE_IIR_PARAMS_FILE = "selective_iir_params.pt"


@torch.no_grad()
def _load_selective_iir_params(model, stem_shards_dir: Path):
    """Load Selective-IIR parameters (w_mu, w_alpha, m0) from checkpoint.

    Falls back to ``reset_parameters()`` from model config if the file
    does not exist (e.g. initialising from a base STEM checkpoint).
    """
    iir_path = stem_shards_dir / SELECTIVE_IIR_PARAMS_FILE
    if iir_path.exists():
        iir_sd = torch.load(iir_path, map_location="cpu")
        for i, sel_iir_mem in enumerate(model.selective_iir_memories):
            prefix = f"selective_iir_memories.{i}"
            for pname, param in sel_iir_mem.named_parameters():
                key = f"{prefix}.{pname}"
                if key in iir_sd:
                    param.data.copy_(iir_sd[key].to(param.device))
                    logger.info(
                        f"Loaded {key}: shape={param.shape}, "
                        f"values={param.data.flatten()[:4].tolist()}"
                    )
                else:
                    logger.warning(f"{key} not found in {iir_path}")
            for bname, buf in sel_iir_mem.named_buffers():
                key = f"{prefix}.{bname}"
                if key in iir_sd:
                    buf.data.copy_(iir_sd[key].to(buf.device))
                    logger.info(f"Loaded buffer {key}")
        logger.info(f"Loaded Selective-IIR parameters from {iir_path}")
    else:
        logger.info(
            f"No {SELECTIVE_IIR_PARAMS_FILE} in {stem_shards_dir}, "
            "falling back to reset_parameters() from model config"
        )
        for sel_iir_mem in model.selective_iir_memories:
            sel_iir_mem.reset_parameters()


def sync_stem_params_across_dp(model):
    """Broadcast stem_embeddings and selective_iir_memories weights from
    dp_rank 0 to all STEM data-parallel ranks.
    """
    if get_stem_data_parallel_world_size() <= 1:
        return

    dp_group = get_stem_data_parallel_group()
    src_rank = torch.distributed.get_global_rank(dp_group, 0)
    for param in model.stem_parameters():
        torch.distributed.broadcast(param.data, src=src_rank, group=dp_group)
    logger.info(
        "Synchronized stem_embeddings and selective_iir_memories "
        "weights across STEM data-parallel ranks"
    )


import wandb

logger = logging.getLogger()


@dataclass
class SelectiveIIRStemTrainArgs(TrainArgs):
    model: SSSStemLMTransformerArgs = field(
        default_factory=SSSStemLMTransformerArgs,
    )

    # Separate learning rate and weight decay for stem parameters
    # (stem_embeddings + selective_iir_memories).
    stem_lr: Optional[float] = None
    stem_weight_decay: Optional[float] = None


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def train(args: SelectiveIIRStemTrainArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(
            args.data.tokenizer.name, args.data.tokenizer.path,
        )
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

        # build dataloader
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

        # Initialize stem process groups for ParallelEmbedding
        initialize_stem_process_group(args.distributed.stem_parallel_size)
        logger.info(
            f"Initialized stem process groups with parallel size: "
            f"{args.distributed.stem_parallel_size}"
        )

        # ---- Resolve model class & helpers from the registry ----
        if args.model_type not in SELECTIVE_IIR_STEM_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{args.model_type}'. "
                f"Available: {list(SELECTIVE_IIR_STEM_MODEL_REGISTRY.keys())}"
            )
        (
            stem_model_cls, _stem_args_cls,
            _build_fsdp_plan,
            _get_no_recompute_ops, _get_num_flop_per_token,
        ) = SELECTIVE_IIR_STEM_MODEL_REGISTRY[args.model_type]
        logger.info(
            f"Using Selective-IIR-STEM model type: {args.model_type} "
            f"({stem_model_cls.__name__})"
        )

        torch.manual_seed(args.seed)
        logger.info("Building model")

        with torch.device("meta"):
            model = stem_model_cls(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=_build_fsdp_plan(args.model),
            tp_parallelize=None,
            no_recompute_ops=_get_no_recompute_ops(),
        )

        model = model.to_empty(device="cuda")

        # Ensure non-FSDP parameters require gradients
        for name, param in model.stem_embeddings.named_parameters():
            param.requires_grad = True
        for name, param in model.selective_iir_memories.named_parameters():
            param.requires_grad = True
        logger.info(
            "Ensured stem_embeddings and selective_iir_memories "
            "parameters require gradients"
        )

        if not args.checkpoint.init_ckpt_path:
            with torch.random.fork_rng(
                devices=[torch.cuda.current_device()]
            ):
                torch.manual_seed(args.model.seed)
                model.init_weights()
            sync_stem_params_across_dp(model)

        # Verify stem_embeddings after init
        for i, embedding in enumerate(model.stem_embeddings):
            for param_name, param in embedding.named_parameters():
                if param.numel() > 0:
                    is_zero = (param.abs().max() == 0).item()
                    param_norm = param.norm().item()
                    logger.info(
                        f"stem_embeddings[{i}].{param_name}: "
                        f"device={param.device}, shape={param.shape}, "
                        f"norm={param_norm:.6f}, is_zero={is_zero}"
                    )
                    if is_zero:
                        logger.error(
                            f"ERROR: stem_embeddings[{i}].{param_name} "
                            f"is still all zeros after init_weights()!"
                        )

        # Verify selective_iir_memories after init
        for i, sel_iir in enumerate(model.selective_iir_memories):
            mu_bias = sel_iir.mu_bias_value
            alpha_bias = sel_iir.alpha_bias_value
            logger.info(
                f"selective_iir_memories[{i}]: "
                f"mu_bias={mu_bias:.4f} (σ→{torch.sigmoid(torch.tensor(mu_bias)).item():.4f}), "
                f"alpha_bias={alpha_bias:.4f}, "
                f"w_mu_norm={sel_iir.w_mu.weight.norm().item():.6f}, "
                f"w_alpha_norm={sel_iir.w_alpha.weight.norm().item():.6f}"
            )

        check_model_value_range(model, range=10.0, std=1.0)

        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} "
            f"({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        from torch.optim import AdamW
        from lingua.optim import build_lr_fn

        # Optimizer for lm_transformer (DTensors via FSDP)
        lm_optimizer = AdamW(
            model.lm_transformer.parameters(),
            lr=args.optim.lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
            eps=args.optim.epsilon,
            fused=True,
        )

        # Optimizer for stem_embeddings + selective_iir_memories
        stem_lr = (
            args.stem_lr if args.stem_lr is not None else args.optim.lr
        )
        stem_wd = (
            args.stem_weight_decay
            if args.stem_weight_decay is not None
            else args.optim.weight_decay
        )
        logger.info(f"Stem optimizer: lr={stem_lr}, weight_decay={stem_wd}")
        stem_optimizer = AdamW(
            list(model.stem_parameters()),
            lr=stem_lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=stem_wd,
            eps=args.optim.epsilon,
            fused=False,
        )

        # Create schedulers for both optimizers
        lr_fn = build_lr_fn(args.optim, args.steps)
        lm_scheduler = lr_scheduler.LambdaLR(lm_optimizer, lr_fn)
        stem_scheduler = lr_scheduler.LambdaLR(stem_optimizer, lr_fn)

        optimizer = {"lm": lm_optimizer, "stem": stem_optimizer}
        scheduler = {"lm": lm_scheduler, "stem": stem_scheduler}

        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree,
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = StemCheckpointManager.instantiate_and_make_dir(
            args.checkpoint,
        )

        if args.checkpoint.init_ckpt_path:
            logger.info(
                f"Loading initial model from "
                f"{args.checkpoint.init_ckpt_path}"
            )
            if args.checkpoint.continue_training_from_init:
                load_from_checkpoint(
                    args.checkpoint.init_ckpt_path,
                    model,
                    optimizer=optimizer,
                    model_key="model",
                )
            else:
                load_from_checkpoint(
                    args.checkpoint.init_ckpt_path,
                    model,
                    model_key="model",
                )
            model.rope_embeddings.reset_parameters()

            # Load selective-IIR params from checkpoint (or config defaults)
            stem_shards_dir = (
                Path(args.checkpoint.init_ckpt_path) / "stem_shards"
            )
            _load_selective_iir_params(model, stem_shards_dir)
            if stem_shards_dir.exists() and any(
                stem_shards_dir.glob("stem_model_mp*.pt")
            ):
                logger.info(
                    "Pre-computed stem embeddings found in init "
                    "checkpoint, skipping random reset"
                )
            else:
                logger.info(
                    "No pre-computed stem embeddings in init "
                    "checkpoint, initializing randomly"
                )
                with torch.random.fork_rng(
                    devices=[torch.cuda.current_device()]
                ):
                    torch.manual_seed(args.model.seed)
                    model.reset_stem_embeddings()
            sync_stem_params_across_dp(model)

        checkpoint.load(model, optimizer, train_state, world_mesh)
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(
                    Path(args.dump_dir) / "probe", exist_ok=True,
                )
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir)
                    / "probe"
                    / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        gc.disable()

        # ---- train loop ----
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(
                args.data,
                state=train_state.data_loader_state,
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
            train_state.acc_step = (
                train_state.acc_step % args.grad_acc_steps
            )

            curr_lr = float(optimizer["lm"].param_groups[0]["lr"])
            curr_stem_lr = float(optimizer["stem"].param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)

            if every_n_steps(
                train_state, args.gc_collect_freq, acc_step=0
            ):
                logger.info("garbage collection")
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen = labels.shape

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            if (args.probe_freq is not None) and every_n_steps(
                train_state,
                args.probe_freq,
                acc_step=1 % args.grad_acc_steps,
            ):
                assert (
                    next(model.parameters()).grad is None
                ), "Can't probe model if grads are not reset"

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = (
                        seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    )
                    probe_loss = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                    )
                    probe_loss.backward()
                    optimizer["lm"].zero_grad()
                    optimizer["stem"].zero_grad()

                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            loss = model(input_ids, labels)

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(
                    train_state.acc_step == 0,
                )

            loss = loss / args.grad_acc_steps
            loss.backward()
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            stem_grad_norm = -1.0
            if train_state.acc_step == 0:
                # Clip lm_transformer gradients (DTensors from FSDP)
                lm_params = [
                    p
                    for p in model.lm_transformer.parameters()
                    if p.grad is not None
                ]
                if lm_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        lm_params,
                        max_norm=args.optim.clip,
                        foreach=True,
                    )
                    grad_norm = (
                        grad_norm.full_tensor()
                        if isinstance(grad_norm, DTensor)
                        else grad_norm
                    ).item()

                # Clip stem + selective-IIR gradients (regular Tensors)
                stem_params = [
                    p
                    for p in model.stem_parameters()
                    if p.grad is not None
                ]
                if stem_params:
                    stem_grad_norm = torch.nn.utils.clip_grad_norm_(
                        stem_params,
                        max_norm=args.optim.clip,
                        foreach=False,
                    ).item()

                # Sync non-FSDP gradients across STEM DP ranks
                if get_stem_data_parallel_world_size() > 1:
                    dp_group = get_stem_data_parallel_group()
                    for param in model.stem_parameters():
                        if param.grad is not None:
                            torch.distributed.all_reduce(
                                param.grad,
                                op=torch.distributed.ReduceOp.AVG,
                                group=dp_group,
                            )

                optimizer["lm"].step()
                optimizer["stem"].step()
                scheduler["lm"].step()
                scheduler["stem"].step()
                optimizer["lm"].zero_grad()
                optimizer["stem"].zero_grad()
                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(
                start_timer.elapsed_time(end_timer) * 1e-3, 4,
            )

            if torch_profiler:
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
                    args.grad_acc_steps * train_state.step
                    + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps
                    * args.data.batch_size
                    * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                FLOPS = (
                    _get_num_flop_per_token(
                        model_param_count
                        - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                optim_dict = {
                    "grad_norm": grad_norm,
                    "lr": curr_lr,
                    "stem_lr": curr_stem_lr,
                    "total_tokens": total_tokens,
                }
                if stem_grad_norm >= 0:
                    optim_dict["stem_grad_norm"] = stem_grad_norm

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
                        "optim": optim_dict,
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()

                # ---- Log selective-IIR gate statistics per layer ----
                sel_iir_dict = {}
                for layer_idx, sel_iir in enumerate(
                    model.selective_iir_memories
                ):
                    # Effective mu from bias (input-independent component)
                    mu_bias = sel_iir.mu_bias_value
                    mu_eff = torch.sigmoid(
                        torch.tensor(mu_bias)
                    ).item()
                    alpha_bias = sel_iir.alpha_bias_value
                    # Weight norms (how much the gates depend on input)
                    w_mu_norm = sel_iir.w_mu.weight.norm().item()
                    w_alpha_norm = sel_iir.w_alpha.weight.norm().item()

                    sel_iir_dict[
                        f"optim/sel_iir_mu_eff_{layer_idx}"
                    ] = mu_eff
                    sel_iir_dict[
                        f"optim/sel_iir_alpha_bias_{layer_idx}"
                    ] = alpha_bias
                    sel_iir_dict[
                        f"optim/sel_iir_w_mu_norm_{layer_idx}"
                    ] = w_mu_norm
                    sel_iir_dict[
                        f"optim/sel_iir_w_alpha_norm_{layer_idx}"
                    ] = w_alpha_norm

                metrics.update(dist_mean_dict(to_sync))
                metrics.update(sel_iir_dict)

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                log_msg = (
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  grad: {grad_norm:.2e}"
                )
                if stem_grad_norm >= 0:
                    log_msg += f"  stem_grad: {stem_grad_norm:.2e}"
                log_msg += (
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  stem_lr: {curr_stem_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )
                if sel_iir_dict:
                    mu_strs = [
                        f"L{k.split('_')[-1]}={v:.4f}"
                        for k, v in sel_iir_dict.items()
                        if "mu_eff" in k
                    ]
                    wnorm_strs = [
                        f"L{k.split('_')[-1]}={v:.4f}"
                        for k, v in sel_iir_dict.items()
                        if "w_mu_norm" in k
                    ]
                    log_msg += f"  mu_eff: [{', '.join(mu_strs)}]"
                    log_msg += f"  w_mu: [{', '.join(wnorm_strs)}]"
                logger.info(log_msg)

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0,
            ) or every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0,
            ):
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            if args.eval is not None and (
                every_n_steps(
                    train_state,
                    args.checkpoint.eval.every,
                    acc_step=0,
                )
                or every_n_steps(
                    train_state, args.steps, acc_step=0,
                )
            ):
                from apps.main.stem_eval import (
                    launch_stem_eval,
                    EVAL_FOLDER_NAME,
                    StemEvalArgs,
                )

                eval_args = dataclass_from_dict(StemEvalArgs, args.eval)

                eval_args.model_type = args.model_type
                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.stem_parallel_size = (
                    args.distributed.stem_parallel_size
                )
                eval_args.dump_dir = str(
                    os.path.join(
                        args.dump_dir,
                        "evals",
                        EVAL_FOLDER_NAME.format(train_state.step),
                    )
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    launch_stem_eval(eval_args)
                elif get_is_master():
                    if (
                        wandb.run is not None
                        and args.logging.wandb is not None
                    ):
                        eval_args.wandb = deepcopy(args.logging.wandb)
                    assert args.async_eval_gpus > 0
                    logger.info(
                        f"Launching evals on {args.async_eval_gpus} gpus"
                    )
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
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
    gc.collect()


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(SelectiveIIRStemTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()