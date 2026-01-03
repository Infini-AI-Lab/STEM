# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

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
from apps.main.stem import StemLMTransformerArgs, StemLMTransformer, build_stem_lm_fsdp_grouping_plan
from apps.main.transformer import (
    get_no_recompute_ops,
    get_num_flop_per_token,
)
from lingua.stem_dist_utils import initialize_stem_process_group

import wandb

logger = logging.getLogger()

@dataclass
class StemTrainArgs(TrainArgs):
    model: StemLMTransformerArgs = field(default_factory=StemLMTransformerArgs)
    stem_parallel_size: int = 8
    
    
preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True
    
    
def train(args: StemTrainArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(
            args,
            tokenizer.n_words,
        )
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")
        
        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        # Initialize stem process groups for ParallelEmbedding
        # This MUST be called before creating the model with ParallelEmbedding
        initialize_stem_process_group(args.stem_parallel_size)
        logger.info(f"Initialized stem process groups with parallel size: {args.stem_parallel_size}")

        torch.manual_seed(args.seed)
        logger.info("Building model")
        
        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = StemLMTransformer(args.model)
        logger.info("Model is built !")
        
        model_param_count = get_num_params(model)
        
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_stem_lm_fsdp_grouping_plan(args.model),
            tp_parallelize=None,
            no_recompute_ops=get_no_recompute_ops(),
        )
        
        model = model.to_empty(device="cuda")
        
        # Ensure stem_embeddings parameters require gradients and verify they're on the correct device
        for i, embedding in enumerate(model.stem_embeddings):
            # Verify device after to_empty()
            weight_device = embedding.weight.device
            if weight_device.type != "cuda":
                logger.warning(
                    f"stem_embeddings[{i}].weight is on {weight_device} after to_empty(), "
                    f"expected cuda. This may cause initialization issues."
                )
            for param in embedding.parameters():
                param.requires_grad = True
                # Debug: Check if parameter is initialized (not all zeros)
                if param.numel() > 0:
                    is_zero = (param.abs().max() == 0).item()
                    if is_zero:
                        logger.warning(
                            f"stem_embeddings[{i}] parameter {param.shape} is all zeros before init_weights()"
                        )
        logger.info("Ensured stem_embeddings parameters require gradients")

        # Initialize model weights if not loading from init checkpoint
        # (init checkpoint loading happens after optimizer creation to allow loading optimizer states)
        if not args.checkpoint.init_ckpt_path:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        
        # Verify stem_embeddings are initialized after init_weights()
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
                            f"ERROR: stem_embeddings[{i}].{param_name} is still all zeros after init_weights()!"
                        )
        
        check_model_value_range(model, range=10.0, std=1.0)
        
        logger.info(f"Model size: {model_param_count:,} total parameters")
        
        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")
        
        # build optimizer after apply parallelisms to the model
        # Create separate optimizers for lm_transformer and stem_embeddings
        # to avoid mixing DTensors and regular Tensors
        from torch.optim import AdamW
        from lingua.optim import build_lr_fn
        
        # Create optimizer for lm_transformer (DTensors)
        lm_optimizer = AdamW(
            model.lm_transformer.parameters(),
            lr=args.optim.lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
            eps=args.optim.epsilon,
            fused=True,
        )
        
        # Create optimizer for stem_embeddings (regular Tensors)
        stem_optimizer = AdamW(
            model.stem_embeddings.parameters(),
            lr=args.optim.lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
            eps=args.optim.epsilon,
            fused=False,  # Disable fused for regular tensors
        )
        
        # Create schedulers for both optimizers
        lr_fn = build_lr_fn(args.optim, args.steps)
        from torch.optim import lr_scheduler
        lm_scheduler = lr_scheduler.LambdaLR(lm_optimizer, lr_fn)
        stem_scheduler = lr_scheduler.LambdaLR(stem_optimizer, lr_fn)
        
        # Store both optimizers and schedulers
        optimizer = {"lm": lm_optimizer, "stem": stem_optimizer}
        scheduler = {"lm": lm_scheduler, "stem": stem_scheduler}
        
        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )
        
        # Use StemCheckpointManager which handles ParallelEmbedding checkpointing
        checkpoint = StemCheckpointManager.instantiate_and_make_dir(args.checkpoint)
        
        # Load from init checkpoint if specified (before loading from latest checkpoint)
        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            from lingua.stem_checkpoint import load_from_checkpoint
            load_from_checkpoint(
                args.checkpoint.init_ckpt_path, 
                model, 
                optimizer=optimizer,
                model_key="model"
            )
            model.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
            # Ensure stem_embeddings are initialized even when loading from checkpoint
            # (in case they're not in the checkpoint)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.reset_stem_embeddings()
        
        # Load from latest checkpoint (or continue from init checkpoint)
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
        
        # train loop
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
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get batch
            curr_lr = float(optimizer["lm"].param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(
                batch,
                dtype=torch.long,
            )

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
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

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                # Here we do a fake forward and backward pass on a smaller
                # batch size to avoid OOM
                # This assumes the model has no stateful layers (batch norm..)
                assert (
                    next(model.parameters()).grad is None
                ), "Can't probe model if grads are not reset"

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    # Non compiled model uses roughly 2x memory in our exps
                    # So we divide bsz by 2 or seqlen by 2
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                    )
                    probe_loss.backward()
                    # We zero grads to cancel this fake step
                    optimizer["lm"].zero_grad()
                    optimizer["stem"].zero_grad()

                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            loss = model(input_ids, labels)

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            stem_grad_norm = -1.0
            if train_state.acc_step == 0:
                # Clip gradients separately for lm_transformer and stem_embeddings
                # since they have different tensor types (DTensor vs regular Tensor)
                # Clip gradients from lm_transformer (DTensors from FSDP)
                lm_params = [p for p in model.lm_transformer.parameters() if p.grad is not None]
                if lm_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        lm_params, max_norm=args.optim.clip, foreach=True
                    )
                    grad_norm = (
                        grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                    ).item()
                
                # Clip gradients from stem_embeddings (regular Tensors, manually managed)
                stem_params = [p for p in model.stem_embeddings.parameters() if p.grad is not None]
                if stem_params:
                    stem_grad_norm = torch.nn.utils.clip_grad_norm_(
                        stem_params, max_norm=args.optim.clip, foreach=False
                    ).item()
                else:
                    # Debug: check if stem_embeddings have gradients at all
                    all_stem_params = list(model.stem_embeddings.parameters())
                    params_with_grad = [p for p in all_stem_params if p.grad is not None]
                    params_without_grad = [p for p in all_stem_params if p.grad is None]
                    if params_without_grad:
                        logger.warning(
                            f"Warning: {len(params_without_grad)}/{len(all_stem_params)} stem_embeddings parameters "
                            f"have no gradients. This may indicate a gradient flow issue."
                        )
                    # Check if any gradients are zero
                    if params_with_grad:
                        zero_grads = [p for p in params_with_grad if p.grad is not None and p.grad.abs().max() == 0]
                        if zero_grads:
                            logger.warning(
                                f"Warning: {len(zero_grads)}/{len(params_with_grad)} stem_embeddings parameters "
                                f"have zero gradients."
                            )

                optimizer["lm"].step()
                optimizer["stem"].step()
                scheduler["lm"].step()
                scheduler["stem"].step()
                optimizer["lm"].zero_grad()
                optimizer["stem"].zero_grad()
                train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                xformers.profiler.step()

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
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
                    "lr": curr_lr,
                    "total_tokens": total_tokens,
                }
                # Add stem gradient norm if available
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
                metrics.update(dist_mean_dict(to_sync))

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
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )
                logger.info(log_msg)

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                # Pass full optimizer dict - checkpoint manager will handle both optimizers
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            if args.eval is not None and (every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ) or every_n_steps(train_state, args.steps, acc_step=0)):
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
                    logger.info(f"Launching evals on {args.async_eval_gpus} gpus")
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
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(StemTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()