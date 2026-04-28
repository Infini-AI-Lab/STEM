# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
#
# Full Longcat n-gram LM training (FSDP backbone + vocabulary-parallel n-gram
# tables, dual optimizers, :class:`lingua.longcat_checkpoint.LongcatCheckpointManager`,
# init loads via :mod:`lingua.longcat_checkpoint`).
# Run: ``python -m apps.main.longcat_ngram_train config=apps/main/configs/longcat_ngram_train.yaml``

from copy import deepcopy
import gc
import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple, Union


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
from lingua.longcat_checkpoint import (
    LongcatCheckpointManager,
    NGRAM_MODEL_SHARD_GLOB,
    NGRAM_SHARD_SUBDIR,
    load_longcat_init_from_checkpoint,
    merge_longcat_backbone_dcp_seed_then_warmup,
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
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.logger import init_logger
from lingua.tokenizer import build_tokenizer
from lingua.profiling import maybe_run_profiler
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

from apps.main.train import TrainArgs, TrainState, validate_train_args, every_n_steps
from apps.main.longcat import (
    LongcatLMTransformerArgs,
    LongcatLMTransformer,
    LongcatOLMo3LMTransformer,
    LongcatOLMo3LMTransformerArgs,
    LongcatQwen3LMTransformer,
    LongcatQwen3LMTransformerArgs,
    build_longcat_lm_fsdp_grouping_plan,
    build_longcat_olmo3_lm_fsdp_grouping_plan,
    build_longcat_qwen3_lm_fsdp_grouping_plan,
)
from apps.main.olmo3 import (
    get_no_recompute_ops as olmo3_get_no_recompute_ops,
    get_num_flop_per_token as olmo3_get_num_flop_per_token,
)
from apps.main.qwen3 import (
    get_no_recompute_ops as qwen3_get_no_recompute_ops,
    get_num_flop_per_token as qwen3_get_num_flop_per_token,
)
from apps.main.transformer import (
    get_no_recompute_ops as llama_get_no_recompute_ops,
    get_num_flop_per_token as llama_get_num_flop_per_token,
)
from lingua.stem_dist_utils import (
    initialize_stem_process_group,
    get_stem_data_parallel_group,
    get_stem_data_parallel_world_size,
)


def sync_ngram_embeddings_across_dp(
    model: Union[
        LongcatLMTransformer,
        LongcatQwen3LMTransformer,
        LongcatOLMo3LMTransformer,
    ],
):
    """Broadcast n-gram embedding weights from DP rank 0 (same rationale as STEM)."""
    if get_stem_data_parallel_world_size() <= 1:
        return

    dp_group = get_stem_data_parallel_group()
    src_rank = torch.distributed.get_global_rank(dp_group, 0)
    for param in model.ngram_embeddings.parameters():
        torch.distributed.broadcast(param.data, src=src_rank, group=dp_group)
    logger.info(
        "Synchronized ngram_embeddings weights across vocabulary-parallel DP ranks"
    )


LONGCAT_MODEL_REGISTRY = {
    "llama": (
        LongcatLMTransformer,
        LongcatLMTransformerArgs,
        build_longcat_lm_fsdp_grouping_plan,
        llama_get_no_recompute_ops,
        llama_get_num_flop_per_token,
    ),
    "qwen3": (
        LongcatQwen3LMTransformer,
        LongcatQwen3LMTransformerArgs,
        build_longcat_qwen3_lm_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops,
        qwen3_get_num_flop_per_token,
    ),
    "olmo3": (
        LongcatOLMo3LMTransformer,
        LongcatOLMo3LMTransformerArgs,
        build_longcat_olmo3_lm_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops,
        olmo3_get_num_flop_per_token,
    ),
}

import wandb

logger = logging.getLogger()


@dataclass
class LongcatTrainLossOut:
    """Bundle for training: backward through ``loss``.

    For distillation, ``loss`` is built so ``loss.item()`` is CE (for ``loss/out``
    logging) while autograd follows the weighted CE+KL objective. Optional
    ``distill_loss`` is the raw KL term for ``loss/distill`` metrics.
    """

    loss: torch.Tensor
    distill_loss: Optional[torch.Tensor] = None
    

def unpack_longcat_train_loss_out(
    out: Union[torch.Tensor, LongcatTrainLossOut],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(out, LongcatTrainLossOut):
        return out.loss, out.distill_loss
    return out, None


def _tensor_to_log_scalar(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t.item()


@dataclass
class LongcatTrainArgs(TrainArgs):
    model_type: str = "llama"
    # Union[Longcat*Args, ...] breaks OmegaConf.structured / merge; type follows model_type.
    model: Any = field(default_factory=LongcatLMTransformerArgs)

    # Separate learning rate / schedule for ``ngram_embeddings`` (outside FSDP).
    ngram_lr: Optional[float] = None
    ngram_weight_decay: Optional[float] = None
    ngram_warmup: Optional[int] = None
    ngram_scheduler: Optional[str] = None
    ngram_lr_min_ratio: Optional[float] = None
    freeze_base: bool = False
    
    
preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True
    
    
def train(args: LongcatTrainArgs):
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
        initialize_stem_process_group(args.distributed.stem_parallel_size)
        logger.info(f"Initialized stem process groups with parallel size: {args.distributed.stem_parallel_size}")

        # ---- Resolve model class & helpers from the registry ----
        if args.model_type not in LONGCAT_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{args.model_type}'. "
                f"Available: {list(LONGCAT_MODEL_REGISTRY.keys())}"
            )
        (
            longcat_model_cls,
            _longcat_args_cls,
            _build_fsdp_plan,
            _get_no_recompute_ops,
            _get_num_flop_per_token,
        ) = LONGCAT_MODEL_REGISTRY[args.model_type]
        logger.info(
            f"Using Longcat model type: {args.model_type} ({longcat_model_cls.__name__})"
        )

        torch.manual_seed(args.seed)
        logger.info("Building model")
        
        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = longcat_model_cls(args.model)
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

        freeze_base = args.freeze_base
        if freeze_base:
            for param in model.lm_transformer.parameters():
                param.requires_grad = False
            logger.info("Base model parameters are frozen (freeze_base=True)")
        else:
            logger.info("Base model parameters are trainable (freeze_base=False)")
        
        # Per-shard VocabParallelEmbedding modules live under ``ngram_embeddings.embedders``.
        for i, embedding in enumerate(model.ngram_embeddings.embedders):
            # Verify device after to_empty()
            weight_device = embedding.weight.device
            if weight_device.type != "cuda":
                logger.warning(
                    f"ngram_embeddings.embedders[{i}].weight is on {weight_device} after to_empty(), "
                    f"expected cuda. This may cause initialization issues."
                )
            for param in embedding.parameters():
                param.requires_grad = True
        logger.info("Ensured ngram_embeddings parameters require gradients")

        # Initialize model weights if not loading from init checkpoint
        # (init checkpoint loading happens after optimizer creation to allow loading optimizer states)
        if not args.checkpoint.init_ckpt_path:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
            # Ensure ngram_embeddings are identical across DP ranks
            # (RNG state may diverge due to FSDP-sharded lm_transformer init)
            sync_ngram_embeddings_across_dp(model)
            
        
        # Verify ngram shard embeddings are initialized after init_weights()
        for i, embedding in enumerate(model.ngram_embeddings.embedders):
            for param_name, param in embedding.named_parameters():
                if param.numel() > 0:
                    is_zero = (param.abs().max() == 0).item()
                    param_norm = param.norm().item()
                    logger.info(
                        f"ngram_embeddings.embedders[{i}].{param_name}: "
                        f"device={param.device}, shape={param.shape}, "
                        f"norm={param_norm:.6f}, is_zero={is_zero}"
                    )
                    if is_zero and not getattr(
                        args.model, "ngram_embeddings_zero_reset", False
                    ):
                        logger.error(
                            f"ERROR: ngram_embeddings.embedders[{i}].{param_name} is still all zeros after init_weights()!"
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
        # Create separate optimizers for lm_transformer and ngram_embeddings
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
        
        # Create optimizer for ngram_embeddings (regular Tensors)
        # Use dedicated stem lr / weight_decay when provided, else fall back to main optim values
        ngram_lr = args.ngram_lr if args.ngram_lr is not None else args.optim.lr
        ngram_wd = (
            args.ngram_weight_decay
            if args.ngram_weight_decay is not None
            else args.optim.weight_decay
        )
        logger.info(f"Ngram optimizer: lr={ngram_lr}, weight_decay={ngram_wd}")
        ngram_optimizer = AdamW(
            model.ngram_embeddings.parameters(),
            lr=ngram_lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=ngram_wd,
            eps=args.optim.epsilon,
            fused=False,  # Disable fused for regular tensors
        )
        
        # Create schedulers for both optimizers.
        lm_lr_fn = build_lr_fn(args.optim, args.steps)
        ngram_warmup = args.ngram_warmup if args.ngram_warmup is not None else args.optim.warmup
        ngram_scheduler_name = (
            args.ngram_scheduler
            if args.ngram_scheduler is not None
            else args.optim.scheduler
        )
        ngram_lr_min_ratio = (
            args.ngram_lr_min_ratio
            if args.ngram_lr_min_ratio is not None
            else args.optim.lr_min_ratio
        )
        ngram_optim_args = replace(
            args.optim,
            warmup=ngram_warmup,
            scheduler=ngram_scheduler_name,
            lr_min_ratio=ngram_lr_min_ratio,
            initial_token_offset=0,
            global_final_step=args.steps,
        )
        logger.info(
            f"Ngram scheduler: scheduler={ngram_scheduler_name}, warmup={ngram_warmup}, "
            f"lr_min_ratio={ngram_lr_min_ratio}"
        )
        ngram_lr_fn = build_lr_fn(ngram_optim_args, args.steps)
        from torch.optim import lr_scheduler
        lm_scheduler = lr_scheduler.LambdaLR(lm_optimizer, lm_lr_fn)
        ngram_scheduler = lr_scheduler.LambdaLR(ngram_optimizer, ngram_lr_fn)

        # Store both optimizers and schedulers
        optimizer = {"lm": lm_optimizer, "ngram": ngram_optimizer}
        scheduler = {"lm": lm_scheduler, "ngram": ngram_scheduler}
        
        
        data_rank = dp_rank
        data_world_size = dp_degree
        if args.data.node_local:
            local_rank_env = os.environ.get("LOCAL_RANK")
            local_world_env = os.environ.get("LOCAL_WORLD_SIZE")
            assert (
                local_rank_env is not None and local_world_env is not None
            ), "data.node_local=true requires LOCAL_RANK and LOCAL_WORLD_SIZE to be set"
            data_rank = int(local_rank_env)
            data_world_size = int(local_world_env)
            assert data_world_size > 0, "LOCAL_WORLD_SIZE must be > 0"
            assert (
                0 <= data_rank < data_world_size
            ), f"LOCAL_RANK ({data_rank}) must be in [0, {data_world_size})"
            logger.info(
                "Using node-local dataloader sharding: "
                f"rank {data_rank}/{data_world_size} "
                f"(global dp rank {dp_rank}/{dp_degree})"
            )
        else:
            logger.info(
                f"Using global DP dataloader sharding: rank {data_rank}/{data_world_size}"
            )

        data_loader_state = init_dataloader_state_from_args(
            args.data, data_rank, data_world_size
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )
        
        checkpoint = LongcatCheckpointManager.instantiate_and_make_dir(args.checkpoint)
        
        # Load from init checkpoint if specified (before loading from latest checkpoint)
        if args.checkpoint.init_ckpt_path:
            seed_merge = getattr(
                args.checkpoint, "merge_lm_optim_seed_ckpt_path", None
            )
            if seed_merge:
                if not args.checkpoint.continue_training_from_init:
                    raise ValueError(
                        "checkpoint.merge_lm_optim_seed_ckpt_path requires "
                        "checkpoint.continue_training_from_init=true "
                        "(LM + stem optimizers must exist for merge)."
                    )
                logger.info(
                    "Merging init checkpoints: legacy seed=%s then warmup=%s",
                    seed_merge,
                    args.checkpoint.init_ckpt_path,
                )
                merge_longcat_backbone_dcp_seed_then_warmup(
                    model,
                    optimizer,
                    seed_merge,
                    args.checkpoint.init_ckpt_path,
                )
            else:
                logger.info(
                    f"Loading initial model from {args.checkpoint.init_ckpt_path}"
                )
                if args.checkpoint.continue_training_from_init:
                    load_longcat_init_from_checkpoint(
                        args.checkpoint.init_ckpt_path,
                        model,
                        optimizer=optimizer,
                        model_key="model",
                        legacy_lm_transformer=args.checkpoint.legacy_init_ckpt_lm_transformer,
                    )
                else:
                    load_longcat_init_from_checkpoint(
                        args.checkpoint.init_ckpt_path,
                        model,
                        model_key="model",
                        legacy_lm_transformer=args.checkpoint.legacy_init_ckpt_lm_transformer,
                    )
                if args.checkpoint.legacy_init_ckpt_lm_transformer:
                    model.init_ngram_fused_projection()
            model.rope_embeddings.reset_parameters()
            ngram_shards_dir = Path(args.checkpoint.init_ckpt_path) / NGRAM_SHARD_SUBDIR
            if ngram_shards_dir.exists() and any(ngram_shards_dir.glob(NGRAM_MODEL_SHARD_GLOB)):
                logger.info(
                    "Pre-computed vocabulary-parallel shards in init checkpoint; "
                    "skipping random n-gram table reset"
                )
            else:
                if getattr(args.model, "ngram_embeddings_zero_reset", False):
                    logger.info(
                        "No n-gram shards in init checkpoint; zero-initializing "
                        "(model.ngram_embeddings_zero_reset=True)"
                    )
                    model.reset_ngram_embeddings()
                else:
                    logger.info(
                        "No n-gram shards in init checkpoint; re-initializing tables "
                        "(VocabParallelEmbedding defaults)"
                    )
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(args.model.seed)
                        model.reset_ngram_embeddings()
            sync_ngram_embeddings_across_dp(model)
            
            
        # Load from latest checkpoint (or continue from init checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        stage_start_step = train_state.step
        if args.stage_steps is None:
            target_step = args.steps
        else:
            target_step = min(args.steps, stage_start_step + args.stage_steps)
            logger.info(
                "Stage-limited training enabled: "
                f"start_step={stage_start_step}, stage_steps={args.stage_steps}, "
                f"target_step={target_step}, global_steps={args.steps}"
            )
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
        logger.info(f"Starting training from step {train_state.step}")
        saved = False
        while train_state.step < target_step:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get batch
            curr_lr = float(optimizer["lm"].param_groups[0]["lr"])
            curr_ngram_lr = float(optimizer["ngram"].param_groups[0]["lr"])
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
                    probe_raw = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                    )
                    probe_loss, _ = unpack_longcat_train_loss_out(probe_raw)
                    probe_loss.backward()
                    # We zero grads to cancel this fake step
                    optimizer["lm"].zero_grad()
                    optimizer["ngram"].zero_grad()

                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            teacher_logits = None
            compute_teacher_logits = getattr(model, "compute_teacher_logits", None)
            if callable(compute_teacher_logits):
                teacher_logits = compute_teacher_logits(
                    token_values=input_ids,
                    tok_idx=None,
                    mask=None,
                    attn_impl="sdpa",
                )

            if teacher_logits is None:
                raw_loss_out = model(input_ids, labels)
            else:
                raw_loss_out = model(input_ids, labels, teacher_logits=teacher_logits)

            loss, log_distill_t = unpack_longcat_train_loss_out(raw_loss_out)

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
            ngram_grad_norm = -1.0
            if train_state.acc_step == 0:
                # Clip gradients separately for lm_transformer and ngram_embeddings
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
                
                # Sync ngram_embeddings gradients across STEM data-parallel ranks.
                # FSDP handles gradient sync for lm_transformer, but ngram_embeddings
                # are managed manually and need an explicit all-reduce when there are
                # multiple data-parallel groups (e.g. multi-node with intra-node STEM MP).
                if get_stem_data_parallel_world_size() > 1:
                    dp_group = get_stem_data_parallel_group()
                    for param in model.ngram_embeddings.parameters():
                        if param.grad is not None:
                            torch.distributed.all_reduce(
                                param.grad,
                                op=torch.distributed.ReduceOp.AVG,
                                group=dp_group,
                            )

                # Clip gradients from ngram_embeddings (regular Tensors, manually managed)
                ngram_params = [
                    p for p in model.ngram_embeddings.parameters() if p.grad is not None
                ]
                if ngram_params:
                    ngram_grad_norm = torch.nn.utils.clip_grad_norm_(
                        ngram_params, max_norm=args.optim.clip, foreach=False
                    ).item()
                else:
                    all_ngram_params = list(model.ngram_embeddings.parameters())
                    params_with_grad = [p for p in all_ngram_params if p.grad is not None]
                    params_without_grad = [
                        p for p in all_ngram_params if p.grad is None
                    ]
                    if params_without_grad:
                        logger.warning(
                            f"Warning: {len(params_without_grad)}/{len(all_ngram_params)} "
                            "ngram_embeddings parameters have no gradients."
                        )
                    if params_with_grad:
                        zero_grads = [
                            p
                            for p in params_with_grad
                            if p.grad is not None and p.grad.abs().max() == 0
                        ]
                        if zero_grads:
                            logger.warning(
                                f"Warning: {len(zero_grads)}/{len(params_with_grad)} "
                                "ngram_embeddings parameters have zero gradients."
                            )

                optimizer["lm"].step()
                optimizer["ngram"].step()
                scheduler["lm"].step()
                scheduler["ngram"].step()
                optimizer["lm"].zero_grad()
                optimizer["ngram"].zero_grad()
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
                    _get_num_flop_per_token(
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
                    "ngram_lr": curr_ngram_lr,
                    "total_tokens": total_tokens,
                }
                # Add n-gram embedding gradient norm if available
                if ngram_grad_norm >= 0:
                    optim_dict["ngram_grad_norm"] = ngram_grad_norm
                
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
                distill_aux = _tensor_to_log_scalar(log_distill_t)
                if distill_aux is not None:
                    to_sync["loss/distill"] = distill_aux

                alpha_dict = {}
                for layer_idx, layer in enumerate(model.lm_transformer.layers):
                    ff = layer.feed_forward
                    if hasattr(ff, "alpha"):
                        alpha_val = ff.alpha
                        if isinstance(alpha_val, DTensor):
                            alpha_val = alpha_val.full_tensor()
                        sig = torch.sigmoid(alpha_val).item()
                        alpha_dict[f"optim/gate_{layer_idx}"] = sig

                metrics.update(dist_mean_dict(to_sync))
                metrics.update(alpha_dict)

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
                if distill_aux is not None:
                    log_msg += f"  distill: {round(distill_aux, 4):>7}"
                if ngram_grad_norm >= 0:
                    log_msg += f"  ngram_grad: {ngram_grad_norm:.2e}"
                log_msg += (
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  ngram_lr: {curr_ngram_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )
                if alpha_dict:
                    alpha_strs = [f"L{k.split('_')[1]}={v:.4f}" for k, v in alpha_dict.items()]
                    log_msg += f"  gates: [{', '.join(alpha_strs)}]"
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
            ) or every_n_steps(train_state, target_step, acc_step=0)):
                from apps.main.eval import EVAL_FOLDER_NAME, EvalArgs
                from apps.main.longcat_ngram_eval import launch_longcat_eval

                eval_args = dataclass_from_dict(EvalArgs, args.eval)

                eval_args.model_type = args.model_type
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
                    launch_longcat_eval(eval_args)
                elif get_is_master():
                    if wandb.run is not None and args.logging.wandb is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                    assert args.async_eval_gpus > 0
                    logger.info(f"Launching evals on {args.async_eval_gpus} gpus")
                    with clean_env():
                        launch_job(
                            StoolArgs(
                                asdict(eval_args),
                                script="apps.main.longcat_eval",
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

    default_cfg = OmegaConf.structured(LongcatTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
        
        