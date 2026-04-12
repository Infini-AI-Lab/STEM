# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Interpolated STEM finetuning.

This script finetunes a pretrained model with STEM embeddings using a smooth
interpolation schedule.  Instead of immediately replacing the up-projection
(w3) with the STEM embedding, it interpolates:

    up = alpha * w3(x) + (1 - alpha) * stem_embedding
    output = w2(SiLU(w1(x)) * up)

Alpha decays exponentially from 1 (full baseline up-projection) to 0 (full
STEM embedding) during training, controlled by ``interp_half_life`` and
``interp_steps``.  This ensures:

1. Low initial loss (model starts in its pretrained state).
2. Gradual transition to STEM-only (final architecture).
3. The up-projections (w3) remain frozen throughout since they are dispensed
   with at the end.

Usage:
    torchrun --nproc_per_node=8 -m apps.main.stem_interpolated_finetune \\
        config=apps/main/configs/stem_interpolated_qwen3_1.7b.yaml
"""

from copy import deepcopy
import gc
import json
import logging
import os
import re
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
import xformers.profiler
from torch.optim import lr_scheduler
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
from lingua.optim import build_lr_fn
from lingua.logger import init_logger
from lingua.tokenizer import build_tokenizer
from lingua.profiling import maybe_run_profiler
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

from apps.main.train import TrainArgs, TrainState, validate_train_args, every_n_steps
from apps.main.stem import StemLMTransformerArgs
from apps.main.stem_interpolated import (
    InterpolatedStemLMTransformer,
    INTERP_STEM_MODEL_REGISTRY,
    compute_interp_alpha,
)
from lingua.stem_dist_utils import (
    initialize_stem_process_group,
    get_stem_data_parallel_group,
    get_stem_data_parallel_rank,
    get_stem_data_parallel_world_size,
)


def consolidate_interpolated_checkpoints(
    ckpt_dir: str,
    stem_layers: Optional[List[int]] = None,
) -> Path:
    """
    Consolidate an interpolated STEM DCP checkpoint, stripping w3 weights
    from stem layers so that it matches the StemLMTransformer format expected
    by the eval pipeline.

    The DCP backbone checkpoint saved by StemCheckpointManager wraps everything
    under a ``model`` key and includes ``lm_transformer.*`` prefixed FQNs.
    This function:

    1. Reads DCP metadata.
    2. Allocates tensors for model keys only (skips optimizer state and w3
       weights in stem layers).
    3. Loads via ``dcp.load`` (no distributed setup required).
    4. Renames keys: strips the ``model.`` DCP wrapper so the consolidated
       dict uses ``lm_transformer.layers.…`` directly — matching what
       ``load_consolidated_model_and_tokenizer`` expects.
    5. Saves as ``<ckpt_dir>/consolidated/consolidated.pth`` together with
       a copy of ``params.json``.

    Args:
        ckpt_dir: Path to the DCP checkpoint directory
                  (e.g. ``logs/stem_interpolated/checkpoints/0000000200``).
        stem_layers: Layer indices that are stem layers.  If *None*, the
                     list is read from ``params.json`` in the checkpoint
                     directory.

    Returns:
        Path to the ``consolidated/`` directory.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    CONSOLIDATE_FOLDER = "consolidated"
    CONSOLIDATE_NAME = "consolidated.pth"
    CONFIG_NAME = "params.json"

    ckpt_path = Path(ckpt_dir)
    consolidate_path = ckpt_path / CONSOLIDATE_FOLDER

    # Already consolidated → nothing to do.
    if (consolidate_path / CONSOLIDATE_NAME).exists():
        logger.info(
            f"Consolidated checkpoint already exists at {consolidate_path}"
        )
        return consolidate_path

    consolidate_path.mkdir(exist_ok=True)

    # ---- Resolve stem_layers from params.json when not provided -----------
    if stem_layers is None:
        config_path = ckpt_path / CONFIG_NAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"params.json not found at {config_path} and "
                f"stem_layers not provided"
            )
        with open(config_path, "r") as f:
            config = json.load(f)
        stem_layers = config.get("model", {}).get("stem_layers", [])
        if not stem_layers:
            logger.warning(
                "No stem_layers found in params.json; "
                "proceeding without w3 filtering."
            )

    logger.info(f"Consolidating interpolated checkpoint from {ckpt_dir}")
    logger.info(f"Stem layers (w3 will be stripped): {stem_layers}")

    stem_layers_set = set(stem_layers)

    # Pattern to match w3 FQNs inside stem layers.
    # DCP FQNs look like:
    #   model.lm_transformer.layers.<idx>.feed_forward.w3.weight
    w3_fqn_pattern = re.compile(
        r"(?:model\.)?(?:lm_transformer\.)?layers\.(\d+)"
        r"\.feed_forward\.w3\."
    )

    # ---- Step 1: Use dcp_to_torch_save to consolidate the full DCP -------
    # This uses _EmptyStateDictLoadPlanner internally with an empty dict,
    # which is the correct API contract.
    raw_save_path = consolidate_path / "raw_consolidated.pth"
    logger.info("Consolidating DCP checkpoint via dcp_to_torch_save ...")
    dcp_to_torch_save(str(ckpt_dir), str(raw_save_path))

    # ---- Step 2: Load, filter w3 keys & optimizer, rename ----------------
    # dcp_to_torch_save reconstructs the original nested dict structure via
    # planner_data, so the loaded dict looks like:
    #   {"model": {"lm_transformer.layers.0.…": tensor, …}, "optim": {…}}
    sd = torch.load(str(raw_save_path), map_location="cpu", weights_only=True)

    logger.info(f"Raw consolidated top-level keys: {list(sd.keys())}")

    # Drop optimizer state entirely — not needed for eval.
    if "optim" in sd:
        logger.info("Removing optimizer state from consolidated checkpoint")
        del sd["optim"]

    # Extract the model sub-dict.  After DCP reconstruction the model
    # weights live under sd["model"] as a flat dict with keys like
    # ``lm_transformer.layers.0.attention.wq.weight``.
    model_sd = sd.get("model", sd)

    # Filter out w3 weights from stem layers.
    skipped_keys: list = []
    keys_to_remove: list = []
    for fqn in list(model_sd.keys()):
        match = w3_fqn_pattern.search(fqn)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx in stem_layers_set:
                skipped_keys.append(fqn)
                keys_to_remove.append(fqn)

    for k in keys_to_remove:
        del model_sd[k]

    logger.info(
        f"Kept {len(model_sd)} model tensors, "
        f"removed {len(skipped_keys)} w3 keys from stem layers"
    )
    if skipped_keys:
        logger.info(f"Removed w3 keys: {skipped_keys}")

    # The eval pipeline (load_consolidated_model_and_tokenizer) expects
    # keys like ``lm_transformer.layers.0.…`` at the top level of the
    # saved dict.  model_sd already has this format.
    renamed_sd = model_sd
    del sd

    # ---- Step 3: Save final consolidated checkpoint ----------------------
    save_path = consolidate_path / CONSOLIDATE_NAME
    logger.info(f"Saving consolidated checkpoint ({len(renamed_sd)} keys) to {save_path}")
    torch.save(renamed_sd, str(save_path))

    # Remove the intermediate raw file.
    raw_save_path.unlink(missing_ok=True)

    # Copy params.json alongside the consolidated checkpoint.
    config_src = ckpt_path / CONFIG_NAME
    if config_src.exists():
        (consolidate_path / CONFIG_NAME).write_text(config_src.read_text())
        logger.info("Copied params.json to consolidated directory")

    logger.info("Interpolated checkpoint consolidation complete!")
    return consolidate_path


def sync_stem_embeddings_across_dp(model):
    """Broadcast stem_embeddings weights from dp_rank 0 to all STEM data-parallel ranks.

    This ensures all DP ranks start with identical stem_embeddings weights,
    which is required because:
      - FSDP-sharded lm_transformer init may consume different amounts of RNG
        on different ranks, causing the RNG state to diverge by the time
        stem_embeddings are initialized.
      - reset_stem_embeddings() after checkpoint loading also uses RNG
        (unless model.stem_embeddings_zero_reset is True).

    Must be called after any stem_embeddings initialization or reset.
    """
    if get_stem_data_parallel_world_size() <= 1:
        return  # Single DP group, nothing to sync

    dp_group = get_stem_data_parallel_group()
    src_rank = torch.distributed.get_global_rank(dp_group, 0)
    for param in model.stem_embeddings.parameters():
        torch.distributed.broadcast(param.data, src=src_rank, group=dp_group)
    logger.info("Synchronized stem_embeddings weights across STEM data-parallel ranks")


import wandb

logger = logging.getLogger()


@dataclass
class InterpolatedFinetuneArgs(TrainArgs):
    model: StemLMTransformerArgs = field(default_factory=StemLMTransformerArgs)

    # Separate learning rate and weight decay for stem_embeddings.
    # When None, falls back to the values in ``optim``.
    stem_lr: Optional[float] = None
    stem_weight_decay: Optional[float] = None

    # ---- Interpolation schedule ----
    # Half-life for exponential decay of alpha: alpha(t) = exp(-t * ln2 / half_life)
    # After half_life steps, alpha ≈ 0.5.  After ~7*half_life, alpha < 0.01.
    interp_half_life: int = 1000

    # Total interpolation phase (steps).  After this many steps, alpha is
    # clamped to 0 and the model uses pure STEM embeddings for the rest of
    # training.  Set to None to let alpha decay throughout the entire run.
    interp_steps: Optional[int] = None


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def train(args: InterpolatedFinetuneArgs):
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
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
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
        if args.model_type not in INTERP_STEM_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{args.model_type}'. "
                f"Available: {list(INTERP_STEM_MODEL_REGISTRY.keys())}"
            )
        (
            stem_model_cls, _stem_args_cls,
            _build_fsdp_plan,
            _get_no_recompute_ops, _get_num_flop_per_token,
        ) = INTERP_STEM_MODEL_REGISTRY[args.model_type]
        logger.info(
            f"Using Interpolated STEM model type: {args.model_type} "
            f"({stem_model_cls.__name__})"
        )

        # ---- Disable torch.compile ----
        # Alpha is a dynamic float that changes every step; torch.compile would
        # re-specialize the graph each time, causing severe slowdowns.
        saved_compile = args.distributed.compile
        if args.distributed.compile:
            logger.info(
                "Disabling torch.compile for interpolated STEM finetuning "
                "(dynamic alpha is incompatible with graph caching)"
            )
            args.distributed.compile = False

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models
        # much bigger than 1 gpu's memory
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

        # Restore compile flag (for config serialization)
        args.distributed.compile = saved_compile

        model = model.to_empty(device="cuda")

        # Ensure stem_embeddings parameters require gradients
        for i, embedding in enumerate(model.stem_embeddings):
            weight_device = embedding.weight.device
            if weight_device.type != "cuda":
                logger.warning(
                    f"stem_embeddings[{i}].weight is on {weight_device} "
                    f"after to_empty(), expected cuda."
                )
            for param in embedding.parameters():
                param.requires_grad = True
        logger.info("Ensured stem_embeddings parameters require gradients")

        # Initialize model weights if not loading from init checkpoint
        if not args.checkpoint.init_ckpt_path:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
            sync_stem_embeddings_across_dp(model)

        # ---- Freeze up-projections (w3) in all stem layers ----
        # This must happen AFTER loading weights but BEFORE creating the optimizer.
        # We freeze here proactively; if loading from checkpoint, we freeze again
        # below after load.
        frozen_count = model.freeze_up_projections()
        logger.info(
            f"Frozen {frozen_count} up-projection (w3) parameter tensors "
            f"across {len(model.stem_layers)} stem layers"
        )

        # Verify stem_embeddings initialization
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

        check_model_value_range(model, range=10.0, std=1.0)

        # Count trainable vs frozen parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params
        logger.info(
            f"Model size: {model_param_count:,} total parameters"
        )
        logger.info(
            f"Trainable: {trainable_params:,}  |  "
            f"Frozen (w3): {frozen_params:,}"
        )

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} "
            f"({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # ---- Build optimizer ----
        # Create separate optimizers for lm_transformer and stem_embeddings
        # to avoid mixing DTensors and regular Tensors.
        # Only include trainable parameters (w3 is frozen).
        from torch.optim import AdamW

        lm_trainable_params = [
            p for p in model.lm_transformer.parameters() if p.requires_grad
        ]
        lm_optimizer = AdamW(
            lm_trainable_params,
            lr=args.optim.lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
            eps=args.optim.epsilon,
            fused=True,
        )

        # Stem embeddings optimizer
        stem_lr = args.stem_lr if args.stem_lr is not None else args.optim.lr
        stem_wd = (
            args.stem_weight_decay
            if args.stem_weight_decay is not None
            else args.optim.weight_decay
        )
        logger.info(f"Stem optimizer: lr={stem_lr}, weight_decay={stem_wd}")
        stem_optimizer = AdamW(
            model.stem_embeddings.parameters(),
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
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        # Use StemCheckpointManager for ParallelEmbedding checkpointing
        checkpoint = StemCheckpointManager.instantiate_and_make_dir(args.checkpoint)

        # Load from init checkpoint if specified
        if args.checkpoint.init_ckpt_path:
            logger.info(
                f"Loading initial model from {args.checkpoint.init_ckpt_path}"
            )
            if args.checkpoint.continue_training_from_init:
                load_from_checkpoint(
                    args.checkpoint.init_ckpt_path,
                    model,
                    optimizer=optimizer,
                    model_key="model",
                    legacy_lm_transformer=args.checkpoint.legacy_init_ckpt_lm_transformer,
                )
            else:
                load_from_checkpoint(
                    args.checkpoint.init_ckpt_path,
                    model,
                    model_key="model",
                    legacy_lm_transformer=args.checkpoint.legacy_init_ckpt_lm_transformer,
                )
            model.rope_embeddings.reset_parameters()

            # Check for pre-computed STEM embeddings in the init checkpoint
            stem_shards_dir = (
                Path(args.checkpoint.init_ckpt_path) / "stem_shards"
            )
            if stem_shards_dir.exists() and any(
                stem_shards_dir.glob("stem_model_mp*.pt")
            ):
                logger.info(
                    "Pre-computed stem embeddings found in init checkpoint, "
                    "skipping random reset"
                )
            else:
                if getattr(args.model, "stem_embeddings_zero_reset", False):
                    logger.info(
                        "No pre-computed stem embeddings in init checkpoint, "
                        "initializing stem embeddings to zeros "
                        "(model.stem_embeddings_zero_reset=True)"
                    )
                    model.reset_stem_embeddings()
                else:
                    logger.info(
                        "No pre-computed stem embeddings in init checkpoint, "
                        "re-initializing with ParallelEmbedding default (Xavier normal)"
                    )
                    with torch.random.fork_rng(
                        devices=[torch.cuda.current_device()]
                    ):
                        torch.manual_seed(args.model.seed)
                        model.reset_stem_embeddings()

            sync_stem_embeddings_across_dp(model)

            # Re-freeze w3 after loading (checkpoint may have reset requires_grad)
            model.freeze_up_projections()

        # Load from latest checkpoint (or continue from init checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        # Re-freeze w3 after any checkpoint loading
        model.freeze_up_projections()

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

        # Log interpolation schedule info
        logger.info(
            f"Interpolation schedule: half_life={args.interp_half_life}, "
            f"interp_steps={args.interp_steps}"
        )
        alpha_at_start = compute_interp_alpha(
            train_state.step, args.interp_half_life, args.interp_steps
        )
        logger.info(f"Alpha at step {train_state.step}: {alpha_at_start:.6f}")

        # ---- Train loop ----
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
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # ---- Compute & set interpolation alpha ----
            alpha = compute_interp_alpha(
                train_state.step, args.interp_half_life, args.interp_steps
            )
            model.set_alpha(alpha)

            # ---- Get batch ----
            curr_lr = float(optimizer["lm"].param_groups[0]["lr"])
            curr_stem_lr = float(optimizer["stem"].param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                gc.collect()

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen = labels.shape

            # ---- Forward ----
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # Probe (optional)
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq,
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
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
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
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            loss = loss / args.grad_acc_steps
            loss.backward()
            loss = loss.detach() * args.grad_acc_steps

            # ---- Optimizer step ----
            grad_norm = -1.0
            stem_grad_norm = -1.0
            if train_state.acc_step == 0:
                # Clip gradients for lm_transformer (DTensors from FSDP)
                lm_params = [
                    p for p in model.lm_transformer.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                if lm_params:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        lm_params, max_norm=args.optim.clip, foreach=True
                    )
                    grad_norm = (
                        grad_norm.full_tensor()
                        if isinstance(grad_norm, DTensor)
                        else grad_norm
                    ).item()

                # Clip gradients for stem_embeddings (regular Tensors)
                stem_params = [
                    p for p in model.stem_embeddings.parameters()
                    if p.grad is not None
                ]
                if stem_params:
                    stem_grad_norm = torch.nn.utils.clip_grad_norm_(
                        stem_params, max_norm=args.optim.clip, foreach=False
                    ).item()
                else:
                    all_stem_params = list(model.stem_embeddings.parameters())
                    params_without_grad = [
                        p for p in all_stem_params if p.grad is None
                    ]
                    if params_without_grad:
                        logger.warning(
                            f"Warning: {len(params_without_grad)}/"
                            f"{len(all_stem_params)} stem_embeddings "
                            f"parameters have no gradients."
                        )

                # Sync stem_embeddings gradients across STEM data-parallel ranks
                if get_stem_data_parallel_world_size() > 1:
                    dp_group = get_stem_data_parallel_group()
                    for param in model.stem_embeddings.parameters():
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
                start_timer.elapsed_time(end_timer) * 1e-3, 4
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
                    total_acc_steps * args.data.batch_size * args.data.seq_len
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
                    "alpha": alpha,
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
                    f"  alpha: {alpha:.4f}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  stem_lr: {curr_stem_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )
                logger.info(log_msg)

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
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
                    train_state, args.checkpoint.eval.every, acc_step=0
                )
                or every_n_steps(train_state, args.steps, acc_step=0)
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

                # Pre-consolidate checkpoint: strip frozen w3 weights from
                # stem layers so the consolidated file matches the
                # StemLMTransformer format expected by the eval pipeline.
                if get_is_master():
                    consolidate_interpolated_checkpoints(
                        eval_args.ckpt_dir,
                        stem_layers=list(model.stem_layers),
                    )
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

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
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(InterpolatedFinetuneArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()

