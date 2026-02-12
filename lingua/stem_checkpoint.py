# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
from typing import Optional, Union, Any
import logging
import json
import re
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict as dcp_get_state_dict,
    get_model_state_dict as dcp_get_model_state_dict,
)

from pathlib import Path
from typing import Dict

# Make sure ParallelEmbedding and the STEM PG helpers are imported
from lingua.stem_dist_utils import (
    ParallelEmbedding, 
    get_stem_model_parallel_rank, 
    get_stem_data_parallel_rank,
    get_stem_model_parallel_world_size,
)
from lingua.distributed import get_is_master
from lingua.checkpoint import CheckpointManager, load_from_checkpoint as load_backbone_from_checkpoint
from omegaconf import OmegaConf

logger = logging.getLogger("STEM_CHECKPOINT")

FOLDER_NAME = "{:010d}"
RE_FOLDER = r"\d{10}"

RE_CKPT = r"__\d_\d\.distcp"

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"
CONSOLIDATE_STEM_NAME = "consolidated_stem.pth"

CONFIG_NAME = "params.json"
TRAIN_STATE_NAME = "train_state_{:05d}.json"
RE_DIGITS = re.compile(r"\d+")

STEM_SUBDIR_NAME = "stem_shards"
STEM_MODEL_FILE_TEMPLATE = "stem_model_mp{mp_rank}.pt"
STEM_OPTIM_FILE_TEMPLATE = "stem_optim_mp{mp_rank}.pt"

def _get_key_step(name: str):
    """Extract step number from checkpoint folder name."""
    return int(re.findall(RE_DIGITS, name)[-1])

def _iter_stem_params(model: nn.Module):
    for module_name, module in model.named_modules():
        if isinstance(module, ParallelEmbedding):
            for p_name, p in module.named_parameters(recurse=False):
                fq_name = f"{module_name}.{p_name}" if module_name else p_name
                yield fq_name, p
                
def extract_stem_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Build a state_dict only containing ParallelEmbedding params.
    """
    return {name: param for name, param in _iter_stem_params(model)}


def extract_stem_optimizer_state_dict(
    stem_optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> Dict[str, Any]:
    """
    Extract optimizer state dict for stem embeddings (ParallelEmbedding params only).
    
    Returns a filtered optimizer state dict containing only states for ParallelEmbedding parameters.
    """
    # Get set of stem parameter objects
    stem_params_set = {p for _, p in _iter_stem_params(model)}
    
    # Filter optimizer.state to only include stem parameters
    # optimizer.state uses actual parameter tensors as keys
    filtered_state = {}
    for param, state in stem_optimizer.state.items():
        if param in stem_params_set:
            filtered_state[param] = state
    
    # Build param_groups containing only stem params
    # optimizer.param_groups has actual parameter tensors in "params" list
    filtered_param_groups = []
    for group in stem_optimizer.param_groups:
        stem_params_in_group = [p for p in group["params"] if p in stem_params_set]
        if stem_params_in_group:
            filtered_group = {k: v for k, v in group.items() if k != "params"}
            filtered_group["params"] = stem_params_in_group
            filtered_param_groups.append(filtered_group)
    
    return {
        "state": filtered_state,
        "param_groups": filtered_param_groups,
    }



def split_backbone_and_stem_state_dict(
    model: nn.Module,
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
    model_key: str = "model",
    optim_key: str = "optim",
    stem_optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Build a DCP-compatible state_dict for the backbone only, and separate
    STEM (ParallelEmbedding) state_dicts for model params and optimizer states.
    
    Args:
        model: The model
        optimizer: Backbone optimizer (for lm_transformer) - can be dict with "lm" key or single optimizer
        model_key: Key for model in state dict
        optim_key: Key for optimizer in state dict
        stem_optimizer: Separate optimizer for stem embeddings
    
    Returns:
        fsdp_state_dict: Backbone state dict for DCP
        stem_model_sd: Stem model parameters state dict
        stem_optim_sd: Stem optimizer state dict (None if stem_optimizer not provided)
    """
    # Handle optimizer - extract lm_optimizer if it's a dict
    if isinstance(optimizer, dict):
        backbone_optimizer = optimizer.get("lm")
    else:
        backbone_optimizer = optimizer
    
    if backbone_optimizer is not None:
        model_sd, optim_sd = dcp_get_state_dict(model, backbone_optimizer)
    else:
        model_sd = dcp_get_model_state_dict(model)
        optim_sd = None

    # STEM params (ParallelEmbedding)
    stem_model_sd = extract_stem_state_dict(model)

    # Remove STEM params from the DCP-managed model state
    for k in stem_model_sd.keys():
        if k in model_sd:
            del model_sd[k]

    fsdp_state_dict: Dict[str, Any] = {}
    if model_key != "":
        fsdp_state_dict[model_key] = model_sd
    else:
        # "bare" model dict for single-entity checkpoints
        fsdp_state_dict = model_sd

    if optim_sd is not None and backbone_optimizer is not None and optim_key:
        fsdp_state_dict[optim_key] = optim_sd

    # Extract stem optimizer state if provided
    stem_optim_sd = None
    if stem_optimizer is not None:
        stem_optim_sd = extract_stem_optimizer_state_dict(stem_optimizer, model)
    return fsdp_state_dict, stem_model_sd, stem_optim_sd


def save_stem_shards(
    stem_model_sd: Dict[str, torch.Tensor],
    ckpt_dir: Path,
    model: nn.Module,
    stem_optim_sd: Optional[Dict[str, Any]] = None,
):
    """
    Save local STEM shards (model params and optionally optimizer states) for this STEM MP rank.

    Assumptions:
    - STEM model parallel topology (world size / rank mapping) is
      the same at save and load time.
    - STEM params are always sharded the same way (model parallel across embedding dim).
    """
    if not stem_model_sd and not stem_optim_sd:
        return

    # Ensure stem_shards directory exists before any rank writes.
    # The mkdir is done by rank 0; the barrier must include ALL ranks
    # (not just dp_rank == 0) to keep the global-PG operation count
    # balanced — otherwise downstream barriers/collectives will mismatch.
    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    if get_is_master() and not stem_dir.exists():
        stem_dir.mkdir(parents=False, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    mp_rank = get_stem_model_parallel_rank()
    dp_rank = get_stem_data_parallel_rank()

    # Save only one DP replica per MP shard to avoid duplicates
    if dp_rank != 0:
        return

    # Save model parameters
    if stem_model_sd:
        model_shard_path = stem_dir / STEM_MODEL_FILE_TEMPLATE.format(mp_rank=mp_rank)
        # Move to CPU for device-agnostic checkpointing
        cpu_model_sd = {k: v.detach().cpu() for k, v in stem_model_sd.items()}
        torch.save(cpu_model_sd, model_shard_path)
        logger.info(f"Saved stem model shard for mp_rank={mp_rank} to {model_shard_path}")

    # Save optimizer states
    if stem_optim_sd:
        optim_shard_path = stem_dir / STEM_OPTIM_FILE_TEMPLATE.format(mp_rank=mp_rank)
        
        # Build mapping from param_id to param_name for serialization
        param_id_to_name = {id(param): name for name, param in _iter_stem_params(model)}
        
        # Convert optimizer state dict to use param names instead of IDs
        cpu_optim_sd = {
            "state": {},  # Will use param names as keys
            "param_groups": [],
        }
        
        # Convert state dict: param object -> param_name
        # Note: optimizer state dict uses parameter objects as keys, not IDs
        for param_obj, param_state in stem_optim_sd.get("state", {}).items():
            param_id = id(param_obj)
            if param_id in param_id_to_name:
                param_name = param_id_to_name[param_id]
                cpu_param_state = {}
                for state_key, state_value in param_state.items():
                    if isinstance(state_value, torch.Tensor):
                        cpu_param_state[state_key] = state_value.detach().cpu()
                    else:
                        cpu_param_state[state_key] = state_value
                cpu_optim_sd["state"][param_name] = cpu_param_state
        
        # Convert param_groups: replace param objects with param names
        for group in stem_optim_sd.get("param_groups", []):
            cpu_group = {k: v for k, v in group.items() if k != "params"}
            cpu_group["param_names"] = [
                param_id_to_name[id(param_obj)] 
                for param_obj in group["params"] 
                if id(param_obj) in param_id_to_name
            ]
            cpu_optim_sd["param_groups"].append(cpu_group)
        
        torch.save(cpu_optim_sd, optim_shard_path)
        logger.info(f"Saved stem optimizer shard for mp_rank={mp_rank} to {optim_shard_path}")
    
    
def load_stem_shards(
    model: nn.Module,
    ckpt_dir: Path,
    stem_optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
):
    """
    Load STEM shards (model params and optionally optimizer states) for current STEM MP rank.
    Assumes the checkpoint was saved with the same model parallel world size as the current run.
    
    Args:
        model: The model containing ParallelEmbedding modules
        ckpt_dir: Checkpoint directory
        stem_optimizer: Optional optimizer for stem embeddings to load state into
        map_location: Device to load tensors to
    """
    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    if not stem_dir.exists():
        # No STEM checkpoint -> assume fresh init
        logger.info("No stem_shards directory found, skipping stem checkpoint load")
        return

    mp_rank = get_stem_model_parallel_rank()
    load_loc = map_location or torch.device("cuda", torch.cuda.current_device())

    # Load model parameters
    model_shard_path = stem_dir / STEM_MODEL_FILE_TEMPLATE.format(mp_rank=mp_rank)
    if model_shard_path.exists():
        loaded_model_sd = torch.load(model_shard_path, map_location=load_loc)
        # Copy tensors into actual ParallelEmbedding params
        with torch.no_grad():
            for module_name, module in model.named_modules():
                if isinstance(module, ParallelEmbedding):
                    for p_name, p in module.named_parameters(recurse=False):
                        fq_name = f"{module_name}.{p_name}" if module_name else p_name
                        if fq_name in loaded_model_sd:
                            p.copy_(loaded_model_sd[fq_name].to(p.device))
        logger.info(f"Loaded stem model shard for mp_rank={mp_rank} from {model_shard_path}")
    else:
        logger.warning(f"Stem model shard for mp_rank={mp_rank} not found at {model_shard_path}")

    # Load optimizer states
    if stem_optimizer is not None:
        optim_shard_path = stem_dir / STEM_OPTIM_FILE_TEMPLATE.format(mp_rank=mp_rank)
        if optim_shard_path.exists():
            loaded_optim_sd = torch.load(optim_shard_path, map_location=load_loc)
            
            # Reconstruct parameter objects from names
            param_name_to_obj = {name: param for name, param in _iter_stem_params(model)}
            
            # Get current optimizer's state dict to preserve its structure
            current_optim_sd = stem_optimizer.state_dict()
            
            # Build mapping from param names to param objects for loaded state
            loaded_state = loaded_optim_sd.get("state", {})
            
            # Reconstruct state dict using current optimizer's param_groups structure
            # but updating states from loaded checkpoint
            state_dict_to_load = {
                "state": {},
                "param_groups": current_optim_sd["param_groups"],  # Use current structure
            }
            
            # Map loaded states (keyed by param_name) to param objects
            for param_name, loaded_param_state in loaded_state.items():
                if param_name in param_name_to_obj:
                    param_obj = param_name_to_obj[param_name]
                    # Move state tensors to correct device
                    device_state = {}
                    for state_key, state_value in loaded_param_state.items():
                        if isinstance(state_value, torch.Tensor):
                            device_state[state_key] = state_value.to(load_loc)
                        else:
                            device_state[state_key] = state_value
                    state_dict_to_load["state"][param_obj] = device_state
            
            # Load into optimizer
            stem_optimizer.load_state_dict(state_dict_to_load)
            logger.info(f"Loaded stem optimizer shard for mp_rank={mp_rank} from {optim_shard_path}")
        else:
            logger.warning(f"Stem optimizer shard for mp_rank={mp_rank} not found at {optim_shard_path}, starting with fresh optimizer state")


SOURCE_STEM_MP_SIZE = 8  # Default: checkpoints are always saved with 8-way sharding


def load_stem_shards_resharded(
    model: nn.Module,
    ckpt_dir: Path,
    stem_optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
):
    """
    Generalized STEM shard loader that handles resharding from ``source_mp_size``
    shards to the current STEM model-parallel world size.

    The source checkpoint is assumed to have ``source_mp_size`` shards (default 8).
    The current model-parallel world size must be <= ``source_mp_size`` and must
    evenly divide it.  When loading with fewer ranks than source shards,
    consecutive source shards are concatenated along the embedding dimension
    (dim=1 of the weight tensor) for each target rank.

    Examples with source_mp_size=8:
      - target_mp_size=8: rank i loads shard i              (1-to-1)
      - target_mp_size=4: rank i loads shards [2i, 2i+1]    (merge 2)
      - target_mp_size=2: rank i loads shards [4i .. 4i+3]  (merge 4)
      - target_mp_size=1: rank 0 loads all 8 shards          (merge 8)

    Args:
        model: The model containing ParallelEmbedding modules.
        ckpt_dir: Root checkpoint directory (must contain a ``stem_shards/`` sub-dir).
        stem_optimizer: Optional optimizer for stem embeddings to restore state into.
        map_location: Device to map loaded tensors to (defaults to current CUDA device).
        source_mp_size: Number of model-parallel shards the checkpoint was saved with.
    """
    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    if not stem_dir.exists():
        logger.info("No stem_shards directory found, skipping stem checkpoint load")
        return

    source_mp_size = len(list(stem_dir.glob("stem_model_mp*.pt")))
    target_mp_size = get_stem_model_parallel_world_size()
    mp_rank = get_stem_model_parallel_rank()
    load_loc = map_location or torch.device("cuda", torch.cuda.current_device())

    # --- Validate resharding geometry ---
    assert source_mp_size >= target_mp_size, (
        f"Source MP size ({source_mp_size}) must be >= target MP size ({target_mp_size})"
    )
    assert source_mp_size % target_mp_size == 0, (
        f"Source MP size ({source_mp_size}) must be evenly divisible by "
        f"target MP size ({target_mp_size})"
    )

    shards_per_rank = source_mp_size // target_mp_size
    source_shard_start = mp_rank * shards_per_rank
    source_shard_indices = list(range(source_shard_start, source_shard_start + shards_per_rank))

    logger.info(
        f"Resharding STEM: source_mp_size={source_mp_size}, target_mp_size={target_mp_size}, "
        f"mp_rank={mp_rank}, loading source shards {source_shard_indices}"
    )

    # ------------------------------------------------------------------
    # 1) Load & merge model parameters
    # ------------------------------------------------------------------
    merged_model_sd: Dict[str, list] = {}
    for src_rank in source_shard_indices:
        shard_path = stem_dir / STEM_MODEL_FILE_TEMPLATE.format(mp_rank=src_rank)
        if shard_path.exists():
            shard_sd = torch.load(shard_path, map_location=load_loc)
            for k, v in shard_sd.items():
                merged_model_sd.setdefault(k, []).append(v)
        else:
            logger.warning(f"Source stem model shard not found at {shard_path}")

    # Concatenate along embedding dim (dim=1 for weight shape [V, D/source_mp_size])
    for k in merged_model_sd:
        merged_model_sd[k] = torch.cat(merged_model_sd[k], dim=1)

    # Copy merged weights into the model's ParallelEmbedding parameters
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, ParallelEmbedding):
                for p_name, p in module.named_parameters(recurse=False):
                    fq_name = f"{module_name}.{p_name}" if module_name else p_name
                    if fq_name in merged_model_sd:
                        p.copy_(merged_model_sd[fq_name].to(p.device))

    logger.info(
        f"Loaded & merged stem model shards {source_shard_indices} "
        f"for mp_rank={mp_rank}"
    )

    # ------------------------------------------------------------------
    # 2) Load & merge optimizer states
    # ------------------------------------------------------------------
    if stem_optimizer is None:
        return

    # Collect optimizer states from all source shards for this rank.
    # Structure: param_name -> state_key -> [tensor_from_each_shard]
    merged_optim_state: Dict[str, Dict[str, list]] = {}
    any_loaded = False

    for src_rank in source_shard_indices:
        optim_path = stem_dir / STEM_OPTIM_FILE_TEMPLATE.format(mp_rank=src_rank)
        if not optim_path.exists():
            logger.warning(f"Source stem optimizer shard not found at {optim_path}")
            continue
        shard_optim = torch.load(optim_path, map_location=load_loc)
        any_loaded = True

        for param_name, param_state in shard_optim.get("state", {}).items():
            if param_name not in merged_optim_state:
                merged_optim_state[param_name] = {}
            for state_key, state_value in param_state.items():
                merged_optim_state[param_name].setdefault(state_key, []).append(
                    state_value
                )

    if not any_loaded:
        logger.warning(
            "No stem optimizer shards found, starting with fresh optimizer state"
        )
        return

    # Build the state dict that can be loaded into the current optimizer
    param_name_to_obj = {name: param for name, param in _iter_stem_params(model)}
    current_optim_sd = stem_optimizer.state_dict()

    state_dict_to_load: Dict[str, Any] = {
        "state": {},
        "param_groups": current_optim_sd["param_groups"],  # keep current structure
    }

    for param_name, per_key_values in merged_optim_state.items():
        if param_name not in param_name_to_obj:
            continue
        param_obj = param_name_to_obj[param_name]
        merged_state: Dict[str, Any] = {}

        for state_key, values in per_key_values.items():
            first = values[0]
            if isinstance(first, torch.Tensor):
                if first.dim() >= 2:
                    # 2-D+ state (exp_avg, exp_avg_sq, …) — concat along
                    # the embedding dimension (dim=1), mirroring the weight
                    merged_state[state_key] = torch.cat(values, dim=1).to(load_loc)
                elif first.dim() == 1:
                    # 1-D state — unusual for embeddings but handle safely
                    merged_state[state_key] = torch.cat(values, dim=0).to(load_loc)
                else:
                    # 0-D (scalar) tensor (e.g. ``step``) — identical across
                    # shards, take the first
                    merged_state[state_key] = first.to(load_loc)
            else:
                # Non-tensor scalars (int / float step counters)
                merged_state[state_key] = first

        state_dict_to_load["state"][param_obj] = merged_state

    stem_optimizer.load_state_dict(state_dict_to_load)
    logger.info(
        f"Loaded & merged stem optimizer shards {source_shard_indices} "
        f"for mp_rank={mp_rank}"
    )
                        

def load_from_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
    model_key: str = "model",
    optim_key: str = "optim",
):
    ckpt_path = Path(ckpt_dir)

    if not (ckpt_path / ".metadata").exists():
        raise ValueError(
            "Please convert the checkpoint to distcp format using "
            "`torch.distributed.checkpoint.format_utils.torch_save_to_dcp` before loading it"
        )

    # Extract stem_optimizer from optimizer dict if present
    if isinstance(optimizer, dict):
        backbone_optimizer = optimizer.get("lm")
        stem_optimizer = optimizer.get("stem")
    else:
        backbone_optimizer = optimizer
        stem_optimizer = None

    # 1) Build backbone state_dict containers WITHOUT STEM params
    backbone_state_dict = {}
    if backbone_optimizer is not None:
        backbone_state_dict[model_key], backbone_state_dict[optim_key] = dcp_get_state_dict(model.lm_transformer, backbone_optimizer)
    else:
        backbone_state_dict[model_key] = dcp_get_model_state_dict(model.lm_transformer)
        if model_key == "": 
            backbone_state_dict = backbone_state_dict.pop(model_key)
            
    dcp.load(backbone_state_dict, checkpoint_id=str(ckpt_path))

    # 3) Load STEM shards (model params and optimizer states) for current STEM MP rank
    #    (no-op if stem_shards dir doesn't exist, e.g. old checkpoints)
    load_stem_shards_resharded(model, ckpt_path, stem_optimizer=stem_optimizer)
  

def consolidate_stem_shards(ckpt_dir: str):
    """
    Consolidates all STEM shards in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    consolidate_path = Path(ckpt_dir) / CONSOLIDATE_FOLDER
    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    consolidate_state_dict = {}
    for shard_file in stem_dir.glob("stem_model_mp*.pt"):
        state_dict = torch.load(shard_file, map_location="cpu")
        for k, v in state_dict.items():
            if k in consolidate_state_dict:
                consolidate_state_dict[k].append(v)
            else:
                consolidate_state_dict[k] = [v]
    
    for k, v in consolidate_state_dict.items():
        consolidate_state_dict[k] = torch.cat(v, dim=1)
    torch.save(consolidate_state_dict, consolidate_path / CONSOLIDATE_STEM_NAME)
    logger.info("Consolidated STEM shards !")
    return consolidate_path
                        
class StemCheckpointManager(CheckpointManager):
    
    @torch.no_grad()
    def get_state_dict(
        self,
        model,
        optimizer,
        stem_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Returns:
          fsdp_state_dict: mapping to be passed to dcp.save/dcp.load
          stem_model_sd: local STEM (ParallelEmbedding) model params shard for this rank
          stem_optim_sd: local STEM optimizer state shard for this rank (None if stem_optimizer not provided)
        """
        fsdp_state_dict, stem_model_sd, stem_optim_sd = split_backbone_and_stem_state_dict(
            model, optimizer, stem_optimizer=stem_optimizer
        )
        return fsdp_state_dict, stem_model_sd, stem_optim_sd
    
    def clean_up(self):
        """Override clean_up to allow stem_shards directory in addition to consolidated."""
        logger.info("Cleaning up checkpoints...")
        dump_folders = []
        eval_folders = []
        other_folders = []
        for p in self.existing_saves:
            is_dump = _get_key_step(p.name) % self.dump_every.every == 0
            is_eval = _get_key_step(p.name) % self.eval_every.every == 0
            if is_dump:
                dump_folders.append(p)
            if is_eval:
                eval_folders.append(p)
            if not (is_dump or is_eval):
                other_folders.append(p)

        logger.info(f"Dump folders: {dump_folders}")
        logger.info(f"Eval folders: {eval_folders}")
        logger.info(f"Other folders: {other_folders}")

        if self.dump_every.keep > 0:
            dump_folders = dump_folders[-self.dump_every.keep :]
        if self.eval_every.keep > 0:
            eval_folders = eval_folders[-self.eval_every.keep :]

        folder_to_keep = set(other_folders + dump_folders + eval_folders)
        folder_to_remove = set(self.existing_saves) - folder_to_keep

        logger.info(f"Removing folders: {folder_to_remove}")

        if dist.get_rank() == 0:
            for folder in folder_to_remove:
                for file in folder.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        # Allow both consolidated and stem_shards directories
                        assert file.name in [CONSOLIDATE_FOLDER, STEM_SUBDIR_NAME], (
                            f"Unexpected directory {file.name} in checkpoint folder. "
                            f"Expected one of: {CONSOLIDATE_FOLDER}, {STEM_SUBDIR_NAME}"
                        )
                        for f in file.iterdir():
                            f.unlink()
                        file.rmdir()
                folder.rmdir()

        dist.barrier()

        self.existing_saves = list(folder_to_keep)
        self.existing_saves.sort(key=lambda p: _get_key_step(p.name))
    
    def save(
        self,
        model,
        optimizer,
        train_state,
        config,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> bool:

        # When creating directory check if only rank0 or is there other solution
        path = Path(self.path)
        curr_save_dir = self._create_folder(path, FOLDER_NAME.format(train_state.step))
        logger.info(f"Saving to: {str(curr_save_dir)}")

        if dist.is_initialized():
            dist.barrier()

        logger.info("Saving...")
        # Extract stem_optimizer from optimizer dict if present
        stem_optimizer = None
        if isinstance(optimizer, dict):
            stem_optimizer = optimizer.get("stem")
        
        fsdp_state_dict, stem_model_sd, stem_optim_sd = self.get_state_dict(
            model, optimizer, stem_optimizer=stem_optimizer
        )

        # 1) Save backbone (FSDP/TP-managed) via DCP
        dcp.save(fsdp_state_dict, checkpoint_id=curr_save_dir)
        logger.info("Backbone model+optim state dict saved")

        # 2) Save STEM embeddings and optimizer states via custom sharded checkpoint
        save_stem_shards(stem_model_sd, curr_save_dir, model, stem_optim_sd=stem_optim_sd)
        logger.info("STEM (ParallelEmbedding) model and optimizer shards saved")
        
        logger.info("State dict saved!")

        if dist.is_initialized():
            dist.barrier()

        if get_is_master():
            with open(curr_save_dir / CONFIG_NAME, "w") as f:
                json.dump(
                    OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
                    f,
                )

        # Add json dump here
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME.format(dp_rank)
            logger.info(
                f"Saving train state to: {str(curr_save_dir / train_state_name)}"
            )
            with open(curr_save_dir / train_state_name, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved !")

        self.existing_saves.append(curr_save_dir)

        self.clean_up()

        if dist.is_initialized():
            dist.barrier()
        return True

    @torch.no_grad()
    def load(
        self,
        model: nn.Module,
        optimizer,
        train_state,
        device_mesh: DeviceMesh,
        path: Optional[Path] = None,
    ):
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        # Loading tries to load the provided path, if not available the last saved step and finally from the init path
        path = path or self.get_last_step_path(dp_rank=dp_rank)
        # If none of those are available don't do anything
        if path is None:
            # If no checkpoints exist do nothing
            return

        # Only load train state if it's provided, the files exist and we're not loading from init path
        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        train_state_path = path / train_state_name
        if train_state_path.exists() and train_state_path.stat().st_size > 0:
            logger.info("Reloading train state")
            with open(train_state_path, "r") as f:
                train_state_dict = json.load(f)
            train_state.load_state_dict(train_state_dict)
            logger.info("Train state reloaded")
        else:
            logger.info(f"Train state file not found or empty at {train_state_path}, skipping train state load")

        logger.info(f"Loading from: {str(path)}")
        # 1) Prepare containers for backbone state (FSDP/TP)
        # Extract stem_optimizer from optimizer dict if present
        stem_optimizer = None
        if isinstance(optimizer, dict):
            stem_optimizer = optimizer.get("stem")
        
        fsdp_state_dict, _, _ = self.get_state_dict(
            model=model,
            optimizer=optimizer,
            stem_optimizer=stem_optimizer,
        )

        # 2) Load backbone via DCP
        dcp.load(fsdp_state_dict, checkpoint_id=path)
        logger.info("Backbone model and optim reloaded (FSDP/TP)")

        # 3) Load STEM shards (model params and optimizer states) and copy into ParallelEmbedding params
        load_stem_shards(model, path, stem_optimizer=stem_optimizer)
        logger.info("STEM (ParallelEmbedding) model and optimizer shards reloaded")
        
    