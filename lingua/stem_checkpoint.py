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
)
from lingua.distributed import get_is_master
from lingua.checkpoint import CheckpointManager
from omegaconf import OmegaConf

logger = logging.getLogger("STEM_CHECKPOINT")

FOLDER_NAME = "{:010d}"
RE_FOLDER = r"\d{10}"

RE_CKPT = r"__\d_\d\.distcp"

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"

CONFIG_NAME = "params.json"
TRAIN_STATE_NAME = "train_state_{:05d}.json"
RE_DIGITS = re.compile(r"\d+")

STEM_SUBDIR_NAME = "stem_shards"
STEM_FILE_TEMPLATE = "stem_mp{mp_rank}.pt"

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



def split_backbone_and_stem_state_dict(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_key: str = "model",
    optim_key: str = "optim",
):
    """
    Build a DCP-compatible state_dict for the backbone only, and a separate
    STEM (ParallelEmbedding) state_dict for this rank.
    """
    if optimizer is not None:
        model_sd, optim_sd = dcp_get_state_dict(model, optimizer)
    else:
        model_sd = dcp_get_model_state_dict(model)
        optim_sd = None

    # STEM params (ParallelEmbedding)
    stem_sd = extract_stem_state_dict(model)

    # Remove STEM params from the DCP-managed model state
    for k in stem_sd.keys():
        if k in model_sd:
            del model_sd[k]

    fsdp_state_dict: Dict[str, Any] = {}
    if model_key != "":
        fsdp_state_dict[model_key] = model_sd
    else:
        # "bare" model dict for single-entity checkpoints
        fsdp_state_dict = model_sd

    if optim_sd is not None and optimizer is not None and optim_key:
        fsdp_state_dict[optim_key] = optim_sd

    return fsdp_state_dict, stem_sd


def save_stem_shards(
    stem_sd: Dict[str, torch.Tensor],
    ckpt_dir: Path,
):
    """
    Save local STEM shard for this STEM MP rank in its own file.

    Assumptions:
    - STEM model parallel topology (world size / rank mapping) is
      the same at save and load time.
    - STEM params are always sharded the same way (8-way dim parallel).
    """
    if not stem_sd:
        return

    mp_rank = get_stem_model_parallel_rank()
    dp_rank = get_stem_data_parallel_rank()

    # Save only one DP replica per MP shard to avoid duplicates
    if dp_rank != 0:
        return

    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    if get_is_master() and not stem_dir.exists():
        stem_dir.mkdir(parents=False, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    shard_path = stem_dir / STEM_FILE_TEMPLATE.format(mp_rank=mp_rank)

    # Move to CPU for device-agnostic checkpointing
    cpu_sd = {k: v.detach().cpu() for k, v in stem_sd.items()}
    torch.save(cpu_sd, shard_path)
    
    
def load_stem_shards(
    model: nn.Module,
    ckpt_dir: Path,
    map_location: Optional[Union[str, torch.device]] = None,
):
    """
    Load STEM shard for current STEM MP rank and copy into ParallelEmbedding params.
    """
    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    if not stem_dir.exists():
        # No STEM checkpoint -> assume fresh init
        return

    mp_rank = get_stem_model_parallel_rank()
    shard_path = stem_dir / STEM_FILE_TEMPLATE.format(mp_rank=mp_rank)
    if not shard_path.exists():
        raise FileNotFoundError(
            f"STEM shard for mp_rank={mp_rank} not found at {shard_path}"
        )

    load_loc = map_location or torch.device("cuda", torch.cuda.current_device())
    loaded_sd = torch.load(shard_path, map_location=load_loc)

    # Copy tensors into actual ParallelEmbedding params
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, ParallelEmbedding):
                for p_name, p in module.named_parameters(recurse=False):
                    fq_name = f"{module_name}.{p_name}" if module_name else p_name
                    if fq_name in loaded_sd:
                        p.copy_(loaded_sd[fq_name].to(p.device))
                        

def load_from_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_key: str = "model",
    optim_key: str = "optim",
):
    ckpt_path = Path(ckpt_dir)

    if not (ckpt_path / ".metadata").exists():
        raise ValueError(
            "Please convert the checkpoint to distcp format using "
            "`torch.distributed.checkpoint.format_utils.torch_save_to_dcp` before loading it"
        )

    # 1) Build backbone state_dict containers WITHOUT STEM params
    fsdp_state_dict, _ = split_backbone_and_stem_state_dict(
        model=model,
        optimizer=optimizer,
        model_key=model_key,
        optim_key=optim_key,
    )

    # 2) Load backbone (FSDP/TP-managed) via DCP
    dcp.load(fsdp_state_dict, checkpoint_id=str(ckpt_path))

    # 3) Load STEM shards for current STEM MP rank
    #    (no-op if stem_shards dir doesn't exist, e.g. old checkpoints)
    load_stem_shards(model, ckpt_path)
  
                        
class StemCheckpointManager(CheckpointManager):
    
    @torch.no_grad()
    def get_state_dict(
        self,
        model,
        optimizer,
    ):
        """
        Returns:
          fsdp_state_dict: mapping to be passed to dcp.save/dcp.load
          stem_state_dict: local STEM (ParallelEmbedding) shard for this rank
        """
        fsdp_state_dict, stem_state_dict = split_backbone_and_stem_state_dict(
            model, optimizer
        )
        return fsdp_state_dict, stem_state_dict
    
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
        fsdp_state_dict, stem_state_dict = self.get_state_dict(model, optimizer)

        # 1) Save backbone (FSDP/TP-managed) via DCP
        dcp.save(fsdp_state_dict, checkpoint_id=curr_save_dir)
        logger.info("Backbone model+optim state dict saved")

        # 2) Save STEM embeddings via custom sharded checkpoint
        save_stem_shards(stem_state_dict, curr_save_dir)
        logger.info("STEM (ParallelEmbedding) shards saved")
        
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
        logger.info("Reloading train state")
        with open(path / train_state_name, "r") as f:
            train_state_dict = json.load(f)
        train_state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        logger.info(f"Loading from: {str(path)}")
        # 1) Prepare containers for backbone state (FSDP/TP)
        fsdp_state_dict, _ = self.get_state_dict(
            model=model,
            optimizer=optimizer,
        )

        # 2) Load backbone via DCP
        dcp.load(fsdp_state_dict, checkpoint_id=path)
        logger.info("Backbone model and optim reloaded (FSDP/TP)")

        # 3) Load STEM shards and copy into ParallelEmbedding params
        load_stem_shards(model, path)
        logger.info("STEM (ParallelEmbedding) shards reloaded")