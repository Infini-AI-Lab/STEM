# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Longcat n-gram checkpoints: DCP backbone + vocabulary-parallel n-gram tables under
``ngram_shards/`` (``ngram_model_mp*.pt`` / ``ngram_optim_mp*.pt``).

Does not import ``lingua.stem_checkpoint``. Training uses
:class:`LongcatCheckpointManager` for periodic save/resume.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from torch.distributed._tensor import DeviceMesh
from torch.distributed.checkpoint.state_dict import (
    get_state_dict as dcp_get_state_dict,
    get_model_state_dict as dcp_get_model_state_dict,
)

from lingua.checkpoint import (
    CONFIG_NAME,
    CONSOLIDATE_FOLDER,
    CheckpointArgs,
    CheckpointManager,
    FOLDER_NAME,
    TRAIN_STATE_NAME,
)
from lingua.distributed import get_is_master
from lingua.stem_dist_utils import (
    ParallelEmbedding,
    VocabParallelEmbedding,
    get_stem_data_parallel_rank,
    get_stem_model_parallel_rank,
    get_stem_model_parallel_world_size,
)

logger = logging.getLogger("LONGCAT_CHECKPOINT")

RE_DIGITS = re.compile(r"\d+")


def _get_key_step(name: str) -> int:
    return int(re.findall(RE_DIGITS, name)[-1])


# On-disk layout (ngram-prefixed; not ``stem_shards``).
NGRAM_SHARD_SUBDIR = "ngram_shards"
NGRAM_MODEL_SHARD_TMPL = "ngram_model_mp{mp_rank}.pt"
NGRAM_OPTIM_SHARD_TMPL = "ngram_optim_mp{mp_rank}.pt"
NGRAM_MODEL_SHARD_GLOB = "ngram_model_mp*.pt"


def _partial_load_planner():
    try:
        from torch.distributed.checkpoint import DefaultLoadPlanner
    except ImportError:  # pragma: no cover
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    return DefaultLoadPlanner(allow_partial_load=True)


def _iter_vocab_parallel_params(model: nn.Module):
    for module_name, module in model.named_modules():
        if isinstance(module, (ParallelEmbedding, VocabParallelEmbedding)):
            for p_name, p in module.named_parameters(recurse=False):
                fq_name = f"{module_name}.{p_name}" if module_name else p_name
                yield fq_name, p


def _vocab_parallel_weight_cat_dim(model: nn.Module, fq_name: str) -> int:
    if not fq_name.endswith(".weight"):
        return 1
    prefix = fq_name.rsplit(".", 1)[0]
    try:
        mod = model.get_submodule(prefix)
    except (AttributeError, KeyError, ValueError):
        return 1
    if isinstance(mod, VocabParallelEmbedding):
        return 0
    return 1


def _extract_vocab_parallel_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param for name, param in _iter_vocab_parallel_params(model)}


def _extract_vocab_parallel_optimizer_state_dict(
    ngram_optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> Dict[str, Any]:
    vp_params = {p for _, p in _iter_vocab_parallel_params(model)}
    filtered_state = {
        param: state
        for param, state in ngram_optimizer.state.items()
        if param in vp_params
    }
    filtered_param_groups = []
    for group in ngram_optimizer.param_groups:
        params_in_group = [p for p in group["params"] if p in vp_params]
        if params_in_group:
            filtered_group = {k: v for k, v in group.items() if k != "params"}
            filtered_group["params"] = params_in_group
            filtered_param_groups.append(filtered_group)
    return {"state": filtered_state, "param_groups": filtered_param_groups}


def _split_backbone_and_vocab_parallel_state_dict(
    model: nn.Module,
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
    model_key: str = "model",
    optim_key: str = "optim",
    ngram_optimizer: Optional[torch.optim.Optimizer] = None,
):
    if isinstance(optimizer, dict):
        backbone_optimizer = optimizer.get("lm")
    else:
        backbone_optimizer = optimizer

    if backbone_optimizer is not None:
        model_sd, optim_sd = dcp_get_state_dict(model, backbone_optimizer)
    else:
        model_sd = dcp_get_model_state_dict(model)
        optim_sd = None

    vp_sd = _extract_vocab_parallel_state_dict(model)
    for k in vp_sd.keys():
        if k in model_sd:
            del model_sd[k]

    fsdp_state_dict: Dict[str, Any] = {}
    if model_key != "":
        fsdp_state_dict[model_key] = model_sd
    else:
        fsdp_state_dict = model_sd

    if optim_sd is not None and backbone_optimizer is not None and optim_key:
        fsdp_state_dict[optim_key] = optim_sd

    ngram_optim_sd = None
    if ngram_optimizer is not None:
        ngram_optim_sd = _extract_vocab_parallel_optimizer_state_dict(
            ngram_optimizer, model
        )
    return fsdp_state_dict, vp_sd, ngram_optim_sd


def _save_ngram_shards(
    ngram_model_sd: Dict[str, torch.Tensor],
    ckpt_dir: Path,
    model: nn.Module,
    ngram_optim_sd: Optional[Dict[str, Any]] = None,
) -> None:
    if not ngram_model_sd and not ngram_optim_sd:
        return

    ngram_dir = ckpt_dir / NGRAM_SHARD_SUBDIR
    if get_is_master() and not ngram_dir.exists():
        ngram_dir.mkdir(parents=False, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    mp_rank = get_stem_model_parallel_rank()
    dp_rank = get_stem_data_parallel_rank()
    if dp_rank != 0:
        return

    if ngram_model_sd:
        model_shard_path = ngram_dir / NGRAM_MODEL_SHARD_TMPL.format(mp_rank=mp_rank)
        cpu_model_sd = {k: v.detach().cpu() for k, v in ngram_model_sd.items()}
        torch.save(cpu_model_sd, model_shard_path)
        logger.info(
            "Saved ngram model shard mp_rank=%s to %s", mp_rank, model_shard_path
        )

    if ngram_optim_sd:
        optim_shard_path = ngram_dir / NGRAM_OPTIM_SHARD_TMPL.format(mp_rank=mp_rank)
        param_id_to_name = {id(param): name for name, param in _iter_vocab_parallel_params(model)}
        cpu_optim_sd: Dict[str, Any] = {"state": {}, "param_groups": []}
        for param_obj, param_state in ngram_optim_sd.get("state", {}).items():
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
        for group in ngram_optim_sd.get("param_groups", []):
            cpu_group = {k: v for k, v in group.items() if k != "params"}
            cpu_group["param_names"] = [
                param_id_to_name[id(param_obj)]
                for param_obj in group["params"]
                if id(param_obj) in param_id_to_name
            ]
            cpu_optim_sd["param_groups"].append(cpu_group)
        torch.save(cpu_optim_sd, optim_shard_path)
        logger.info(
            "Saved ngram optimizer shard mp_rank=%s to %s", mp_rank, optim_shard_path
        )


def _load_ngram_shards(
    model: nn.Module,
    ckpt_dir: Path,
    ngram_optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> None:
    ngram_dir = ckpt_dir / NGRAM_SHARD_SUBDIR
    if not ngram_dir.exists():
        logger.info("No %s directory, skipping ngram shard load", NGRAM_SHARD_SUBDIR)
        return

    mp_rank = get_stem_model_parallel_rank()
    load_loc = map_location or torch.device("cuda", torch.cuda.current_device())

    model_shard_path = ngram_dir / NGRAM_MODEL_SHARD_TMPL.format(mp_rank=mp_rank)
    if model_shard_path.exists():
        loaded_model_sd = torch.load(model_shard_path, map_location=load_loc)
        with torch.no_grad():
            for module_name, module in model.named_modules():
                if isinstance(module, (ParallelEmbedding, VocabParallelEmbedding)):
                    for p_name, p in module.named_parameters(recurse=False):
                        fq_name = f"{module_name}.{p_name}" if module_name else p_name
                        if fq_name in loaded_model_sd:
                            p.copy_(loaded_model_sd[fq_name].to(p.device))
        logger.info(
            "Loaded ngram model shard mp_rank=%s from %s", mp_rank, model_shard_path
        )
    else:
        logger.warning("Ngram model shard missing at %s", model_shard_path)

    if ngram_optimizer is None:
        return

    optim_shard_path = ngram_dir / NGRAM_OPTIM_SHARD_TMPL.format(mp_rank=mp_rank)
    if optim_shard_path.exists():
        loaded_optim_sd = torch.load(optim_shard_path, map_location=load_loc)
        param_name_to_obj = {name: param for name, param in _iter_vocab_parallel_params(model)}
        current_optim_sd = ngram_optimizer.state_dict()
        state_dict_to_load: Dict[str, Any] = {
            "state": {},
            "param_groups": current_optim_sd["param_groups"],
        }
        for param_name, loaded_param_state in loaded_optim_sd.get("state", {}).items():
            if param_name in param_name_to_obj:
                param_obj = param_name_to_obj[param_name]
                device_state = {}
                for state_key, state_value in loaded_param_state.items():
                    if isinstance(state_value, torch.Tensor):
                        device_state[state_key] = state_value.to(load_loc)
                    else:
                        device_state[state_key] = state_value
                state_dict_to_load["state"][param_obj] = device_state
        ngram_optimizer.load_state_dict(state_dict_to_load)
        logger.info(
            "Loaded ngram optimizer shard mp_rank=%s from %s",
            mp_rank,
            optim_shard_path,
        )
    else:
        logger.warning(
            "Ngram optimizer shard missing at %s, keeping current optimizer state",
            optim_shard_path,
        )


def _load_ngram_shards_resharded(
    model: nn.Module,
    ckpt_dir: Path,
    ngram_optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> None:
    ngram_dir = ckpt_dir / NGRAM_SHARD_SUBDIR
    if not ngram_dir.exists():
        logger.info("No %s directory, skipping ngram shard load", NGRAM_SHARD_SUBDIR)
        return

    source_mp_size = len(list(ngram_dir.glob(NGRAM_MODEL_SHARD_GLOB)))
    target_mp_size = get_stem_model_parallel_world_size()
    mp_rank = get_stem_model_parallel_rank()
    load_loc = map_location or torch.device("cuda", torch.cuda.current_device())

    assert source_mp_size >= target_mp_size, (
        f"Source MP size ({source_mp_size}) must be >= target MP size ({target_mp_size})"
    )
    assert source_mp_size % target_mp_size == 0, (
        f"Source MP size ({source_mp_size}) must be evenly divisible by "
        f"target MP size ({target_mp_size})"
    )

    shards_per_rank = source_mp_size // target_mp_size
    source_shard_start = mp_rank * shards_per_rank
    source_shard_indices = list(
        range(source_shard_start, source_shard_start + shards_per_rank)
    )

    logger.info(
        "Ngram shards: source_mp_size=%s target_mp_size=%s mp_rank=%s indices=%s",
        source_mp_size,
        target_mp_size,
        mp_rank,
        source_shard_indices,
    )

    merged_model_sd: Dict[str, List[torch.Tensor]] = {}
    for src_rank in source_shard_indices:
        shard_path = ngram_dir / NGRAM_MODEL_SHARD_TMPL.format(mp_rank=src_rank)
        if shard_path.exists():
            shard_sd = torch.load(shard_path, map_location=load_loc)
            for k, v in shard_sd.items():
                merged_model_sd.setdefault(k, []).append(v)
        else:
            logger.warning("Source ngram shard not found at %s", shard_path)

    for k in merged_model_sd:
        dim = _vocab_parallel_weight_cat_dim(model, k)
        merged_model_sd[k] = torch.cat(merged_model_sd[k], dim=dim)

    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, (ParallelEmbedding, VocabParallelEmbedding)):
                for p_name, p in module.named_parameters(recurse=False):
                    fq_name = f"{module_name}.{p_name}" if module_name else p_name
                    if fq_name in merged_model_sd:
                        p.copy_(merged_model_sd[fq_name].to(p.device))

    logger.info(
        "Loaded & merged ngram model shards %s for mp_rank=%s",
        source_shard_indices,
        mp_rank,
    )

    if ngram_optimizer is None:
        return

    merged_optim_state: Dict[str, Dict[str, list]] = {}
    any_loaded = False
    for src_rank in source_shard_indices:
        optim_path = ngram_dir / NGRAM_OPTIM_SHARD_TMPL.format(mp_rank=src_rank)
        if not optim_path.exists():
            logger.warning("Source ngram optimizer shard not found at %s", optim_path)
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
            "No ngram optimizer shards under %s; keeping fresh ngram optimizer state",
            ngram_dir,
        )
        return

    param_name_to_obj = {name: param for name, param in _iter_vocab_parallel_params(model)}
    current_optim_sd = ngram_optimizer.state_dict()
    state_dict_to_load: Dict[str, Any] = {
        "state": {},
        "param_groups": current_optim_sd["param_groups"],
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
                    cat_dim = _vocab_parallel_weight_cat_dim(model, param_name)
                    merged_state[state_key] = torch.cat(values, dim=cat_dim).to(load_loc)
                elif first.dim() == 1:
                    merged_state[state_key] = torch.cat(values, dim=0).to(load_loc)
                else:
                    merged_state[state_key] = first.to(load_loc)
            else:
                merged_state[state_key] = first
        state_dict_to_load["state"][param_obj] = merged_state

    ngram_optimizer.load_state_dict(state_dict_to_load)
    logger.info(
        "Loaded & merged ngram optimizer shards %s for mp_rank=%s",
        source_shard_indices,
        mp_rank,
    )


def load_ngram_shards_resharded_for_inference(
    model: nn.Module,
    step_ckpt_dir: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
) -> bool:
    """
    Load vocabulary-parallel n-gram weights from ``step_ckpt_dir / ngram_shards/``.

    Training saves these shards separately from the FSDP backbone; the consolidated
    ``.pth`` under ``consolidated/`` typically does **not** include full
    ``ngram_embeddings`` tensors. This calls :func:`_load_ngram_shards_resharded`
    with ``ngram_optimizer=None`` (same MP merge rules as resume).

    Returns ``True`` if shard files were found and load was attempted, ``False`` if
    no ``ngram_shards/`` directory or no ``ngram_model_mp*.pt`` files.
    """
    step_ckpt_dir = Path(step_ckpt_dir)
    ngram_dir = step_ckpt_dir / NGRAM_SHARD_SUBDIR
    if not ngram_dir.exists() or not any(ngram_dir.glob(NGRAM_MODEL_SHARD_GLOB)):
        return False
    _load_ngram_shards_resharded(
        model, step_ckpt_dir, ngram_optimizer=None, map_location=map_location
    )
    return True


def load_longcat_init_from_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
    model_key: str = "model",
    optim_key: str = "optim",
    legacy_lm_transformer: bool = False,
) -> None:
    """
    Load init DCP into a Longcat model.

    Non-legacy: full-root DCP; vocabulary-parallel weights live under
    ``ngram_shards/`` beside the DCP tree.

    Legacy: ``lm_transformer``-only DCP with partial load (dense checkpoints lack
    ``ngram_fused``). Call ``model.init_ngram_fused_projection()`` after when
    loading a dense baseline.
    """
    ckpt_path = Path(ckpt_dir)
    if not (ckpt_path / ".metadata").exists():
        raise ValueError(
            "Please convert the checkpoint to distcp format using "
            "`torch.distributed.checkpoint.format_utils.torch_save_to_dcp` before loading it"
        )

    if isinstance(optimizer, dict):
        backbone_optimizer = optimizer.get("lm")
        ngram_optimizer = optimizer.get("ngram")
    else:
        backbone_optimizer = optimizer
        ngram_optimizer = None

    if legacy_lm_transformer:
        backbone_state_dict: Dict[str, Any] = {}
        if backbone_optimizer is not None:
            backbone_state_dict[model_key], backbone_state_dict[optim_key] = (
                dcp_get_state_dict(model.lm_transformer, backbone_optimizer)
            )
        else:
            backbone_state_dict[model_key] = dcp_get_model_state_dict(
                model.lm_transformer
            )
            if model_key == "":
                backbone_state_dict = backbone_state_dict.pop(model_key)
        dcp.load(
            backbone_state_dict,
            checkpoint_id=str(ckpt_path),
            planner=_partial_load_planner(),
        )
        logger.info(
            "Longcat init: partial legacy load into lm_transformer from %s",
            ckpt_dir,
        )
    else:
        fsdp_state_dict, _, _ = _split_backbone_and_vocab_parallel_state_dict(
            model,
            optimizer,
            model_key=model_key,
            optim_key=optim_key,
            ngram_optimizer=ngram_optimizer,
        )
        dcp.load(fsdp_state_dict, checkpoint_id=str(ckpt_path))

    _load_ngram_shards_resharded(model, ckpt_path, ngram_optimizer=ngram_optimizer)


@torch.no_grad()
def merge_longcat_backbone_dcp_seed_then_warmup(
    model: nn.Module,
    optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    seed_ckpt_dir: str,
    warmup_ckpt_dir: str,
) -> None:
    """
    Legacy seed (``lm_transformer`` DCP, partial load) then warmup full-root DCP
    (partial load) and warmup ``ngram_shards/``.
    """
    seed_path = Path(seed_ckpt_dir)
    warmup_path = Path(warmup_ckpt_dir)
    for label, p in (("seed", seed_path), ("warmup", warmup_path)):
        if not (p / ".metadata").exists():
            raise ValueError(
                f"{label} checkpoint is not a DCP directory (missing .metadata): {p}"
            )

    if isinstance(optimizer, dict):
        backbone_optimizer = optimizer.get("lm")
        ngram_optimizer = optimizer.get("ngram")
    else:
        backbone_optimizer = optimizer
        ngram_optimizer = None

    if backbone_optimizer is None:
        raise ValueError(
            "merge_longcat_backbone_dcp_seed_then_warmup requires an LM optimizer "
            "in ``optimizer`` (e.g. ``{'lm': ..., 'ngram': ...}``)."
        )

    model_key = "model"
    optim_key = "optim"

    seed_lm_sd: Dict[str, Any] = {}
    seed_lm_sd[model_key], seed_lm_sd[optim_key] = dcp_get_state_dict(
        model.lm_transformer, backbone_optimizer
    )
    logger.info(
        "Longcat merge: partial legacy seed load (lm_transformer): %s",
        seed_ckpt_dir,
    )
    dcp.load(
        seed_lm_sd,
        checkpoint_id=str(seed_path),
        planner=_partial_load_planner(),
    )

    seed_ngram_dir = seed_path / NGRAM_SHARD_SUBDIR
    if seed_ngram_dir.exists() and any(seed_ngram_dir.glob(NGRAM_MODEL_SHARD_GLOB)):
        logger.info(
            "Longcat merge: loading ngram_shards from seed (optional): %s",
            seed_ckpt_dir,
        )
        _load_ngram_shards_resharded(
            model, seed_path, ngram_optimizer=ngram_optimizer
        )

    fsdp_state_dict, _, _ = _split_backbone_and_vocab_parallel_state_dict(
        model,
        optimizer,
        model_key=model_key,
        optim_key=optim_key,
        ngram_optimizer=ngram_optimizer,
    )
    logger.info(
        "Longcat merge: partial load from warmup (missing keys keep seed values): %s",
        warmup_ckpt_dir,
    )
    dcp.load(
        fsdp_state_dict,
        checkpoint_id=str(warmup_path),
        planner=_partial_load_planner(),
    )

    logger.info("Longcat merge: ngram_shards from warmup: %s", warmup_ckpt_dir)
    _load_ngram_shards_resharded(
        model, warmup_path, ngram_optimizer=ngram_optimizer
    )


class LongcatCheckpointManager(CheckpointManager):
    """Save/load DCP + ``ngram_shards/`` for Longcat n-gram training."""

    @torch.no_grad()
    def get_state_dict(
        self,
        model,
        optimizer,
        ngram_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        return _split_backbone_and_vocab_parallel_state_dict(
            model, optimizer, ngram_optimizer=ngram_optimizer
        )

    def clean_up(self) -> None:
        logger.info("Cleaning up Longcat checkpoints...")
        dump_folders: List[Path] = []
        eval_folders: List[Path] = []
        other_folders: List[Path] = []
        for p in self.existing_saves:
            is_dump = _get_key_step(p.name) % self.dump_every.every == 0
            is_eval = _get_key_step(p.name) % self.eval_every.every == 0
            if is_dump:
                dump_folders.append(p)
            if is_eval:
                eval_folders.append(p)
            if not (is_dump or is_eval):
                other_folders.append(p)

        if self.dump_every.keep > 0:
            dump_folders = dump_folders[-self.dump_every.keep :]
        if self.eval_every.keep > 0:
            eval_folders = eval_folders[-self.eval_every.keep :]

        folder_to_keep = set(other_folders + dump_folders + eval_folders)
        folder_to_remove = set(self.existing_saves) - folder_to_keep
        logger.info("Removing folders: %s", folder_to_remove)

        if dist.get_rank() == 0:
            for folder in folder_to_remove:
                for file in folder.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        assert file.name in (CONSOLIDATE_FOLDER, NGRAM_SHARD_SUBDIR), (
                            f"Unexpected directory {file.name} in checkpoint folder. "
                            f"Expected one of: {CONSOLIDATE_FOLDER}, {NGRAM_SHARD_SUBDIR}"
                        )
                        for f in file.iterdir():
                            f.unlink()
                        file.rmdir()
                folder.rmdir()

        dist.barrier()
        self.existing_saves = sorted(folder_to_keep, key=lambda p: _get_key_step(p.name))

    def save(
        self,
        model,
        optimizer,
        train_state,
        config,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> bool:
        path = Path(self.path)
        curr_save_dir = self._create_folder(path, FOLDER_NAME.format(train_state.step))
        logger.info("Saving Longcat checkpoint to %s", curr_save_dir)

        if dist.is_initialized():
            dist.barrier()

        ngram_optimizer = None
        if isinstance(optimizer, dict):
            ngram_optimizer = optimizer.get("ngram")
            
        fsdp_state_dict, ngram_model_sd, ngram_optim_sd = self.get_state_dict(
            model, optimizer, ngram_optimizer=ngram_optimizer
        )

        dcp.save(fsdp_state_dict, checkpoint_id=curr_save_dir)
        logger.info("Longcat backbone (DCP) saved")

        _save_ngram_shards(ngram_model_sd, curr_save_dir, model, ngram_optim_sd=ngram_optim_sd)
        logger.info("Ngram shards saved under %s", NGRAM_SHARD_SUBDIR)

        if dist.is_initialized():
            dist.barrier()

        if get_is_master():
            with open(curr_save_dir / CONFIG_NAME, "w") as f:
                json.dump(
                    OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
                    f,
                )

        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME.format(dp_rank)
            with open(curr_save_dir / train_state_name, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved to %s", curr_save_dir / train_state_name)

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
    ) -> None:
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        path = path or self.get_last_step_path(dp_rank=dp_rank)
        if path is None:
            return

        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        train_state_path = path / train_state_name
        if train_state_path.exists() and train_state_path.stat().st_size > 0:
            logger.info("Reloading train state")
            with open(train_state_path, "r") as f:
                train_state.load_state_dict(json.load(f))
            logger.info("Train state reloaded")
        else:
            logger.info(
                "Train state missing or empty at %s, skipping", train_state_path
            )

        ngram_optimizer = optimizer.get("ngram")
        logger.info("Loading Longcat checkpoint from %s", path)

        
        fsdp_state_dict, _, _ = self.get_state_dict(
            model=model,
            optimizer=optimizer,
            ngram_optimizer=ngram_optimizer,
        )

        dcp.load(fsdp_state_dict, checkpoint_id=path)
        logger.info("Longcat backbone (DCP) reloaded")

        _load_ngram_shards(model, path, ngram_optimizer=ngram_optimizer)
        logger.info("Ngram shards reloaded from %s", path / NGRAM_SHARD_SUBDIR)

    @classmethod
    def instantiate_and_make_dir(
        cls, args: CheckpointArgs, train_stage: Optional[int] = None
    ):
        if get_is_master():
            os.makedirs(args.path, exist_ok=True)
        dist.barrier()
        return cls(args, train_stage=train_stage)
