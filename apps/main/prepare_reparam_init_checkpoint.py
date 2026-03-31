#!/usr/bin/env python3
"""
Prepare reparameterized STEM init checkpoint (distcp + residual stem_shards).

Inputs:
- Base checkpoint directory in distcp format (backbone weights).
- Warmup step directory from stem_projection_warmup.py containing:
  - projections.pt
  - stem_shards/stem_model_mp*.pt (token means in STEM space)

Outputs:
- New distcp checkpoint whose model state includes projection keys under the
  same namespace as token embeddings (e.g. `projections.{i}.weight` or
  `lm_transformer.projections.{i}.weight`).
- Residual stem shards from warmup checkpoint, resharded to target
  stem_parallel_size.
"""

import argparse
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp


logger = logging.getLogger(__name__)


def _resolve_warmup_step_dir(warmup_ckpt_path: Path, use_latest: bool) -> Path:
    if not warmup_ckpt_path.exists():
        raise FileNotFoundError(f"warmup checkpoint path does not exist: {warmup_ckpt_path}")
    if warmup_ckpt_path.is_dir() and re.match(r"\d{10}", warmup_ckpt_path.name):
        return warmup_ckpt_path
    if not use_latest:
        return warmup_ckpt_path
    step_dirs = sorted(
        [d for d in warmup_ckpt_path.iterdir() if d.is_dir() and re.match(r"\d{10}", d.name)],
        key=lambda p: int(p.name),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No numbered step dirs under {warmup_ckpt_path}")
    return step_dirs[-1]


def _load_warmup_projections(step_dir: Path, projection_file_name: str) -> Dict[int, torch.Tensor]:
    proj_path = step_dir / projection_file_name
    if not proj_path.exists():
        raise FileNotFoundError(f"Missing projection file: {proj_path}")
    raw = torch.load(proj_path, map_location="cpu")
    out: Dict[int, torch.Tensor] = {}
    for k, v in raw.items():
        m = re.match(r"(\d+)\.weight$", k)
        if m is None:
            continue
        out[int(m.group(1))] = v
    if not out:
        raise RuntimeError(f"No projection weights found in {proj_path}")
    return out


def _load_source_stem_shards(step_dir: Path) -> List[Dict[str, torch.Tensor]]:
    stem_dir = step_dir / "stem_shards"
    if not stem_dir.exists():
        raise FileNotFoundError(f"Missing stem_shards dir: {stem_dir}")
    shard_paths = sorted(stem_dir.glob("stem_model_mp*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No stem_model_mp*.pt files found in {stem_dir}")
    return [torch.load(p, map_location="cpu") for p in shard_paths]


def _merge_full_tables(shards: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = sorted(set().union(*[set(sd.keys()) for sd in shards]))
    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        parts = [sd[k] for sd in shards if k in sd]
        if not parts:
            continue
        merged[k] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    if not merged:
        raise RuntimeError("No STEM tensors found while merging warmup shards")
    return merged


def _read_base_checkpoint(base_init_ckpt_path: Path) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        consolidated_path = td / "base_consolidated.pth"
        dcp_to_torch_save(str(base_init_ckpt_path), str(consolidated_path))
        state = torch.load(consolidated_path, map_location="cpu")

    if isinstance(state, dict) and "model" in state:
        model_sd = state["model"]
    elif isinstance(state, dict):
        model_sd = state
    else:
        raise RuntimeError("Unsupported consolidated checkpoint format")
    return state, model_sd


def _save_resharded_stem(residuals: Dict[str, torch.Tensor], output_dir: Path, stem_parallel_size: int):
    sample = next(iter(residuals.values()))
    hidden_dim = sample.shape[1]
    if hidden_dim % stem_parallel_size != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by stem_parallel_size={stem_parallel_size}"
        )
    shard_w = hidden_dim // stem_parallel_size
    stem_dir = output_dir / "stem_shards"
    if stem_dir.exists():
        shutil.rmtree(stem_dir)
    stem_dir.mkdir(parents=True, exist_ok=True)

    for mp_rank in range(stem_parallel_size):
        start = mp_rank * shard_w
        end = start + shard_w
        shard_sd = {k: v[:, start:end].contiguous().cpu() for k, v in residuals.items()}
        out_path = stem_dir / f"stem_model_mp{mp_rank}.pt"
        torch.save(shard_sd, out_path)
        logger.info(f"Saved {out_path} (hidden slice [{start}:{end}])")


def _copy_base_init_checkpoint(base_init_ckpt_path: Path, output_dir: Path, overwrite: bool):
    if not base_init_ckpt_path.exists():
        raise FileNotFoundError(f"base init checkpoint does not exist: {base_init_ckpt_path}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    shutil.copytree(base_init_ckpt_path, output_dir)
    logger.info(f"Copied base init checkpoint: {base_init_ckpt_path} -> {output_dir}")


def _rewrite_distcp_with_projections(output_dir: Path, full_state: Dict):
    # Remove old distcp payload files and metadata before writing new payload.
    meta = output_dir / ".metadata"
    if meta.exists():
        meta.unlink()
    for p in output_dir.glob("__*.distcp"):
        p.unlink()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        torch_ckpt = td / "reparam_consolidated.pth"
        torch.save(full_state, torch_ckpt)
        torch_save_to_dcp(str(torch_ckpt), str(output_dir))
    logger.info(f"Rewrote distcp payload in {output_dir} with projection weights embedded")


def _update_params_json(output_dir: Path, stem_parallel_size: int):
    params_path = output_dir / "params.json"
    if not params_path.exists():
        logger.warning(f"No params.json found at {params_path}; skipping params update")
        return
    with open(params_path, "r") as f:
        params = json.load(f)
    if "model" not in params:
        params = {"model": params}
    params["distributed"] = params.get("distributed", {})
    params["distributed"]["stem_parallel_size"] = stem_parallel_size
    with open(params_path, "w") as f:
        json.dump(params, f)
    logger.info(f"Updated params.json with distributed.stem_parallel_size={stem_parallel_size}")


def main():
    parser = argparse.ArgumentParser(description="Prepare reparam init checkpoint with embedded projections.")
    parser.add_argument("--base-init-ckpt-path", required=True, help="Path to base init distcp checkpoint dir.")
    parser.add_argument("--warmup-ckpt-path", required=True, help="Path to warmup checkpoints root or exact step dir.")
    parser.add_argument("--output-dir", required=True, help="Output init checkpoint directory.")
    parser.add_argument("--stem-parallel-size", type=int, required=True, help="Target number of stem shards.")
    parser.add_argument("--use-latest-warmup-step", action="store_true", help="Select latest numbered warmup step.")
    parser.add_argument("--projection-file-name", default="projections.pt", help="Projection filename in warmup step.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it exists.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base_init_ckpt_path = Path(args.base_init_ckpt_path)
    warmup_ckpt_path = Path(args.warmup_ckpt_path)
    output_dir = Path(args.output_dir)

    step_dir = _resolve_warmup_step_dir(warmup_ckpt_path, args.use_latest_warmup_step)
    logger.info(f"Using warmup step dir: {step_dir}")

    _copy_base_init_checkpoint(base_init_ckpt_path, output_dir, args.overwrite)

    projections = _load_warmup_projections(step_dir, args.projection_file_name)
    source_shards = _load_source_stem_shards(step_dir)
    merged_residuals = _merge_full_tables(source_shards)

    full_state, model_sd = _read_base_checkpoint(base_init_ckpt_path)
    proj_prefix = "model.projections"

    # Embed projection weights into model state dict so runtime init uses a single distcp load.
    for idx, w in projections.items():
        model_sd[f"{proj_prefix}.{idx}.weight"] = w.cpu()

    _save_resharded_stem(merged_residuals, output_dir, args.stem_parallel_size)

    # Persist updated model payload (with projections included) back to distcp.
    if isinstance(full_state, dict) and "model" in full_state:
        full_state["model"] = model_sd
    else:
        full_state = model_sd
    _rewrite_distcp_with_projections(output_dir, full_state)

    _update_params_json(output_dir, args.stem_parallel_size)

    logger.info("")
    logger.info(f"Prepared reparam init checkpoint at: {output_dir}")
    logger.info("Use this path as checkpoint.init_ckpt_path in stem_reparam_train.py")


if __name__ == "__main__":
    main()
