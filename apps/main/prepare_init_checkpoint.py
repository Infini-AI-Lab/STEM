#!/usr/bin/env python3
"""
Prepare an init checkpoint by rewriting DCP key prefixes.

Primary use-case:
  Convert STEM-saved backbone keys from:
      model.lm_transformer.layers.*
  to:
      model.layers.*
so they can be loaded by init logic that expects LM-only keys.

This script intentionally does NOT change training save/resume logic.
"""

import argparse
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.distributed.checkpoint.format_utils as dcp_format


logger = logging.getLogger(__name__)


TRAIN_STATE_PREFIX = "train_state_"
TRAIN_STATE_SUFFIX = ".json"
STEM_MODEL_PREFIX = "stem_model_mp"
STEM_OPTIM_PREFIX = "stem_optim_mp"
STEM_SHARD_SUFFIX = ".pt"


def _rename_prefixed_keys(obj: Any, src_prefix: str, dst_prefix: str) -> Any:
    """Recursively rename dict keys that start with src_prefix."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            new_key = k
            if isinstance(k, str) and k.startswith(src_prefix):
                new_key = dst_prefix + k[len(src_prefix) :]
            out[new_key] = _rename_prefixed_keys(v, src_prefix, dst_prefix)
        return out
    if isinstance(obj, list):
        return [_rename_prefixed_keys(x, src_prefix, dst_prefix) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_rename_prefixed_keys(x, src_prefix, dst_prefix) for x in obj)
    return obj


def _cast_floating_tensors(obj: Any, dtype: torch.dtype) -> Any:
    """Recursively cast floating-point tensors to dtype."""
    if isinstance(obj, dict):
        return {k: _cast_floating_tensors(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_floating_tensors(x, dtype) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_floating_tensors(x, dtype) for x in obj)
    if torch.is_tensor(obj) and obj.is_floating_point():
        return obj.to(dtype=dtype)
    return obj


def _copy_aux_files(
    src_dir: Path,
    dst_dir: Path,
    copy_train_states: bool,
    copy_stem_optim_shards: bool,
) -> None:
    """Copy non-DCP files needed by STEM init/eval workflows."""
    params_src = src_dir / "params.json"
    if params_src.exists():
        shutil.copy2(params_src, dst_dir / "params.json")
        logger.info("Copied params.json")

    stem_shards_src = src_dir / "stem_shards"
    stem_shards_dst = dst_dir / "stem_shards"
    if stem_shards_src.exists():
        if stem_shards_dst.exists():
            shutil.rmtree(stem_shards_dst)
        stem_shards_dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for p in stem_shards_src.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith(STEM_MODEL_PREFIX) and p.name.endswith(STEM_SHARD_SUFFIX):
                shutil.copy2(p, stem_shards_dst / p.name)
                copied += 1
            elif (
                copy_stem_optim_shards
                and p.name.startswith(STEM_OPTIM_PREFIX)
                and p.name.endswith(STEM_SHARD_SUFFIX)
            ):
                shutil.copy2(p, stem_shards_dst / p.name)
                copied += 1
        logger.info(
            "Copied stem_shards/ (%d files, include_stem_optim=%s)",
            copied,
            copy_stem_optim_shards,
        )

    if copy_train_states:
        # Optional: copy train state sidecars if present.
        for p in src_dir.iterdir():
            if (
                p.is_file()
                and p.name.startswith(TRAIN_STATE_PREFIX)
                and p.name.endswith(TRAIN_STATE_SUFFIX)
            ):
                shutil.copy2(p, dst_dir / p.name)
                logger.info(f"Copied {p.name}")


def _validate_input_dir(input_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input checkpoint directory not found: {input_dir}")
    if not (input_dir / ".metadata").exists():
        raise FileNotFoundError(
            f"Input directory is not a DCP checkpoint (missing .metadata): {input_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite DCP checkpoint key prefixes for init compatibility."
    )
    parser.add_argument("--input-dir", required=True, help="Source DCP checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Destination DCP checkpoint directory")
    parser.add_argument(
        "--src-prefix",
        default="lm_transformer.",
        help="Prefix to remove/replace inside model/optim key spaces",
    )
    parser.add_argument(
        "--dst-prefix",
        default="",
        help="Replacement prefix (default: empty string)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output dir if it exists",
    )
    parser.add_argument(
        "--drop-optim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove optimizer entry from converted checkpoint (default: enabled)",
    )
    parser.add_argument(
        "--keep-train-states",
        action="store_true",
        help="Copy train_state_*.json sidecars (default: do not copy)",
    )
    parser.add_argument(
        "--keep-stem-optim-shards",
        action="store_true",
        help="Copy stem_optim_mp*.pt files from stem_shards/ (default: do not copy)",
    )
    parser.add_argument(
        "--model-dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Dtype for model floating tensors in output checkpoint (default: bf16)",
    )
    parser.add_argument(
        "--model-key",
        default="model",
        help="Top-level key containing model weights (default: model)",
    )
    parser.add_argument(
        "--optim-key",
        default="optim",
        help="Top-level key for optimizer state (default: optim)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    _validate_input_dir(input_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="prepare_init_ckpt_") as tmpdir:
        tmpdir = Path(tmpdir)
        raw_torch_path = tmpdir / "raw_checkpoint.pt"
        rewritten_torch_path = tmpdir / "rewritten_checkpoint.pt"

        logger.info("Converting DCP -> torch.save format")
        dcp_format.dcp_to_torch_save(input_dir, raw_torch_path)

        logger.info("Loading checkpoint object to CPU")
        state_obj = torch.load(raw_torch_path, map_location="cpu", weights_only=False)

        logger.info(
            "Rewriting key prefixes: '%s' -> '%s'", args.src_prefix, args.dst_prefix
        )
        rewritten_obj = _rename_prefixed_keys(
            state_obj, src_prefix=args.src_prefix, dst_prefix=args.dst_prefix
        )

        if args.drop_optim and isinstance(rewritten_obj, dict) and args.optim_key in rewritten_obj:
            logger.info("Dropping optimizer state at key '%s'", args.optim_key)
            rewritten_obj.pop(args.optim_key, None)

        model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32
        if isinstance(rewritten_obj, dict) and args.model_key in rewritten_obj:
            logger.info(
                "Casting model floating tensors under '%s' to %s",
                args.model_key,
                args.model_dtype,
            )
            rewritten_obj[args.model_key] = _cast_floating_tensors(
                rewritten_obj[args.model_key], model_dtype
            )
        elif isinstance(rewritten_obj, dict):
            logger.warning(
                "Model key '%s' not found; skipping dtype cast", args.model_key
            )

        logger.info("Saving rewritten torch object")
        torch.save(rewritten_obj, rewritten_torch_path)

        logger.info("Converting torch.save -> DCP")
        dcp_format.torch_save_to_dcp(rewritten_torch_path, output_dir)

    _copy_aux_files(
        input_dir,
        output_dir,
        copy_train_states=args.keep_train_states,
        copy_stem_optim_shards=args.keep_stem_optim_shards,
    )
    logger.info("Prepared init checkpoint at: %s", output_dir)


if __name__ == "__main__":
    main()
