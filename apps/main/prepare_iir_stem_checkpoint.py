# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Prepare IIR-STEM checkpoint by adding IIR parameters to an existing STEM checkpoint.

Takes a base STEM-init checkpoint (already prepared by ``prepare_stem_checkpoint.py``)
and adds properly initialised IIR parameters (``raw_mu``, ``alpha``, ``m0``) for each
stem layer.  The IIR parameters are saved as ``stem_shards/iir_params.pt`` inside the
output checkpoint directory.

Since IIR parameters are small and replicated (not sharded), they live in a single
file separate from the per-rank stem embedding shards.

The output directory can be used directly as ``checkpoint.init_ckpt_path`` in
``stem_iir_train.py``.

Usage
-----
    # From scratch: prepare STEM checkpoint first, then add IIR params
    python apps/main/prepare_stem_checkpoint.py \\
        --ckpt-path checkpoints/Llama-3.2-1B/distcp \\
        --output-dir checkpoints/Llama-3.2-1B-stem-init \\
        --stem-layers 2 6 10 14 \\
        --stem-parallel-size 4

    python apps/main/prepare_iir_stem_checkpoint.py \\
        --stem-ckpt-path checkpoints/Llama-3.2-1B-stem-init \\
        --output-dir checkpoints/Llama-3.2-1B-iir-stem-init \\
        --mu-init 0.9 \\
        --alpha-init 1.0

    # Or add IIR params in-place (overwrite existing checkpoint):
    python apps/main/prepare_iir_stem_checkpoint.py \\
        --stem-ckpt-path checkpoints/Llama-3.2-1B-stem-init \\
        --mu-init 0.9 \\
        --alpha-init 1.0
"""

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)

IIR_PARAMS_FILE = "iir_params.pt"


def _inverse_sigmoid(x: float) -> float:
    assert 0.0 < x < 1.0, f"mu_init must be in (0, 1), got {x}"
    return math.log(x / (1.0 - x))


def build_iir_params(
    n_stem_layers: int,
    d_ff: int,
    mu_init: float,
    alpha_init: float,
    per_dim_alpha: bool,
    learnable_m0: bool,
) -> Dict[str, torch.Tensor]:
    """Create the IIR parameter dict for all stem layers.

    Keys follow the naming convention used by ``IIRStemLMTransformer``:
        ``iir_memories.{idx}.raw_mu``
        ``iir_memories.{idx}.alpha``
        ``iir_memories.{idx}.m0``
    """
    raw_mu = _inverse_sigmoid(mu_init)
    params: Dict[str, torch.Tensor] = {}

    for i in range(n_stem_layers):
        prefix = f"iir_memories.{i}"
        params[f"{prefix}.raw_mu"] = torch.tensor([raw_mu], dtype=torch.float32)

        if per_dim_alpha:
            params[f"{prefix}.alpha"] = torch.full((d_ff,), alpha_init, dtype=torch.float32)
        else:
            params[f"{prefix}.alpha"] = torch.tensor([alpha_init], dtype=torch.float32)

        params[f"{prefix}.m0"] = torch.zeros(d_ff, dtype=torch.float32)

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Add IIR parameters to an existing STEM checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stem-ckpt-path",
        required=True,
        help="Path to an existing STEM-init checkpoint (must contain stem_shards/).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory.  If omitted, writes iir_params.pt in-place "
            "inside --stem-ckpt-path."
        ),
    )
    parser.add_argument("--mu-init", type=float, default=0.9,
                        help="Initial EMA decay mu in (0,1) (default: %(default)s)")
    parser.add_argument("--alpha-init", type=float, default=1.0,
                        help="Initial memory scale alpha (default: %(default)s)")
    parser.add_argument("--per-dim-alpha", action="store_true",
                        help="Use per-dimension alpha vector instead of scalar")
    parser.add_argument("--learnable-m0", action="store_true",
                        help="Mark m0 as learnable (saved either way)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    src = Path(args.stem_ckpt_path)
    if not src.exists():
        logger.error(f"Source checkpoint does not exist: {src}")
        sys.exit(1)

    stem_shards_dir = src / "stem_shards"
    if not stem_shards_dir.exists():
        logger.error(
            f"No stem_shards/ directory found in {src}. "
            "Please run prepare_stem_checkpoint.py first."
        )
        sys.exit(1)

    # ---- Determine output directory ----
    if args.output_dir is not None:
        dst = Path(args.output_dir)
        if dst != src:
            logger.info(f"Copying {src} -> {dst}")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    else:
        dst = src
        logger.info(f"Writing in-place to {dst}")

    dst_stem_shards = dst / "stem_shards"

    # ---- Infer n_stem_layers and d_ff from existing shards ----
    shard_files = sorted(dst_stem_shards.glob("stem_model_mp*.pt"))
    if not shard_files:
        logger.error("No stem_model_mp*.pt files found in stem_shards/")
        sys.exit(1)

    sample_shard = torch.load(shard_files[0], map_location="cpu")
    n_stem_layers = sum(1 for k in sample_shard if k.startswith("stem_embeddings.") and k.endswith(".weight"))
    n_shards = len(shard_files)

    first_key = next(k for k in sample_shard if k.startswith("stem_embeddings.") and k.endswith(".weight"))
    shard_dim = sample_shard[first_key].shape[1]
    d_ff = shard_dim * n_shards

    logger.info(f"Detected {n_stem_layers} stem layers, d_ff={d_ff} "
                f"({n_shards} shards x {shard_dim})")

    # ---- Build IIR parameters ----
    iir_params = build_iir_params(
        n_stem_layers=n_stem_layers,
        d_ff=d_ff,
        mu_init=args.mu_init,
        alpha_init=args.alpha_init,
        per_dim_alpha=args.per_dim_alpha,
        learnable_m0=args.learnable_m0,
    )

    for k, v in iir_params.items():
        logger.info(f"  {k}: shape={tuple(v.shape)}, value={v.flatten()[:4].tolist()}")

    # ---- Save ----
    iir_path = dst_stem_shards / IIR_PARAMS_FILE
    torch.save(iir_params, iir_path)
    logger.info(f"Saved IIR parameters to {iir_path}")

    # ---- Update params.json ----
    params_json_path = dst / "params.json"
    if params_json_path.exists():
        with open(params_json_path, "r") as f:
            params_dict = json.load(f)
    else:
        params_dict = {}

    if "model" not in params_dict:
        params_dict["model"] = {}

    params_dict["model"]["iir_mu_init"] = args.mu_init
    params_dict["model"]["iir_alpha_init"] = args.alpha_init
    params_dict["model"]["iir_per_dim_alpha"] = args.per_dim_alpha
    params_dict["model"]["iir_learnable_m0"] = args.learnable_m0

    with open(params_json_path, "w") as f:
        json.dump(params_dict, f, indent=2)
    logger.info(f"Updated {params_json_path} with IIR config")

    # ---- Done ----
    logger.info("")
    logger.info(f"IIR-STEM checkpoint ready at: {dst}")
    logger.info(
        f"To use in stem_iir_train.py, set:  "
        f"checkpoint.init_ckpt_path={dst}"
    )
    logger.info(
        f"IIR config: mu_init={args.mu_init}, alpha_init={args.alpha_init}, "
        f"per_dim_alpha={args.per_dim_alpha}, learnable_m0={args.learnable_m0}"
    )


if __name__ == "__main__":
    main()
