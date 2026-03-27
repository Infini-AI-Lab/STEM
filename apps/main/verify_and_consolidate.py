#!/usr/bin/env python3
"""
STEM Checkpoint Verification and Proper Consolidation

This script does three things:
  1. Verifies STEM shard integrity (shapes, norms, ordering)
  2. Consolidates STEM shards with CORRECT sorted ordering
  3. Verifies the consolidated checkpoint loads correctly

Usage:
  python apps/main/verify_and_consolidate.py \
      --stem-ckpt /path/to/stem/checkpoint/0000200000 \
      --vanilla-ckpt /path/to/vanilla/Llama-3.2-1B/distcp \
      --stem-parallel-size 8

Run this BEFORE running stem_diagnostics.py.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Constants (from lingua/stem_checkpoint.py)
# =============================================================================
CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"
CONSOLIDATE_STEM_NAME = "consolidated_stem.pth"
CONFIG_NAME = "params.json"
STEM_SUBDIR_NAME = "stem_shards"


# =============================================================================
# Step 1: Verify STEM Shard Integrity
# =============================================================================
def verify_stem_shards(ckpt_dir: Path, expected_parallel_size: int) -> bool:
    """Inspect all stem_model_mp*.pt shards for shape/norm consistency."""
    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    if not stem_dir.exists():
        logger.error(f"stem_shards/ directory not found at {stem_dir}")
        return False

    shard_files = sorted(stem_dir.glob("stem_model_mp*.pt"))
    n_shards = len(shard_files)
    logger.info(f"Found {n_shards} shard files in {stem_dir}")

    if n_shards == 0:
        logger.error("No stem_model_mp*.pt files found!")
        return False

    if n_shards != expected_parallel_size:
        logger.warning(
            f"Expected {expected_parallel_size} shards (stem_parallel_size), "
            f"found {n_shards}. This may indicate a save/load mismatch."
        )

    # Load and inspect each shard
    all_shapes = {}  # key -> list of shapes across shards
    all_norms = {}   # key -> list of norms across shards
    all_means = {}   # key -> list of means across shards

    for shard_file in shard_files:
        sd = torch.load(shard_file, map_location="cpu", weights_only=True)
        mp_rank = shard_file.stem.split("mp")[-1]  # e.g., "0" from "stem_model_mp0"
        logger.info(f"\n--- {shard_file.name} (mp_rank={mp_rank}) ---")

        for key, tensor in sd.items():
            logger.info(
                f"  {key}: shape={list(tensor.shape)}, "
                f"dtype={tensor.dtype}, "
                f"norm={tensor.norm().item():.4f}, "
                f"mean={tensor.mean().item():.6f}, "
                f"std={tensor.std().item():.6f}, "
                f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}"
            )
            all_shapes.setdefault(key, []).append(tuple(tensor.shape))
            all_norms.setdefault(key, []).append(tensor.norm().item())
            all_means.setdefault(key, []).append(tensor.mean().item())

    # Validate: all shards should have the same keys and shapes
    ok = True
    for key in all_shapes:
        shapes = all_shapes[key]
        if len(set(shapes)) > 1:
            logger.error(f"Shape mismatch across shards for '{key}': {shapes}")
            ok = False
        else:
            V, D_shard = shapes[0]
            D_full = D_shard * n_shards
            logger.info(f"\n  '{key}': per-shard shape ({V}, {D_shard}), "
                        f"reconstructed full shape ({V}, {D_full})")

        # Check norm consistency: large variance suggests shards from different steps
        norms = all_norms[key]
        norm_std = torch.tensor(norms).std().item()
        norm_mean = torch.tensor(norms).mean().item()
        norm_cv = norm_std / norm_mean if norm_mean > 0 else 0
        logger.info(f"  Norms across shards: mean={norm_mean:.4f}, "
                    f"std={norm_std:.4f}, CV={norm_cv:.4f}")
        if norm_cv > 0.3:
            logger.warning(
                f"  HIGH norm variance (CV={norm_cv:.4f}) for '{key}' — "
                f"shards may be from different training steps!"
            )

        # Check for all-zero shards
        for i, n in enumerate(norms):
            if n < 1e-6:
                logger.error(f"  Shard {i} for '{key}' is all-zeros!")
                ok = False

    return ok


# =============================================================================
# Step 2: Properly Consolidate STEM Shards (SORTED order)
# =============================================================================
def consolidate_stem_shards_sorted(ckpt_dir: Path, force: bool = False) -> Path:
    """Consolidate STEM shards with explicit sorted ordering.

    The upstream consolidate_stem_shards() uses glob() which does NOT
    guarantee sorted order. This function explicitly sorts by mp_rank
    index to ensure correct dimension concatenation.
    """
    consolidate_path = ckpt_dir / CONSOLIDATE_FOLDER
    output_file = consolidate_path / CONSOLIDATE_STEM_NAME

    if output_file.exists() and not force:
        logger.info(f"Consolidated STEM file already exists at {output_file}")
        logger.info("Use --force to reconsolidate.")
        return consolidate_path

    stem_dir = ckpt_dir / STEM_SUBDIR_NAME
    if not stem_dir.exists():
        raise FileNotFoundError(f"No stem_shards/ directory at {stem_dir}")

    # Get all shard files and sort by mp_rank index
    shard_files = list(stem_dir.glob("stem_model_mp*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No stem_model_mp*.pt files in {stem_dir}")

    def _extract_rank(path: Path) -> int:
        # stem_model_mp0.pt -> 0
        return int(path.stem.split("mp")[-1])

    shard_files.sort(key=_extract_rank)
    logger.info(f"Consolidating {len(shard_files)} shards in order: "
                f"{[f.name for f in shard_files]}")

    # Load all shards in sorted order
    consolidated = {}
    for shard_file in shard_files:
        sd = torch.load(shard_file, map_location="cpu", weights_only=True)
        rank = _extract_rank(shard_file)
        for key, tensor in sd.items():
            consolidated.setdefault(key, []).append((rank, tensor))

    # Verify ordering and concatenate
    result = {}
    for key, rank_tensor_pairs in consolidated.items():
        # Sort by rank (should already be sorted, but be safe)
        rank_tensor_pairs.sort(key=lambda x: x[0])
        ranks = [r for r, _ in rank_tensor_pairs]
        tensors = [t for _, t in rank_tensor_pairs]

        expected_ranks = list(range(len(ranks)))
        if ranks != expected_ranks:
            logger.error(f"Rank gap for '{key}': found ranks {ranks}, "
                         f"expected {expected_ranks}")
            raise ValueError(f"Missing shards for '{key}'")

        result[key] = torch.cat(tensors, dim=1)
        logger.info(f"  {key}: concatenated {len(tensors)} shards -> "
                    f"shape {list(result[key].shape)}")

    # Save
    consolidate_path.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_file)
    logger.info(f"Saved consolidated STEM weights to {output_file}")
    return consolidate_path


# =============================================================================
# Step 3: Verify Backbone Consolidation
# =============================================================================
def verify_backbone(ckpt_dir: Path) -> bool:
    """Check that the backbone consolidated.pth exists and has expected keys."""
    consolidate_path = ckpt_dir / CONSOLIDATE_FOLDER
    backbone_file = consolidate_path / CONSOLIDATE_NAME

    if not backbone_file.exists():
        logger.info("Backbone consolidated.pth not found. Attempting DCP consolidation...")
        # Check for DCP metadata
        metadata = ckpt_dir / ".metadata"
        if not metadata.exists():
            logger.error(f"No .metadata file at {ckpt_dir} — cannot consolidate backbone")
            return False

        try:
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
            consolidate_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Running dcp_to_torch_save: {ckpt_dir} -> {backbone_file}")
            dcp_to_torch_save(str(ckpt_dir), str(backbone_file))
            # Copy params.json
            params_src = ckpt_dir / CONFIG_NAME
            params_dst = consolidate_path / CONFIG_NAME
            if params_src.exists() and not params_dst.exists():
                params_dst.write_text(params_src.read_text())
            logger.info("Backbone consolidation complete")
        except Exception as e:
            logger.error(f"Backbone consolidation failed: {e}")
            return False

    # Verify params.json
    params_file = consolidate_path / CONFIG_NAME
    if not params_file.exists():
        logger.error(f"params.json not found at {params_file}")
        return False

    # Load and inspect backbone
    logger.info(f"\nInspecting backbone at {backbone_file}")
    sd = torch.load(backbone_file, map_location="cpu", weights_only=True)
    if "model" in sd:
        sd = sd["model"]

    # Check for key patterns
    first_key = next(iter(sd.keys()))
    prefix = ""
    if first_key.startswith("model."):
        prefix = "model."
    elif first_key.startswith("lm_transformer."):
        prefix = "lm_transformer."

    tok_emb_key = f"{prefix}tok_embeddings.weight"
    if tok_emb_key not in sd:
        logger.error(f"tok_embeddings.weight not found (tried key '{tok_emb_key}')")
        logger.info(f"Available keys (first 10): {list(sd.keys())[:10]}")
        return False

    tok_shape = sd[tok_emb_key].shape
    logger.info(f"  tok_embeddings: shape={list(tok_shape)}")

    # Check for w3 keys (should exist for non-STEM layers, absent for STEM layers)
    w3_keys = [k for k in sd if ".w3." in k or k.endswith(".w3.weight")]
    w1_keys = [k for k in sd if ".w1." in k or k.endswith(".w1.weight")]
    logger.info(f"  FFN w1 weight keys: {len(w1_keys)}")
    logger.info(f"  FFN w3 weight keys: {len(w3_keys)} (should be 0 for STEM layers, present for non-STEM)")

    # Check STEM config
    from omegaconf import OmegaConf
    config = OmegaConf.load(params_file)
    stem_layers = config.model.get("stem_layers", None)
    logger.info(f"  stem_layers from config: {stem_layers}")
    logger.info(f"  model dim: {config.model.dim}")
    logger.info(f"  n_layers: {config.model.n_layers}")

    n_layers = config.model.n_layers
    logger.info(f"\n  Layer inventory:")
    for i in range(n_layers):
        w1_found = any(f"layers.{i}.feed_forward.w1" in k for k in sd)
        w3_found = any(f"layers.{i}.feed_forward.w3" in k for k in sd)
        is_stem = stem_layers is not None and i in stem_layers
        expected_w3 = not is_stem
        status = "OK" if (w3_found == expected_w3) else "MISMATCH"
        layer_type = "STEM" if is_stem else "FFN"
        logger.info(f"    layer {i:2d} [{layer_type}]: w1={w1_found}, w3={w3_found} [{status}]")
        if status == "MISMATCH":
            return False

    return True


# =============================================================================
# Step 4: Verify Vanilla Checkpoint
# =============================================================================
def verify_vanilla(vanilla_dir: Path) -> bool:
    """Verify vanilla checkpoint has expected structure."""
    # Try consolidated path
    consolidated = vanilla_dir / CONSOLIDATE_FOLDER
    if (consolidated / CONSOLIDATE_NAME).exists():
        ckpt_file = consolidated / CONSOLIDATE_NAME
        params_file = consolidated / CONFIG_NAME
    elif (vanilla_dir / CONSOLIDATE_NAME).exists():
        ckpt_file = vanilla_dir / CONSOLIDATE_NAME
        params_file = vanilla_dir / CONFIG_NAME
    else:
        # Try DCP consolidation
        if (vanilla_dir / ".metadata").exists():
            logger.info("Vanilla checkpoint needs DCP consolidation...")
            try:
                from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
                consolidated.mkdir(parents=True, exist_ok=True)
                out_file = consolidated / CONSOLIDATE_NAME
                dcp_to_torch_save(str(vanilla_dir), str(out_file))
                params_src = vanilla_dir / CONFIG_NAME
                params_dst = consolidated / CONFIG_NAME
                if params_src.exists():
                    params_dst.write_text(params_src.read_text())
                ckpt_file = out_file
                params_file = params_dst
                logger.info("Vanilla DCP consolidation complete")
            except Exception as e:
                logger.error(f"Vanilla consolidation failed: {e}")
                return False
        else:
            logger.error(f"No loadable checkpoint found at {vanilla_dir}")
            return False

    if not params_file.exists():
        logger.error(f"Vanilla params.json not found at {params_file}")
        return False

    sd = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    if "model" in sd:
        sd = sd["model"]

    # Verify all layers have w3
    w3_keys = sorted([k for k in sd if "w3.weight" in k])
    logger.info(f"\nVanilla checkpoint: {ckpt_file}")
    logger.info(f"  Total keys: {len(sd)}")
    logger.info(f"  w3 weight keys: {len(w3_keys)}")
    for k in w3_keys:
        logger.info(f"    {k}: shape={list(sd[k].shape)}")

    tok_key = next((k for k in sd if "tok_embeddings.weight" in k), None)
    if tok_key:
        logger.info(f"  tok_embeddings: shape={list(sd[tok_key].shape)}")

    return True


# =============================================================================
# Step 5: End-to-End Consolidated Load Test
# =============================================================================
def test_consolidated_load(ckpt_dir: Path) -> bool:
    """Test that the consolidated STEM weights match expected shapes."""
    consolidate_path = ckpt_dir / CONSOLIDATE_FOLDER
    stem_file = consolidate_path / CONSOLIDATE_STEM_NAME

    if not stem_file.exists():
        logger.error(f"Consolidated STEM file not found at {stem_file}")
        return False

    sd = torch.load(stem_file, map_location="cpu", weights_only=True)
    logger.info(f"\nConsolidated STEM weights ({stem_file}):")

    from omegaconf import OmegaConf
    params_file = consolidate_path / CONFIG_NAME
    if params_file.exists():
        config = OmegaConf.load(params_file)
        stem_layers = config.model.get("stem_layers", [])
        n_stem = len(stem_layers)
        logger.info(f"  Expected {n_stem} stem embedding tables")

    for key, tensor in sd.items():
        V, D = tensor.shape
        is_zero = (tensor.abs().max() < 1e-10).item()
        logger.info(
            f"  {key}: shape=({V}, {D}), "
            f"norm={tensor.norm().item():.4f}, "
            f"mean={tensor.mean().item():.6f}, "
            f"all_zeros={is_zero}"
        )
        if is_zero:
            logger.error(f"  CRITICAL: {key} is all zeros!")
            return False

    logger.info(f"  Total embedding tables: {len(sd)}")
    return True


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Verify and consolidate STEM checkpoints"
    )
    parser.add_argument("--stem-ckpt", type=str, required=True,
                        help="Path to STEM checkpoint step directory")
    parser.add_argument("--vanilla-ckpt", type=str, default=None,
                        help="Path to vanilla checkpoint directory")
    parser.add_argument("--stem-parallel-size", type=int, default=8,
                        help="Expected stem_parallel_size used during training")
    parser.add_argument("--force", action="store_true",
                        help="Force reconsolidation even if consolidated file exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    stem_dir = Path(args.stem_ckpt)
    all_ok = True

    # Step 1: Verify shard integrity
    logger.info("=" * 70)
    logger.info("STEP 1: Verifying STEM shard integrity")
    logger.info("=" * 70)
    if not verify_stem_shards(stem_dir, args.stem_parallel_size):
        logger.error("STEM shard verification FAILED")
        all_ok = False
    else:
        logger.info("STEM shard verification PASSED")

    # Step 2: Consolidate backbone
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Verifying/consolidating backbone")
    logger.info("=" * 70)
    if not verify_backbone(stem_dir):
        logger.error("Backbone verification FAILED")
        all_ok = False
    else:
        logger.info("Backbone verification PASSED")

    # Step 3: Consolidate STEM shards (sorted)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Consolidating STEM shards (sorted order)")
    logger.info("=" * 70)
    try:
        consolidate_stem_shards_sorted(stem_dir, force=args.force)
        logger.info("STEM consolidation PASSED")
    except Exception as e:
        logger.error(f"STEM consolidation FAILED: {e}")
        all_ok = False

    # Step 4: Test consolidated load
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Testing consolidated STEM load")
    logger.info("=" * 70)
    if not test_consolidated_load(stem_dir):
        logger.error("Consolidated load test FAILED")
        all_ok = False
    else:
        logger.info("Consolidated load test PASSED")

    # Step 5: Verify vanilla checkpoint
    if args.vanilla_ckpt:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Verifying vanilla checkpoint")
        logger.info("=" * 70)
        vanilla_dir = Path(args.vanilla_ckpt)
        if not verify_vanilla(vanilla_dir):
            logger.error("Vanilla checkpoint verification FAILED")
            all_ok = False
        else:
            logger.info("Vanilla checkpoint verification PASSED")

    # Summary
    logger.info("\n" + "=" * 70)
    if all_ok:
        logger.info("ALL CHECKS PASSED — checkpoints are ready for diagnostics")
        logger.info("\nRun diagnostics with:")
        logger.info(f"  python -m apps.main.stem_diagnostics \\")
        logger.info(f"      config=apps/main/configs/stem_diagnostics.yaml \\")
        logger.info(f"      stem_ckpt_dir={args.stem_ckpt} \\")
        if args.vanilla_ckpt:
            logger.info(f"      vanilla_ckpt_dir={args.vanilla_ckpt}")
    else:
        logger.error("SOME CHECKS FAILED — review the output above before running diagnostics")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()