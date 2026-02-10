# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Prepare STEM checkpoint by computing layerwise w3 × tok_embeddings.

For each stem layer *i*, computes:

    stem_embedding_table = tok_embeddings.weight @ w3_i.weight.T
    shape: (vocab_size, hidden_dim)

i.e., for every token *t*:

    stem_emb[t] = w3_i.weight @ tok_embeddings.weight[t]

This initialises STEM embeddings so that each token's embedding matches the
original FFN up-projection output, providing a "functionally equivalent"
starting point for STEM training.

The output directory is created with:
  - Symlinks to all DCP backbone files from the original pretrained checkpoint
  - A ``stem_shards/`` subdirectory with the pre-computed embeddings, sharded
    along the embedding dimension to match ``stem_parallel_size``.

The resulting directory can be used directly as ``checkpoint.init_ckpt_path``
in ``stem_train.py``.

Usage
-----
    python apps/main/prepare_stem_checkpoint.py \\
        --ckpt-path checkpoints/Llama-3.2-1B/distcp \\
        --output-dir checkpoints/Llama-3.2-1B-stem-init \\
        --stem-layers 1 3 5 7 9 11 13 15 \\
        --stem-parallel-size 8
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_consolidated_checkpoint(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load the consolidated ``.pth`` checkpoint.

    Expects ``<ckpt_path>/consolidated/consolidated.pth`` to exist.
    """
    consolidated_path = Path(ckpt_path) / "consolidated" / "consolidated.pth"
    if not consolidated_path.exists():
        raise FileNotFoundError(
            f"Consolidated checkpoint not found at {consolidated_path}. "
            f"Please run checkpoint consolidation first or provide a checkpoint "
            f"directory that contains consolidated/consolidated.pth."
        )
    logger.info(f"Loading consolidated checkpoint from {consolidated_path}")
    state_dict = torch.load(consolidated_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded {len(state_dict)} keys from checkpoint")
    return state_dict


# ---------------------------------------------------------------------------
# Stem-embedding computation
# ---------------------------------------------------------------------------

def compute_stem_embeddings(
    state_dict: Dict[str, torch.Tensor],
    stem_layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Compute ``tok_embeddings.weight @ w3.weight.T`` per stem layer.

    Returns a dict mapping *stem_index* → ``(vocab_size, hidden_dim)`` tensor
    (float32).
    """
    tok_emb_key = "model.tok_embeddings.weight"
    if tok_emb_key not in state_dict:
        raise KeyError(
            f"Key '{tok_emb_key}' not found in checkpoint. "
            f"Available keys (first 10): {list(state_dict.keys())[:10]}"
        )

    tok_emb = state_dict[tok_emb_key].float()  # (vocab_size, dim)
    vocab_size, dim = tok_emb.shape
    logger.info(f"Token embeddings: shape={tuple(tok_emb.shape)}, dtype=float32 (cast)")

    stem_weights: Dict[int, torch.Tensor] = {}
    for stem_idx, layer_idx in enumerate(stem_layers):
        w3_key = f"model.layers.{layer_idx}.feed_forward.w3.weight"
        if w3_key not in state_dict:
            available_layers = sorted(
                {
                    int(k.split(".")[2])
                    for k in state_dict
                    if k.startswith("model.layers.") and "feed_forward.w3" in k
                }
            )
            raise KeyError(
                f"Key '{w3_key}' not found in checkpoint. "
                f"Layers with w3: {available_layers}"
            )

        w3_weight = state_dict[w3_key].float()  # (hidden_dim, dim)
        hidden_dim = w3_weight.shape[0]

        # stem_table[token] = w3.weight @ tok_emb[token]
        # Vectorised: (vocab_size, dim) @ (dim, hidden_dim) = (vocab_size, hidden_dim)
        stem_weight = tok_emb @ w3_weight.T

        stem_weights[stem_idx] = stem_weight

        logger.info(
            f"  Layer {layer_idx:>2} (stem idx {stem_idx}): "
            f"w3 {tuple(w3_weight.shape)} × tok_emb {tuple(tok_emb.shape)} "
            f"→ stem {tuple(stem_weight.shape)}, "
            f"norm={stem_weight.norm():.4f}, "
            f"mean={stem_weight.mean():.6f}, std={stem_weight.std():.6f}"
        )

    return stem_weights


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_stem_shards(
    stem_weights: Dict[int, torch.Tensor],
    output_dir: Path,
    stem_parallel_size: int,
):
    """Save stem embeddings as sharded checkpoint files.

    Creates ``<output_dir>/stem_shards/stem_model_mp{rank}.pt`` for each MP
    rank.  The embedding dimension is split evenly across shards.
    """
    stem_dir = output_dir / "stem_shards"
    stem_dir.mkdir(parents=True, exist_ok=True)

    first_weight = next(iter(stem_weights.values()))
    vocab_size, hidden_dim = first_weight.shape

    assert hidden_dim % stem_parallel_size == 0, (
        f"hidden_dim ({hidden_dim}) must be divisible by "
        f"stem_parallel_size ({stem_parallel_size})"
    )
    shard_size = hidden_dim // stem_parallel_size

    for mp_rank in range(stem_parallel_size):
        shard_dict: Dict[str, torch.Tensor] = {}
        start = mp_rank * shard_size
        end = start + shard_size

        for stem_idx, full_weight in stem_weights.items():
            key = f"stem_embeddings.{stem_idx}.weight"
            # .clone() is critical: column-slicing creates a view that shares
            # the full underlying storage; without clone torch.save would
            # serialise the entire tensor for every shard.
            shard_dict[key] = full_weight[:, start:end].clone().cpu()

        shard_path = stem_dir / f"stem_model_mp{mp_rank}.pt"
        torch.save(shard_dict, shard_path)
        logger.info(
            f"  Saved stem shard mp_rank={mp_rank}: {shard_path} "
            f"(shape per embedding: ({vocab_size}, {shard_size}))"
        )


def symlink_backbone(src_dir: Path, dst_dir: Path):
    """Create symlinks in *dst_dir* pointing to every item in *src_dir*.

    Skips ``stem_shards`` (if it already exists in the source) and any
    destination paths that already exist.
    """
    for item in src_dir.iterdir():
        if item.name == "stem_shards":
            continue
        dst = dst_dir / item.name
        if not dst.exists():
            os.symlink(item.resolve(), dst)
            logger.info(f"  Symlinked {item.name} → {item.resolve()}")
        else:
            logger.info(f"  Skipped {item.name} (already exists)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare STEM checkpoint by computing layerwise "
            "w3 × tok_embeddings and saving as stem_shards."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help=(
            "Path to pretrained DCP checkpoint directory "
            "(must contain consolidated/consolidated.pth)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the new checkpoint with stem embeddings",
    )
    parser.add_argument(
        "--stem-layers",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 9, 11, 13, 15],
        help="Which transformer layers get stem embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--stem-parallel-size",
        type=int,
        default=8,
        help="Number of stem model-parallel shards (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ckpt_path = Path(args.ckpt_path)
    output_dir = Path(args.output_dir)

    if not ckpt_path.exists():
        logger.error(f"Checkpoint path does not exist: {ckpt_path}")
        sys.exit(1)

    # ---- 1. Load consolidated checkpoint ----
    state_dict = load_consolidated_checkpoint(str(ckpt_path))

    # ---- 2. Compute stem embeddings ----
    logger.info(f"Computing stem embeddings for layers: {args.stem_layers}")
    stem_weights = compute_stem_embeddings(state_dict, args.stem_layers)

    # Free memory – we no longer need the full checkpoint
    del state_dict

    # ---- 3. Create output directory ----
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4. Symlink backbone files from original checkpoint ----
    logger.info(f"Creating symlinks to backbone files from {ckpt_path}")
    symlink_backbone(ckpt_path, output_dir)

    # ---- 5. Save stem shards ----
    logger.info(
        f"Saving stem shards with stem_parallel_size={args.stem_parallel_size}"
    )
    save_stem_shards(stem_weights, output_dir, args.stem_parallel_size)

    # ---- Done ----
    logger.info("")
    logger.info(f"STEM checkpoint saved to: {output_dir}")
    logger.info(
        f"To use in stem_train.py, set:  checkpoint.init_ckpt_path={output_dir}"
    )
    logger.info(
        f"Stem layers: {args.stem_layers}  |  "
        f"Parallel size: {args.stem_parallel_size}  |  "
        f"Shards: {args.stem_parallel_size}"
    )


if __name__ == "__main__":
    main()

