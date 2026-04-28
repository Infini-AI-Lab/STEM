"""
Convert a HuggingFace OLMo3 checkpoint (model.safetensors or sharded) into
the lingua checkpoint format used by apps/main/olmo3.py.

Key differences from the Llama converter (convert_hf_to_lingua.py):
  - Maps OLMo3 post-norm keys:
      post_attention_layernorm / post_feedforward_layernorm
  - Maps q_norm / k_norm weights (OLMo3's full-projection QK-Norm)
  - Permutes q_norm / k_norm weights to match lingua's interleaved RoPE layout
  - Computes ffn_dim_multiplier from OLMo3's explicit intermediate_size

Output structure:
    <output_dir>/
        __0_0.distcp        # DCP backbone
        .metadata           # DCP metadata
        params.json         # Training / model config

Usage:
    python checkpoints/convert_hf_olmo3_to_lingua.py \
        --hf-ckpt-dir checkpoints/OLMo-3-1B-hf \
        --output-dir  checkpoints/OLMo-3-1B-lingua
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permutation helpers for RoPE format conversion
# ---------------------------------------------------------------------------

def permute_qk(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Convert Q/K projection weight from HuggingFace (rotate_half RoPE) format to
    lingua (interleaved matrix-multiply RoPE) format.
    """
    dim = weight.shape[1]
    return (
        weight.view(n_heads, 2, head_dim // 2, dim)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, dim)
    )


def permute_qk_norm_weight(
    weight: torch.Tensor, n_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Permute OLMo3 QK-Norm weight to match lingua's interleaved RoPE layout.

    OLMo3 applies QK-Norm before head reshape over the flattened projection
    dimension (n_heads * head_dim). Reordering therefore follows the same
    per-head permutation used for Q/K projection rows.
    """
    return (
        weight.view(n_heads, 2, head_dim // 2)
        .transpose(1, 2)
        .reshape(n_heads * head_dim)
    )


# ---------------------------------------------------------------------------
# HF → lingua key mapping (OLMo3-specific)
# ---------------------------------------------------------------------------

def _map_hf_key_to_lingua(hf_key: str) -> Optional[str]:
    """Map a single HuggingFace OLMo3 key to its lingua equivalent."""

    # --- Top-level ---
    if hf_key == "model.embed_tokens.weight":
        return "model.tok_embeddings.weight"
    if hf_key == "lm_head.weight":
        return "model.output.weight"
    if hf_key == "model.norm.weight":
        return "model.norm.weight"

    # --- Per-layer ---
    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".")
        layer_idx = int(parts[2])
        rest = ".".join(parts[3:])
        prefix = f"model.layers.{layer_idx}"

        # Attention projections
        attn_map = {
            "self_attn.q_proj.weight": "attention.wq.weight",
            "self_attn.k_proj.weight": "attention.wk.weight",
            "self_attn.v_proj.weight": "attention.wv.weight",
            "self_attn.o_proj.weight": "attention.wo.weight",
        }
        if rest in attn_map:
            return f"{prefix}.{attn_map[rest]}"

        # OLMo3-specific: QK-Norm over full projections
        qk_norm_map = {
            "self_attn.q_norm.weight": "attention.q_norm.weight",
            "self_attn.k_norm.weight": "attention.k_norm.weight",
        }
        if rest in qk_norm_map:
            return f"{prefix}.{qk_norm_map[rest]}"

        # OLMo3 post-norm names (plus alias forms)
        norm_map = {
            "post_attention_layernorm.weight": "post_attention_norm.weight",
            "post_feedforward_layernorm.weight": "post_feedforward_norm.weight",
            "post_attention_norm.weight": "post_attention_norm.weight",
            "post_feedforward_norm.weight": "post_feedforward_norm.weight",
        }
        if rest in norm_map:
            return f"{prefix}.{norm_map[rest]}"

        # MLP / Feed-forward
        mlp_map = {
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.up_proj.weight": "feed_forward.w3.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
        }
        if rest in mlp_map:
            return f"{prefix}.{mlp_map[rest]}"

    return None


def convert_hf_state_dict_to_lingua(
    hf_sd: Dict[str, torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    weight_tying: bool,
) -> Dict[str, torch.Tensor]:
    """
    Convert a HuggingFace OLMo3 state_dict to lingua format.

    Handles:
      - Key renaming (HF → lingua naming convention)
      - Q/K weight permutation (rotate_half → interleaved RoPE layout)
      - Q/K norm weight permutation (same feature reordering as Q/K rows)
      - Weight tying (output → output.tied_module)
    """
    backbone_sd: Dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        lingua_key = _map_hf_key_to_lingua(hf_key)
        if lingua_key is None:
            logger.warning(f"Skipping unmapped HF key: {hf_key}")
            continue

        # Apply Q/K projection permutation for RoPE format difference
        if lingua_key.endswith(".attention.wq.weight"):
            tensor = permute_qk(tensor, n_heads, head_dim)
        elif lingua_key.endswith(".attention.wk.weight"):
            tensor = permute_qk(tensor, n_kv_heads, head_dim)

        # Apply QK-Norm permutation to match reordered feature layout
        if lingua_key.endswith(".attention.q_norm.weight"):
            tensor = permute_qk_norm_weight(tensor, n_heads, head_dim)
        elif lingua_key.endswith(".attention.k_norm.weight"):
            tensor = permute_qk_norm_weight(tensor, n_kv_heads, head_dim)

        # Handle weight tying: output → output.tied_module
        if weight_tying and lingua_key == "model.output.weight":
            lingua_key = "model.output.tied_module.weight"

        backbone_sd[lingua_key] = tensor

    tied_key = "model.output.tied_module.weight"
    emb_key = "model.tok_embeddings.weight"
    if weight_tying and tied_key not in backbone_sd and emb_key in backbone_sd:
        backbone_sd[tied_key] = backbone_sd[emb_key]

    return backbone_sd


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_backbone_dcp(backbone_sd: Dict[str, torch.Tensor], output_dir: Path):
    """Save backbone state_dict in DCP format (creates __0_0.distcp + .metadata)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29513")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving backbone DCP to {output_dir} ({len(backbone_sd)} keys)")
    dcp.save(backbone_sd, checkpoint_id=str(output_dir))
    logger.info("DCP save complete")


def compute_ffn_dim_multiplier(
    dim: int, intermediate_size: Optional[int], multiple_of: int = 256
) -> Optional[float]:
    """
    Compute ffn_dim_multiplier that reproduces the given intermediate_size under
    lingua FeedForward formula.

    Formula:
      base = int(2 * 4 * dim / 3)
      hidden = multiple_of * ceil(int(m * base) / multiple_of)
    """
    if intermediate_size is None:
        return None

    base = int(2 * 4 * dim / 3)
    default_hidden = multiple_of * ((base + multiple_of - 1) // multiple_of)
    if default_hidden == intermediate_size:
        return None

    # Start with exact ratio and verify
    m = intermediate_size / base
    computed = int(m * base)
    computed = multiple_of * ((computed + multiple_of - 1) // multiple_of)
    if computed == intermediate_size:
        return round(m, 10)

    # If exact ratio did not work due to boundary effects, try tiny deltas
    for delta in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4]:
        m_try = intermediate_size / base + delta
        computed = int(m_try * base)
        computed = multiple_of * ((computed + multiple_of - 1) // multiple_of)
        if computed == intermediate_size:
            return round(m_try, 10)

    logger.warning(
        f"Could not find exact ffn_dim_multiplier for intermediate_size={intermediate_size} "
        f"(dim={dim}, multiple_of={multiple_of}). Using raw ratio."
    )
    return round(m, 10)


def create_params_json(
    output_dir: Path,
    reference_params: Optional[str],
    hf_config: dict,
    weight_tying: bool,
    hf_ckpt_dir: str = "",
):
    """
    Create params.json for the output checkpoint.
    """
    if reference_params and Path(reference_params).exists():
        with open(reference_params) as f:
            params = json.load(f)
        logger.info(f"Loaded reference params from {reference_params}")
    else:
        # Minimal skeleton matching lingua format
        params = {
            "name": "large_lm",
            "dump_dir": "logs/debug",
            "seed": 777,
            "grad_acc_steps": 1,
            "gc_collect_freq": 1000,
            "probe_freq": None,
            "steps": 60000,
            "data": {
                "root_dir": "data/dolmino-mix-subset_shuffled",
                "sources": {"ingredient1-common_crawl-high_quality": 1.0},
                "batch_size": 4,
                "seq_len": 4096,
                "n_views": 2,
                "seed": 42,
                "add_bos": True,
                "add_eos": True,
                "load_async": True,
                "prefetch_size": 1024,
                "tokenizer": {
                    "name": "huggingface",
                    "path": "",  # patched below from --hf-ckpt-dir
                },
            },
            "optim": {
                "lr": 2e-05,
                "weight_decay": 0.033,
                "epsilon": 1e-08,
                "beta1": 0.9,
                "beta2": 0.95,
                "clip": 1.0,
                "scheduler": "cosine",
                "warmup": 2000,
                "lr_min_ratio": 1e-06,
                "cycle_length": 1.0,
                "cosine_theta": 1.0,
                "annealing_step": 1000,
                "decay_fraction": 0.1,
                "exp_factor": 0.5,
            },
            "distributed": {
                "dp_shard": 1,
                "dp_replicate": 1,
                "tp_size": 1,
                "selective_activation_checkpointing": False,
                "compile": True,
                "fsdp_type": "full_shard",
                "model_dtype": "bf16",
                "float8_recipe": None,
                "float8_filter": "layers\\.[0-9]+\\.",
                "matmul_allow_tf32": False,
                "detect_anomaly": False,
                "compile_cache_size_limit": 8,
                "spawn_method": "forkserver",
            },
            "checkpoint": {
                "dump": {"every": 20, "keep": 1},
                "eval": {"every": 200, "keep": 1},
                "path": "logs/debug/checkpoints",
                "init_ckpt_path": "",
                "continue_training_from_init": False,
            },
            "logging": {"freq": 10, "acc_freq": None, "wandb": None},
        }

    # ---- Compute FFN configuration from HF intermediate size ----
    dim = hf_config["hidden_size"]
    intermediate_size = hf_config.get("intermediate_size")
    multiple_of = 256
    ffn_dim_multiplier = compute_ffn_dim_multiplier(dim, intermediate_size, multiple_of)

    # ---- Patch model section from HF config ----
    params["model"] = {
        "dim": dim,
        "n_layers": hf_config["num_hidden_layers"],
        "head_dim": hf_config.get("head_dim", dim // hf_config["num_attention_heads"]),
        "n_heads": hf_config["num_attention_heads"],
        "n_kv_heads": hf_config.get(
            "num_key_value_heads", hf_config["num_attention_heads"]
        ),
        "ffn_dim_multiplier": ffn_dim_multiplier,
        "multiple_of": multiple_of,
        "norm_eps": hf_config.get("rms_norm_eps", 1e-5),
        "rope_theta": hf_config.get("rope_theta", 10000.0),
        "rope_scaling": hf_config.get("rope_scaling", None),
        "init_base_std": 0.02,
        "init_std_factor": "disabled",
        "max_seqlen": hf_config.get("max_position_embeddings", 8192),
        "seed": 42,
        "vocab_size": hf_config["vocab_size"],
        "weight_tying": weight_tying,
        "sliding_window": hf_config.get("sliding_window", None),
    }

    # ---- Patch tokenizer to use huggingface tokenizer from HF checkpoint ----
    if "data" in params and "tokenizer" in params["data"]:
        params["data"]["tokenizer"]["name"] = "huggingface"
        params["data"]["tokenizer"]["path"] = str(hf_ckpt_dir) if hf_ckpt_dir else ""

    params_path = output_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved params.json to {params_path}")


def load_hf_checkpoint(hf_dir: Path) -> Dict[str, torch.Tensor]:
    """Load all supported HF checkpoint layouts into one state_dict."""
    hf_sd: Dict[str, torch.Tensor] = {}

    # Try sharded safetensors first
    shard_files = sorted(hf_dir.glob("model-*.safetensors"))
    if shard_files:
        from safetensors.torch import load_file
        for sf in shard_files:
            logger.info(f"  Loading {sf.name}...")
            hf_sd.update(load_file(str(sf)))
        return hf_sd

    # Try single-file variants
    single_st = hf_dir / "model.safetensors"
    single_bin = hf_dir / "pytorch_model.bin"
    if single_st.exists():
        from safetensors.torch import load_file
        logger.info(f"  Loading {single_st.name}...")
        hf_sd.update(load_file(str(single_st)))
        return hf_sd
    if single_bin.exists():
        logger.info(f"  Loading {single_bin.name}...")
        hf_sd.update(torch.load(str(single_bin), map_location="cpu", weights_only=True))
        return hf_sd

    # Try sharded pytorch_model
    pt_shards = sorted(hf_dir.glob("pytorch_model-*.bin"))
    if pt_shards:
        for sf in pt_shards:
            logger.info(f"  Loading {sf.name}...")
            hf_sd.update(torch.load(str(sf), map_location="cpu", weights_only=True))
        return hf_sd

    logger.error(
        f"No checkpoint files found in {hf_dir}. "
        "Expected model.safetensors, model-*.safetensors, "
        "pytorch_model.bin, or pytorch_model-*.bin."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace OLMo3 checkpoint to lingua format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-ckpt-dir",
        required=True,
        help="Path to HF OLMo3 checkpoint directory (contains config.json + weights)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the lingua checkpoint",
    )
    parser.add_argument(
        "--reference-params",
        default=None,
        help="Optional reference params.json to use as base config",
    )
    parser.add_argument(
        "--weight-tying",
        action="store_true",
        default=None,
        help="Force weight tying (output shares tok_embeddings). "
             "Default: auto-detect from HF config.",
    )
    parser.add_argument(
        "--no-weight-tying",
        action="store_true",
        default=None,
        help="Force no weight tying. Default: auto-detect from HF config.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    hf_dir = Path(args.hf_ckpt_dir)
    output_dir = Path(args.output_dir)

    # ---- 1. Load HF config ----
    config_path = hf_dir / "config.json"
    if not config_path.exists():
        logger.error(f"config.json not found in {hf_dir}")
        sys.exit(1)
    with open(config_path) as f:
        hf_config = json.load(f)

    n_heads = hf_config["num_attention_heads"]
    n_kv_heads = hf_config.get("num_key_value_heads", n_heads)
    head_dim = hf_config.get("head_dim", hf_config["hidden_size"] // n_heads)

    # Determine weight tying
    if args.weight_tying:
        weight_tying = True
    elif args.no_weight_tying:
        weight_tying = False
    else:
        weight_tying = (
            hf_config.get("tie_word_embeddings", False)
            or hf_config.get("share_embeddings", False)
        )

    logger.info(
        f"OLMo3 config: dim={hf_config['hidden_size']}, "
        f"n_layers={hf_config['num_hidden_layers']}, "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, "
        f"intermediate_size={hf_config.get('intermediate_size')}"
    )
    logger.info(f"Weight tying: {weight_tying}")

    # ---- 2. Load HF checkpoint ----
    logger.info("Loading HF checkpoint...")
    hf_sd = load_hf_checkpoint(hf_dir)
    logger.info(f"Loaded {len(hf_sd)} keys total")

    # ---- 3. Convert keys ----
    logger.info("Converting HF OLMo3 keys to lingua format...")
    backbone_sd = convert_hf_state_dict_to_lingua(
        hf_sd, n_heads, n_kv_heads, head_dim, weight_tying
    )
    del hf_sd  # free memory

    logger.info(f"Backbone: {len(backbone_sd)} keys")
    for k, v in sorted(backbone_sd.items())[:10]:
        logger.info(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    qk_norm_keys = [k for k in backbone_sd if "q_norm" in k or "k_norm" in k]
    logger.info(
        f"QK-Norm keys found: {len(qk_norm_keys)} "
        f"(expected about {hf_config['num_hidden_layers'] * 2})"
    )

    # ---- 4. Save backbone as DCP ----
    save_backbone_dcp(backbone_sd, output_dir)
    del backbone_sd

    # ---- 5. Create params.json ----
    create_params_json(
        output_dir,
        args.reference_params,
        hf_config,
        weight_tying,
        hf_ckpt_dir=str(hf_dir),
    )

    if dist.is_initialized():
        dist.destroy_process_group()

    logger.info("")
    logger.info(f"Lingua checkpoint saved to: {output_dir}")
    logger.info(f"  Backbone: {output_dir}/__0_0.distcp")
    logger.info(f"  Config:   {output_dir}/params.json")


if __name__ == "__main__":
    main()

