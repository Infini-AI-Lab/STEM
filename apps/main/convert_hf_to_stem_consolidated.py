# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Convert a Hugging Face Llama-style checkpoint (safetensors, single pytorch_model.bin,
# or sharded pytorch_model-XXXX-of-YYYY.bin + pytorch_model.bin.index.json)
# into the STEM / Lingua consolidated layout expected by:
#   - apps.main.stem_generate.load_consolidated_model_and_tokenizer
#   - apps.main.generate.load_consolidated_model_and_tokenizer
#   - lingua.checkpoint.consolidate_checkpoints (downstream DCP flows)
#
# Output layout (under --output-dir):
#   params.json              # copied from --params-json (must match architecture)
#   consolidated/
#     consolidated.pth       # flat state dict with keys model.tok_embeddings.weight, ...
#
# This script does NOT create STEM embedding tables. After conversion, run:
#   python -m apps.main.prepare_stem_checkpoint --ckpt-path <output-dir> ...
# to produce stem_shards/ and a STEM-trainable DCP tree if you need STEM training.
#
# Vanilla inference / stem_eval on an already-STEM-trained HF release still requires
# consolidated_stem.pth or stem_shards/ from the publisher or from prepare_stem_checkpoint.

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HF weight loading (no full model init — memory safe)
# ---------------------------------------------------------------------------


def _load_safetensors_dir(model_dir: Path) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(
            "Loading .safetensors requires the `safetensors` package. "
            "Install with: pip install safetensors"
        ) from e

    index = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"

    if index.exists():
        with open(index, "r", encoding="utf-8") as f:
            weight_map: Dict[str, str] = json.load(f)["weight_map"]
        files_to_keys: Dict[str, List[str]] = defaultdict(list)
        for key, fname in weight_map.items():
            files_to_keys[fname].append(key)
        state: Dict[str, torch.Tensor] = {}
        for fname, keys in sorted(files_to_keys.items()):
            shard_path = model_dir / fname
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard file: {shard_path}")
            partial = load_file(str(shard_path))
            for k in keys:
                if k not in partial:
                    raise KeyError(f"Key {k!r} missing from {shard_path}")
                state[k] = partial[k]
        return state

    if single.exists():
        return dict(load_file(str(single)))

    raise FileNotFoundError(
        f"No model.safetensors or model.safetensors.index.json under {model_dir}"
    )


PYTORCH_MODEL_INDEX = "pytorch_model.bin.index.json"


def _load_sharded_pytorch_bins(model_dir: Path) -> Dict[str, torch.Tensor]:
    """Load HF sharded ``pytorch_model-XXXXX-of-YYYYY.bin`` via ``pytorch_model.bin.index.json``."""
    index_path = model_dir / PYTORCH_MODEL_INDEX
    if not index_path.exists():
        raise FileNotFoundError(f"No {PYTORCH_MODEL_INDEX} in {model_dir}")

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    if "weight_map" not in index_data:
        raise ValueError(f"{index_path} has no 'weight_map' field")

    weight_map: Dict[str, str] = index_data["weight_map"]
    files_to_keys: Dict[str, List[str]] = defaultdict(list)
    for tensor_key, fname in weight_map.items():
        files_to_keys[fname].append(tensor_key)

    state: Dict[str, torch.Tensor] = {}
    for fname in sorted(files_to_keys.keys()):
        shard_path = model_dir / fname
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Missing PyTorch shard referenced by {PYTORCH_MODEL_INDEX}: {shard_path}"
            )
        partial = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(partial, dict):
            raise TypeError(f"Expected dict in {shard_path}, got {type(partial)}")
        keys = files_to_keys[fname]
        for k in keys:
            if k not in partial:
                raise KeyError(
                    f"Tensor {k!r} listed in {PYTORCH_MODEL_INDEX} but missing from {shard_path}"
                )
            state[k] = partial[k]

    logger.info(
        "Loaded sharded PyTorch checkpoint: %d tensors from %d shard file(s)",
        len(state),
        len(files_to_keys),
    )
    return state


def _load_pytorch_bin(model_dir: Path) -> Dict[str, torch.Tensor]:
    """Load a single ``pytorch_model.bin`` or merge shards using ``pytorch_model.bin.index.json``."""
    if (model_dir / PYTORCH_MODEL_INDEX).exists():
        return _load_sharded_pytorch_bins(model_dir)

    candidates = sorted(model_dir.glob("pytorch_model*.bin"))
    if not candidates:
        raise FileNotFoundError(
            f"No pytorch_model*.bin found under {model_dir}. "
            f"Use a directory with model.safetensors, pytorch_model.bin, or "
            f"sharded bins plus {PYTORCH_MODEL_INDEX}."
        )

    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Found {len(candidates)} pytorch_model*.bin files under {model_dir} but "
            f"{PYTORCH_MODEL_INDEX} is missing; cannot merge shards. "
            f"Restore the index file or merge shards into a single pytorch_model.bin."
        )

    path = candidates[0]
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    logger.info("Loaded single-file PyTorch checkpoint from %s (%d tensors)", path.name, len(obj))
    return obj


def load_hf_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
    """Load raw Hugging Face LlamaForCausalLM-style tensors from disk."""
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)

    try:
        return _load_safetensors_dir(model_dir)
    except FileNotFoundError:
        pass

    return _load_pytorch_bin(model_dir)


def read_hf_llama_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _guess_dtype_from_state_dict(stem_sd: Dict[str, torch.Tensor]) -> str:
    for v in stem_sd.values():
        if torch.is_tensor(v) and v.is_floating_point():
            if v.dtype == torch.bfloat16:
                return "bf16"
            if v.dtype == torch.float16:
                return "fp16"
            return "fp32"
    return "bf16"


def _stem_ffn_hidden_dim(
    dim: int, ffn_dim_multiplier: Optional[float], multiple_of: int
) -> int:
    """Match ``lingua.stem.FeedForward`` / ``lingua.transformer.FeedForward`` rounding."""
    hidden_dim = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def infer_ffn_dim_multiplier(
    dim: int, intermediate_size: int, multiple_of: int = 256
) -> Optional[float]:
    """Find ``ffn_dim_multiplier`` such that STEM FFN hidden dim equals HF ``intermediate_size``."""
    base = int(8 * dim / 3)
    for mult in (None, 1.0, 4.0 / 3.0, 1.3, 1.5, 2.0, 8.0 / 3.0):
        if _stem_ffn_hidden_dim(dim, mult, multiple_of) == intermediate_size:
            return mult
    best_m: Optional[float] = None
    best_err = intermediate_size + 1
    for num in range(1, 200_001):
        mult = num / 10_000.0
        h = _stem_ffn_hidden_dim(dim, mult, multiple_of)
        err = abs(h - intermediate_size)
        if err < best_err:
            best_err = err
            best_m = mult
        if h == intermediate_size:
            return mult
    raise ValueError(
        f"Cannot match HF intermediate_size={intermediate_size} for dim={dim}, "
        f"multiple_of={multiple_of} (closest error {best_err} with mult≈{best_m}). "
        f"Use a full STEM params.json with the correct ffn_dim_multiplier."
    )


def synthetic_stem_params_from_hf(
    hf_cfg: dict,
    hf_model_dir: Path,
    *,
    model_dtype: str,
    stem_layers_override: Optional[List[int]] = None,
) -> Any:
    """Build a STEM-style OmegaConf from Hugging Face ``config.json`` fields."""
    from omegaconf import OmegaConf

    dim = int(hf_cfg["hidden_size"])
    n_layers = int(hf_cfg["num_hidden_layers"])
    n_heads = int(hf_cfg["num_attention_heads"])
    n_kv = int(hf_cfg.get("num_key_value_heads", n_heads))
    vocab = int(hf_cfg["vocab_size"])
    intermediate_size = int(hf_cfg["intermediate_size"])
    multiple_of = int(hf_cfg.get("multiple_of", 256))
    norm_eps = float(hf_cfg.get("rms_norm_eps", hf_cfg.get("layer_norm_eps", 1e-5)))
    rope_theta = float(hf_cfg.get("rope_theta", 10000.0))
    max_seqlen = int(hf_cfg.get("max_position_embeddings", 8192))
    weight_tying = bool(hf_cfg.get("tie_word_embeddings", True))

    if hf_cfg.get("head_dim") is not None:
        head_dim = int(hf_cfg["head_dim"])
    else:
        head_dim = dim // n_heads

    ffn_mult = infer_ffn_dim_multiplier(dim, intermediate_size, multiple_of)

    if stem_layers_override is not None:
        stem_layers: Optional[List[int]] = stem_layers_override
    elif hf_cfg.get("stem_layers") is not None:
        stem_layers = list(hf_cfg["stem_layers"])
    else:
        # ``[]`` = no STEM FFN blocks (every layer uses standard ``FeedForward``
        # with w1/w3/w2), which matches a vanilla Hugging Face Llama export after
        # key conversion.  ``None`` would let ``StemTransformer`` default to
        # ``range(1, n_layers)`` (StemFeedForward on most layers) and **will not**
        # match a standard HF state dict.
        stem_layers = []

    model: Dict[str, Any] = {
        "dim": dim,
        "ffn_dim_multiplier": ffn_mult,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "multiple_of": multiple_of,
        "norm_eps": norm_eps,
        "rope_theta": rope_theta,
        "max_seqlen": max_seqlen,
        "weight_tying": weight_tying,
        "init_base_std": 0.02,
        "init_std_factor": "disabled",
        "stem_layers": stem_layers,
        "stem_embedding_dim": None,
        "init_type": "normal",
    }
    sw = hf_cfg.get("sliding_window")
    if sw is not None:
        model["sliding_window"] = int(sw)
    if hf_cfg.get("rope_scaling") is not None:
        model["rope_scaling"] = hf_cfg["rope_scaling"]

    root = {
        "model_type": "llama",
        "model": model,
        "distributed": {"model_dtype": model_dtype},
        "data": {
            "tokenizer": {
                "name": "huggingface",
                "path": str(hf_model_dir.resolve()),
            }
        },
    }
    return OmegaConf.create(root)


def resolve_stem_omega_config(
    params_path: Path,
    hf_dir: Path,
    hf_cfg: dict,
    stem_sd: Dict[str, torch.Tensor],
    stem_layers_csv: Optional[str],
) -> Tuple[Any, bool]:
    """Load params file; if it has no ``model`` key (HF config), synthesize STEM params.

    Returns ``(cfg, used_synthetic)``.
    """
    from omegaconf import OmegaConf, DictConfig

    raw = OmegaConf.load(params_path)
    if isinstance(raw, DictConfig) and "model" in raw:
        return raw, False

    logger.warning(
        "%s has no top-level 'model' key (typical of Hugging Face config.json). "
        "Synthesizing STEM params from HF fields + checkpoint dtype for "
        "validation and output params.json.",
        params_path,
    )
    override: Optional[List[int]] = None
    if stem_layers_csv:
        override = [
            int(x.strip())
            for x in stem_layers_csv.split(",")
            if x.strip() != ""
        ]
    mdtype = _guess_dtype_from_state_dict(stem_sd)
    cfg = synthetic_stem_params_from_hf(
        hf_cfg, hf_dir, model_dtype=mdtype, stem_layers_override=override
    )
    return cfg, True


# ---------------------------------------------------------------------------
# Llama: HF -> STEM (model.* keys as in prepare_stem_checkpoint / training)
# ---------------------------------------------------------------------------

def _should_skip_hf_key(key: str) -> bool:
    if any(key.startswith(p) for p in ("lm_head.", "score.")):
        return False  # handled explicitly
    if key.startswith("model.rotary_emb"):
        return True
    if ".rotary_emb." in key:
        return True
    return False


def convert_llama_hf_to_stem_flat(
    hf_sd: Dict[str, torch.Tensor],
    *,
    tie_word_embeddings: bool,
) -> Dict[str, torch.Tensor]:
    """Map HuggingFace Llama keys to STEM consolidated keys (prefix ``model.``)."""
    out: Dict[str, torch.Tensor] = {}

    for key, tensor in hf_sd.items():
        if _should_skip_hf_key(key):
            continue

        if key == "model.embed_tokens.weight":
            out["model.tok_embeddings.weight"] = tensor
            continue

        if key == "model.norm.weight":
            out["model.norm.weight"] = tensor
            continue

        if key.startswith("model.layers."):
            parts = key.split(".")
            # model.layers.{i}. ...
            if len(parts) < 4:
                continue
            layer_idx = parts[2]
            rest = ".".join(parts[3:])

            if rest == "input_layernorm.weight":
                out[f"model.layers.{layer_idx}.attention_norm.weight"] = tensor
            elif rest == "post_attention_layernorm.weight":
                out[f"model.layers.{layer_idx}.ffn_norm.weight"] = tensor
            elif rest == "self_attn.q_proj.weight":
                out[f"model.layers.{layer_idx}.attention.wq.weight"] = tensor
            elif rest == "self_attn.k_proj.weight":
                out[f"model.layers.{layer_idx}.attention.wk.weight"] = tensor
            elif rest == "self_attn.v_proj.weight":
                out[f"model.layers.{layer_idx}.attention.wv.weight"] = tensor
            elif rest == "self_attn.o_proj.weight":
                out[f"model.layers.{layer_idx}.attention.wo.weight"] = tensor
            elif rest == "mlp.gate_proj.weight":
                out[f"model.layers.{layer_idx}.feed_forward.w1.weight"] = tensor
            elif rest == "mlp.up_proj.weight":
                out[f"model.layers.{layer_idx}.feed_forward.w3.weight"] = tensor
            elif rest == "mlp.down_proj.weight":
                out[f"model.layers.{layer_idx}.feed_forward.w2.weight"] = tensor
            continue

        if key == "lm_head.weight":
            if not tie_word_embeddings:
                out["model.output.weight"] = tensor
            continue

    # Tied LM head: HF omits duplicate lm_head; STEM's LMTransformer state_dict
    # includes ``output.tied_module.weight`` aliasing ``tok_embeddings.weight``.
    if tie_word_embeddings and "model.tok_embeddings.weight" in out:
        w = out["model.tok_embeddings.weight"]
        out["model.output.tied_module.weight"] = w

    return out


def _soft_validate_against_hf_config(
    stem_sd: Dict[str, torch.Tensor], hf_cfg: dict
) -> None:
    """Cheap shape checks vs HF config.json (does not require STEM params.json)."""
    n_layers = int(hf_cfg["num_hidden_layers"])
    hidden = int(hf_cfg["hidden_size"])
    vocab = int(hf_cfg["vocab_size"])
    n_heads = int(hf_cfg["num_attention_heads"])
    n_kv = int(hf_cfg.get("num_key_value_heads", n_heads))
    head_dim = hidden // n_heads

    te = stem_sd.get("model.tok_embeddings.weight")
    if te is not None and list(te.shape) != [vocab, hidden]:
        raise ValueError(
            f"tok_embeddings shape {tuple(te.shape)} != expected [{vocab}, {hidden}] from config.json"
        )

    for i in range(n_layers):
        wq = stem_sd.get(f"model.layers.{i}.attention.wq.weight")
        if wq is None:
            raise ValueError(f"Missing layer {i} attention.wq after conversion")
        if list(wq.shape) != [n_heads * head_dim, hidden]:
            raise ValueError(
                f"Layer {i} wq shape {tuple(wq.shape)} != "
                f"[{n_heads * head_dim}, {hidden}]"
            )
        wk = stem_sd[f"model.layers.{i}.attention.wk.weight"]
        if list(wk.shape) != [n_kv * head_dim, hidden]:
            raise ValueError(f"Layer {i} wk shape mismatch")

    logger.info(
        "Soft validation OK: %d layers, hidden=%d, vocab=%d", n_layers, hidden, vocab
    )


def validate_against_stem_model(
    stem_sd: Dict[str, torch.Tensor],
    cfg: Union[Any, Path, str],
    model_type: str,
) -> None:
    """Load STEM StemLMTransformer from OmegaConf or a path to JSON/YAML."""
    from omegaconf import OmegaConf

    from apps.main.stem import STEM_MODEL_REGISTRY
    from lingua.args import dataclass_from_dict

    if model_type not in STEM_MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type {model_type!r} for validation")

    if isinstance(cfg, (str, Path)):
        cfg = OmegaConf.load(cfg)
    stem_cls, args_cls = STEM_MODEL_REGISTRY[model_type][:2]
    model_args = dataclass_from_dict(args_cls, cfg.model, strict=False)
    model = stem_cls(model_args)

    backbone = {
        k.replace("model.", "lm_transformer.", 1): v
        for k, v in stem_sd.items()
        if k.startswith("model.")
    }
    if len(backbone) != len(stem_sd):
        bad = [k for k in stem_sd if not k.startswith("model.")]
        raise ValueError(f"All consolidated keys must start with 'model.': {bad[:5]!r} ...")

    missing, unexpected = model.load_state_dict(backbone, strict=False)
    stem_expected = len(model.lm_transformer.stem_layers)
    if len(missing) != stem_expected or not all(
        k.startswith("stem_embeddings.") for k in missing
    ):
        raise RuntimeError(
            f"STEM backbone validation failed: missing={missing}, unexpected={unexpected}"
        )
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    logger.info(
        "Strict STEM load check passed (missing only stem_embeddings: %d tensors)",
        stem_expected,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face Llama weights to STEM consolidated.pth layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-model-dir",
        required=True,
        type=Path,
        help="Directory with config.json and model.safetensors (or pytorch_model.bin)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination root (will contain params.json and consolidated/)",
    )
    parser.add_argument(
        "--params-json",
        required=True,
        type=Path,
        help=(
            "STEM training params.json (top-level 'model', 'distributed', 'data') "
            "or Hugging Face config.json (no 'model' key — a STEM params file will "
            "be synthesized from HF fields for output + validation)."
        ),
    )
    parser.add_argument(
        "--stem-layers",
        type=str,
        default=None,
        help=(
            "Comma-separated STEM layer indices for synthesized params only, e.g. "
            "'1,3,5,7,9,11,13,15'. Must match how the checkpoint was trained; wrong "
            "indices break validation (StemFeedForward vs FeedForward). "
            "Default when omitted: [] (no STEM layers — use for vanilla HF Llama), "
            "or config.json 'stem_layers' if present."
        ),
    )
    parser.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If True, lm_head is omitted (output tied to tok embeddings). "
            "Default: read tie_word_embeddings from HF config.json"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("keep", "bf16", "fp16", "fp32"),
        default="keep",
        help="Floating dtype for tensors in consolidated.pth (default: keep HF dtype)",
    )
    parser.add_argument(
        "--no-validate-stem",
        action="store_true",
        help="Skip StemLMTransformer load test (faster; not recommended for production)",
    )
    parser.add_argument(
        "--model-type",
        default="llama",
        help="STEM registry key for validation (default: llama)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    hf_dir = args.hf_model_dir.resolve()
    out_dir = args.output_dir.resolve()
    params_src = args.params_json.resolve()

    if not hf_dir.is_dir():
        raise SystemExit(f"--hf-model-dir is not a directory: {hf_dir}")
    if not params_src.is_file():
        raise SystemExit(f"--params-json not found: {params_src}")

    hf_cfg = read_hf_llama_config(hf_dir)
    arch = hf_cfg.get("architectures", [None])[0]
    model_type_hf = hf_cfg.get("model_type", "")
    if model_type_hf not in ("llama", "llama2", "mistral", "mixtral") and arch not in (
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
    ):
        logger.warning(
            "Config suggests a non-Llama architecture (%s / %s). "
            "Key mapping may be wrong; use only with Llama/Mistral-style MLP and norms.",
            model_type_hf,
            arch,
        )

    tie = args.tie_word_embeddings
    if tie is None:
        tie = bool(hf_cfg.get("tie_word_embeddings", True))

    logger.info("Loading HF weights from %s", hf_dir)
    hf_sd = load_hf_state_dict(hf_dir)
    logger.info("Loaded %d tensors from Hugging Face checkpoint", len(hf_sd))

    stem_sd = convert_llama_hf_to_stem_flat(hf_sd, tie_word_embeddings=tie)
    del hf_sd

    if args.dtype != "keep":
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dt = dtype_map[args.dtype]
        stem_sd = {k: v.to(dtype=dt) if v.is_floating_point() else v for k, v in stem_sd.items()}

    _soft_validate_against_hf_config(stem_sd, hf_cfg)

    stem_cfg, params_synthetic = resolve_stem_omega_config(
        params_src, hf_dir, hf_cfg, stem_sd, args.stem_layers
    )

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {out_dir}. "
            f"Pass --overwrite or choose an empty path."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cons_dir = out_dir / "consolidated"
    cons_dir.mkdir(parents=True, exist_ok=True)

    out_pth = cons_dir / "consolidated.pth"
    logger.info("Writing %s (%d keys)", out_pth, len(stem_sd))
    torch.save(stem_sd, out_pth)

    out_params = out_dir / "params.json"
    if params_synthetic:
        from omegaconf import OmegaConf

        with open(out_params, "w", encoding="utf-8") as f:
            json.dump(
                OmegaConf.to_container(stem_cfg, resolve=True),
                f,
                indent=2,
            )
        logger.info("Wrote synthesized STEM params.json -> %s", out_params)
    else:
        shutil.copy2(params_src, out_params)
        logger.info("Copied params.json -> %s", out_params)

    if not args.no_validate_stem:
        validate_against_stem_model(stem_sd, stem_cfg, model_type=args.model_type)

    logger.info("Done. Consolidated checkpoint root: %s", out_dir)
    logger.info(
        "Next steps:\n"
        "  • Vanilla / ablation: point stem_diagnostics or eval at this directory "
        "(or run consolidate_checkpoints if you only have DCP).\n"
        "  • STEM training init: run prepare_stem_checkpoint.py with --ckpt-path %s\n"
        "  • STEM eval with publisher STEM weights: you still need consolidated_stem.pth "
        "or stem_shards/ beside this tree.",
        out_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
