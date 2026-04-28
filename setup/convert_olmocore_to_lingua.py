import argparse
import base64
import copy
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

logger = logging.getLogger(__name__)

# FSDP / module prefixes seen on OLMoCore optimizer param_names
_OLMO_OPTIM_NAME_PREFIXES: Tuple[str, ...] = (
    "_fsdp_wrapped_module.transformer.",
    "_fsdp_wrapped_module.module.transformer.",
)


def _print_tensor_stats(key: str, tensor: torch.Tensor) -> None:
    """Print element-wise mean and (population) std for one tensor."""
    t = tensor.detach()
    if t.numel() == 0:
        print(f"{key}: mean=n/a std=n/a (empty) shape={tuple(t.shape)}")
        return
    tf = t.float() if not t.is_floating_point() else t
    if tf.dtype in (torch.bfloat16, torch.float16):
        tf = tf.float()
    mean = tf.mean().item()
    std = tf.std(unbiased=False).item()
    print(f"{key}: mean={mean:.6g} std={std:.6g} shape={tuple(t.shape)} dtype={t.dtype}")


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

def map_olmocore_key_to_lingua(olmocore_key: str) -> str:
    """Map an OLMoCore key to its lingua equivalent."""
    map_dict = {
        "transformer.wte.weight": "model.tok_embeddings.weight",
        "transformer.ln_f.weight": "model.norm.weight",
    }
    block_map_dict = {
        "attn_out.weight": "attention.wo.weight",
        "attn_norm.weight": "post_attention_norm.weight",
        "ff_out.weight": "feed_forward.w2.weight",
        "ff_norm.weight": "post_feedforward_norm.weight",
    }
    if olmocore_key.startswith("transformer.blocks."):
        prefix, layer_idx = get_prefix_info(olmocore_key)
        rest = olmocore_key.split(".")[3:]
        rest = ".".join(rest)
        return f"model.layers.{layer_idx}.{block_map_dict[rest]}"
    else:
        try:
            return map_dict[olmocore_key]
        except KeyError:
            logger.warning(f"Skipping unmapped OLMoCore key: {olmocore_key}")
            return None
    
    
        
def get_prefix_info(olmocore_key: str):
    parts = olmocore_key.split(".")
    layer_idx = int(parts[2])
    prefix = f"model.layers.{layer_idx}"
    return prefix, layer_idx
        
def convert_olmocore_to_lingua(
    olmocore_sd: Dict[str, torch.Tensor], 
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    weight_tying: bool,
) -> Dict[str, torch.Tensor]:
    """
    Convert an OLMoCore state dict to lingua format.
    """
    backbone_sd: Dict[str, torch.Tensor] = {}
    for olmocore_key, tensor in olmocore_sd.items():
        _print_tensor_stats(olmocore_key, tensor)
        if "att_proj.weight" in olmocore_key:
            prefix, layer_idx = get_prefix_info(olmocore_key)
            tensors = torch.split(tensor, [n_heads * head_dim, n_kv_heads * head_dim, n_heads * head_dim], dim=0)
            wq, wk, wv = tensors
            wq = permute_qk(wq, n_heads, head_dim)
            wk = permute_qk(wk, n_kv_heads, head_dim)
            backbone_sd[f"model.layers.{layer_idx}.attention.wq.weight"] = wq
            backbone_sd[f"model.layers.{layer_idx}.attention.wk.weight"] = wk
            backbone_sd[f"model.layers.{layer_idx}.attention.wv.weight"] = wv.clone()
        elif "q_norm.weight" in olmocore_key:
            prefix, layer_idx = get_prefix_info(olmocore_key)
            q_norm = permute_qk_norm_weight(tensor, n_heads, head_dim)
            backbone_sd[f"model.layers.{layer_idx}.attention.q_norm.weight"] = q_norm
        elif "k_norm.weight" in olmocore_key:
            prefix, layer_idx = get_prefix_info(olmocore_key)
            k_norm = permute_qk_norm_weight(tensor, n_kv_heads, head_dim)
            backbone_sd[f"model.layers.{layer_idx}.attention.k_norm.weight"] = k_norm
        elif "ff_proj.weight" in olmocore_key:
            prefix, layer_idx = get_prefix_info(olmocore_key)
            w3, w1 = torch.chunk(tensor, 2, dim=0)
            backbone_sd[f"model.layers.{layer_idx}.feed_forward.w3.weight"] = w3
            backbone_sd[f"model.layers.{layer_idx}.feed_forward.w1.weight"] = w1
        elif "transformer.ff_out.weight" in olmocore_key:
            if weight_tying:
                backbone_sd[f"model.output.tied_module.weight"] = tensor
            else:
                backbone_sd[f"model.output.weight"] = tensor
        else:
            lingua_key = map_olmocore_key_to_lingua(olmocore_key)
            if lingua_key is not None:
                backbone_sd[lingua_key] = tensor

    return backbone_sd


def _strip_olmocore_optim_param_name(name: str) -> str:
    for p in _OLMO_OPTIM_NAME_PREFIXES:
        if name.startswith(p):
            return name[len(p) :]
    return name


def _to_olmocore_state_key(stripped: str) -> str:
    """Map optim param suffix to keys used in model.pt / convert_olmocore_to_lingua."""
    if stripped.startswith("blocks."):
        return "transformer." + stripped
    return "transformer." + stripped


def _float_optim_entry(entry: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {
        "step": entry["step"].detach().cpu().float().clone(),
        "exp_avg": entry["exp_avg"].detach().cpu().float().clone(),
        "exp_avg_sq": entry["exp_avg_sq"].detach().cpu().float().clone(),
    }


def _lingua_fqn_from_model_key(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None
    assert model_key.startswith("model.")
    return model_key[len("model.") :]


def convert_olmocore_optim_entry_to_lingua(
    olmocore_key: str,
    entry: Dict[str, Any],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    weight_tying: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Map one OLMoCore optimizer state entry to one or more lingua optimizer states
    (FQNs as in torch.distributed.checkpoint.state_dict.get_state_dict, no ``model.`` prefix).
    """
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    fe = _float_optim_entry(entry)

    if "att_proj.weight" in olmocore_key:
        _, layer_idx = get_prefix_info(olmocore_key)
        ea, eas, st = fe["exp_avg"], fe["exp_avg_sq"], fe["step"]
        wq_ea, wk_ea, wv_ea = torch.split(
            ea, [n_heads * head_dim, n_kv_heads * head_dim, n_heads * head_dim], dim=0
        )
        wq_eas, wk_eas, wv_eas = torch.split(
            eas, [n_heads * head_dim, n_kv_heads * head_dim, n_heads * head_dim], dim=0
        )
        wq_ea = permute_qk(wq_ea, n_heads, head_dim)
        wk_ea = permute_qk(wk_ea, n_kv_heads, head_dim)
        wq_eas = permute_qk(wq_eas, n_heads, head_dim)
        wk_eas = permute_qk(wk_eas, n_kv_heads, head_dim)
        base = f"layers.{layer_idx}.attention"
        out[f"{base}.wq.weight"] = {"step": st.clone(), "exp_avg": wq_ea, "exp_avg_sq": wq_eas}
        out[f"{base}.wk.weight"] = {"step": st.clone(), "exp_avg": wk_ea, "exp_avg_sq": wk_eas}
        out[f"{base}.wv.weight"] = {"step": st.clone(), "exp_avg": wv_ea.clone(), "exp_avg_sq": wv_eas.clone()}
        return out

    if "q_norm.weight" in olmocore_key:
        _, layer_idx = get_prefix_info(olmocore_key)
        out[f"layers.{layer_idx}.attention.q_norm.weight"] = {
            "step": fe["step"].clone(),
            "exp_avg": permute_qk_norm_weight(fe["exp_avg"], n_heads, head_dim),
            "exp_avg_sq": permute_qk_norm_weight(fe["exp_avg_sq"], n_heads, head_dim),
        }
        return out

    if "k_norm.weight" in olmocore_key:
        _, layer_idx = get_prefix_info(olmocore_key)
        out[f"layers.{layer_idx}.attention.k_norm.weight"] = {
            "step": fe["step"].clone(),
            "exp_avg": permute_qk_norm_weight(fe["exp_avg"], n_kv_heads, head_dim),
            "exp_avg_sq": permute_qk_norm_weight(fe["exp_avg_sq"], n_kv_heads, head_dim),
        }
        return out

    if "ff_proj.weight" in olmocore_key:
        _, layer_idx = get_prefix_info(olmocore_key)
        w3_ea, w1_ea = torch.chunk(fe["exp_avg"], 2, dim=0)
        w3_eas, w1_eas = torch.chunk(fe["exp_avg_sq"], 2, dim=0)
        st = fe["step"]
        out[f"layers.{layer_idx}.feed_forward.w3.weight"] = {
            "step": st.clone(),
            "exp_avg": w3_ea,
            "exp_avg_sq": w3_eas,
        }
        out[f"layers.{layer_idx}.feed_forward.w1.weight"] = {
            "step": st.clone(),
            "exp_avg": w1_ea,
            "exp_avg_sq": w1_eas,
        }
        return out

    if olmocore_key == "transformer.ff_out.weight":
        if weight_tying:
            logger.warning(
                "Skipping optimizer state for transformer.ff_out.weight under weight_tying; "
                "only tok_embeddings receives embedding optimizer state."
            )
            return out
        out["output.weight"] = fe
        return out

    mk = map_olmocore_key_to_lingua(olmocore_key)
    fq = _lingua_fqn_from_model_key(mk)
    if fq is not None:
        out[fq] = fe
    return out


def build_olmo3_lm_args(olmocore_config: dict):
    from apps.main.olmo3 import OLMo3LMTransformerArgs

    m = olmocore_config["model"]
    dim = m["d_model"]
    n_heads = m["n_heads"]
    n_kv_heads = m["n_kv_heads"]
    if n_kv_heads is None:
        n_kv_heads = n_heads
    head_dim = m.get("head_dim", dim // n_heads)
    multiple_of = 256
    ffn_dim_multiplier = 1.5
    mlp_h = _effective_olmocore_mlp_hidden_size(olmocore_config)
    if mlp_h is not None:
        computed = compute_ffn_dim_multiplier(dim, mlp_h, multiple_of)
        if computed is not None:
            ffn_dim_multiplier = computed
        else:
            ffn_dim_multiplier = None

    return OLMo3LMTransformerArgs(
        dim=dim,
        n_layers=m["n_layers"],
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_dim_multiplier=ffn_dim_multiplier,
        multiple_of=multiple_of,
        norm_eps=m["layer_norm_eps"],
        rope_theta=m["rope_theta"],
        max_seqlen=m["max_sequence_length"],
        vocab_size=m["embedding_size"],
        weight_tying=m["weight_tying"],
        sliding_window=m.get("sliding_window"),
        seed=m.get("seed", 42),
        init_base_std=0.02,
        init_std_factor="disabled",
    )


def build_lingua_optimizer_state_dict(
    olmocore_optim: Dict[str, Any],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    weight_tying: bool,
    olmocore_config: dict,
) -> Dict[str, Any]:
    """
    Full optimizer state dict (``state`` + ``param_groups``) compatible with
    lingua ``get_state_dict`` / ``dcp.load`` for ``OLMo3LMTransformer`` + AdamW.
    """
    from apps.main.olmo3 import OLMo3LMTransformer
    from lingua.optim import OptimArgs, build_optimizer

    if "state" not in olmocore_optim or "param_groups" not in olmocore_optim:
        raise ValueError(
            "optim.pt must be a PyTorch optimizer dict with 'state' and 'param_groups'"
        )

    id_to_name: Dict[int, str] = {}
    for pg in olmocore_optim["param_groups"]:
        names = pg.get("param_names")
        if names is None:
            raise ValueError(
                "optim.pt param_groups must include 'param_names' (OLMoCore / FSDP export); "
                "cannot map integer state keys without it."
            )
        pids = pg["params"]
        if len(pids) != len(names):
            raise ValueError(
                f"param_groups params ({len(pids)}) and param_names ({len(names)}) length mismatch"
            )
        for pid, pname in zip(pids, names):
            id_to_name[pid] = pname

    converted: Dict[str, Dict[str, torch.Tensor]] = {}
    for pid, entry in olmocore_optim["state"].items():
        stripped = _strip_olmocore_optim_param_name(id_to_name[int(pid)])
        oc_key = _to_olmocore_state_key(stripped)
        piece = convert_olmocore_optim_entry_to_lingua(
            oc_key, entry, n_heads, n_kv_heads, head_dim, weight_tying
        )
        for k, v in piece.items():
            if k in converted:
                logger.warning("Duplicate lingua optim key %s (overwriting)", k)
            converted[k] = v

    args = build_olmo3_lm_args(olmocore_config)
    model = OLMo3LMTransformer(args)
    optim, _ = build_optimizer(
        model,
        OptimArgs(
            lr=float(olmocore_config["optimizer"]["learning_rate"]),
            weight_decay=float(olmocore_config["optimizer"]["weight_decay"]),
        ),
        n_steps=int(
            olmocore_config["scheduler"]["t_max"]
            // olmocore_config["global_train_batch_size"]
        ),
    )
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29514")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    _, template = get_state_dict(
        model, optim, options=StateDictOptions(full_state_dict=True)
    )
    template = copy.deepcopy(template)
    missing: List[str] = []
    extra = set(converted.keys())
    for k in template["state"]:
        if k in converted:
            template["state"][k] = converted[k]
            extra.discard(k)
        else:
            missing.append(k)
    if missing:
        raise RuntimeError(
            f"Converted optimizer is missing {len(missing)} lingua params (e.g. {missing[:5]!r})"
        )
    if extra:
        logger.warning("Unused converted optim keys (not in lingua model): %s", sorted(extra)[:10])

    return template


def backbone_sd_to_model_inner(backbone_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip leading ``model.`` for DCP subtree expected by train.py / checkpoint.py."""
    inner: Dict[str, torch.Tensor] = {}
    for k, v in backbone_sd.items():
        if not k.startswith("model."):
            raise ValueError(f"Expected backbone key to start with 'model.', got {k!r}")
        inner[k[len("model.") :]] = v
    return inner


def _decode_olmo_safetensors_key(raw_key: str) -> Tuple[Tuple[Any, ...], bool]:
    """Decode OLMoCore's pickled, url-safe-base64 safetensors key wrapper."""
    padded = raw_key + "=" * ((4 - len(raw_key) % 4) % 4)
    path, serialized = pickle.loads(base64.urlsafe_b64decode(padded))
    return tuple(path), bool(serialized)


def _safetensors_expected_size(path: Path) -> Optional[int]:
    """Return expected total file size from the safetensors header, if readable."""
    import struct

    try:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
    except Exception:
        return None

    max_end = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        max_end = max(max_end, meta["data_offsets"][1])
    return 8 + header_len + max_end


def _format_bytes(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GiB"


def load_olmocore_model_checkpoint(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    """Load OLMoCore model weights from model.pt or encoded-key model.safetensors."""
    pt_path = ckpt_dir / "model.pt"
    if pt_path.is_file():
        logger.info("  Loading model.pt...")
        return torch.load(pt_path, map_location="cpu", weights_only=False)

    st_path = ckpt_dir / "model.safetensors"
    if st_path.is_file():
        from safetensors.torch import load_file

        logger.info("  Loading model.safetensors...")
        raw_sd = load_file(str(st_path), device="cpu")
        out: Dict[str, torch.Tensor] = {}
        for raw_key, tensor in raw_sd.items():
            key_path, serialized = _decode_olmo_safetensors_key(raw_key)
            if serialized:
                raise ValueError(f"Unexpected serialized model entry in {st_path}: {key_path}")
            if len(key_path) != 1 or not isinstance(key_path[0], str):
                raise ValueError(f"Unexpected model key in {st_path}: {key_path!r}")
            out[key_path[0]] = tensor
        return out

    raise FileNotFoundError(f"No model.pt or model.safetensors found in {ckpt_dir}")


def load_olmocore_optim_checkpoint(path: Path) -> Dict[str, Any]:
    """Load OLMoCore AdamW state from optim.pt or encoded-key optim.safetensors."""
    if path.suffix == ".pt":
        logger.info("Loading OLMoCore optim.pt for AdamW state conversion...")
        return torch.load(path, map_location="cpu", weights_only=False)

    if path.name != "optim.safetensors":
        raise ValueError(f"Unsupported optimizer checkpoint format: {path}")

    from safetensors import safe_open

    logger.info("Loading OLMoCore optim.safetensors for AdamW state conversion...")
    try:
        fctx = safe_open(str(path), framework="pt", device="cpu")
    except Exception as exc:
        expected = _safetensors_expected_size(path)
        actual = path.stat().st_size if path.exists() else 0
        if expected is not None and actual < expected:
            raise RuntimeError(
                f"{path} is incomplete: header expects {_format_bytes(expected)}, "
                f"but file is {_format_bytes(actual)}. Re-download optim.safetensors "
                "or rerun with --no-optim for a model-only conversion."
            ) from exc
        raise

    optim: Dict[str, Any] = {"state": {}, "param_groups": None}
    with fctx as f:
        for raw_key in f.keys():
            key_path, serialized = _decode_olmo_safetensors_key(raw_key)
            tensor = f.get_tensor(raw_key)
            if key_path == ("param_groups",):
                if not serialized:
                    raise ValueError("Expected serialized param_groups in optim.safetensors")
                optim["param_groups"] = pickle.loads(bytes(tensor.cpu().tolist()))
                continue

            if len(key_path) != 3 or key_path[0] != "state":
                logger.warning("Skipping unexpected optimizer safetensors key: %r", key_path)
                continue
            _, pid, field = key_path
            optim["state"].setdefault(int(pid), {})[field] = tensor

    if optim["param_groups"] is None:
        raise ValueError(f"No param_groups entry found in {path}")
    return optim


def _effective_olmocore_mlp_hidden_size(olmocore_config: dict) -> Optional[int]:
    """
    Return the lingua FFN hidden size.

    OLMoCore stores SwiGLU's fused input projection width in mlp_hidden_size
    (w3 + w1), while lingua's FeedForward hidden size is the per-branch width.
    """
    m = olmocore_config["model"]
    mlp_h = m.get("mlp_hidden_size")
    if mlp_h is None:
        return None
    if str(m.get("activation_type", "")).lower() == "swiglu":
        return int(mlp_h) // 2
    return int(mlp_h)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_lingua_dcp(
    model_inner_sd: Dict[str, torch.Tensor],
    output_dir: Path,
    optim_sd: Optional[Dict[str, Any]] = None,
):
    """
    Save DCP checkpoint compatible with ``lingua.checkpoint.load_from_checkpoint``:
    top-level ``model`` (FQNs without ``model.`` prefix) and optional ``optim``.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29513")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"model": model_inner_sd}
    if optim_sd is not None:
        payload["optim"] = optim_sd
    logger.info(
        "Saving DCP to %s (%d model tensors%s)",
        output_dir,
        len(model_inner_sd),
        f", optim" if optim_sd is not None else "",
    )
    dcp.save(payload, checkpoint_id=str(output_dir))
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
    olmocore_config: dict,
    dp_degree: int,
    tokenizer_path: str,
):
    params = {
        "name": "olmo3-lingua",
        "dump_dir": "logs/debug",
        "seed": 777,
        "grad_acc_steps": 1,
        "gc_collect_freq": 1000,
        "probe_freq": None,
        "steps": int(olmocore_config["scheduler"]["t_max"] / olmocore_config["global_train_batch_size"]),
        "data": {
            "root_dir": "data",
            "sources": {
                "dclm-baseline_shuffled": 1.0
            },
            "batch_size": olmocore_config["global_train_batch_size"] // dp_degree,
            "seq_len": olmocore_config["model"]["max_sequence_length"],
            "n_views": 2,
            "seed": 42,
            "add_bos": True,
            "add_eos": True,
            "load_async": True,
            "prefetch_size": 1024,
            "tokenizer": {
                "name": "huggingface",
                "path": "allenai/OLMo-2-0425-1B",
            },
        },
        "optim": {
            "lr": olmocore_config["optimizer"]["learning_rate"],
            "weight_decay": olmocore_config["optimizer"]["weight_decay"],
            "epsilon": 1e-8,
            "beta1": 0.9,
            "beta2": 0.95,
            "clip": 1.0,
            "scheduler": "cosine",
            "warmup": int(olmocore_config["scheduler"]["t_warmup"] / olmocore_config["global_train_batch_size"]),
            "lr_min_ratio": olmocore_config["scheduler"]["alpha_f"],
            "cycle_length": 1.0,
            "cosine_theta": 1.0,
            "annealing_step": 0,
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
            "dump": {"every": 10000, "keep": 1},
            "eval": {"every": 10000, "keep": 1},
            "path": "logs/debug/checkpoints",
            "init_ckpt_path": "",
            "continue_training_from_init": False,
        },
        "logging": {"freq": 10, "acc_freq": None, "wandb": None}, 
    }
    
    dim = olmocore_config["model"]["d_model"]
    multiple_of = 256
    ffn_dim_multiplier = 1.5
    mlp_h = _effective_olmocore_mlp_hidden_size(olmocore_config)
    if mlp_h is not None:
        computed = compute_ffn_dim_multiplier(dim, mlp_h, multiple_of)
        ffn_dim_multiplier = computed
    n_heads = olmocore_config["model"]["n_heads"]
    n_kv_heads = olmocore_config["model"]["n_kv_heads"]
    if n_kv_heads is None:
        n_kv_heads = n_heads
    head_dim = olmocore_config["model"].get(
        "head_dim", dim // n_heads
    )
    
    params["model"] = {
        "dim": dim,
        "n_layers": olmocore_config["model"]["n_layers"],
        "head_dim": head_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "ffn_dim_multiplier": ffn_dim_multiplier,
        "multiple_of": multiple_of,
        "norm_eps": olmocore_config["model"]["layer_norm_eps"],
        "rope_theta": olmocore_config["model"]["rope_theta"],
        "rope_scaling": None,
        "init_base_std": 0.02,
        "init_std_factor": "disabled",
        "max_seqlen": olmocore_config["model"]["max_sequence_length"],
        "seed": 42,
        "vocab_size": olmocore_config["model"]["embedding_size"],
        "weight_tying": olmocore_config["model"]["weight_tying"],
        "sliding_window": olmocore_config["model"].get("sliding_window", None),
    }
    
    if "data" in params and "tokenizer" in params["data"]:
        params["data"]["tokenizer"]["name"] = "huggingface"
        params["data"]["tokenizer"]["path"] = tokenizer_path
        
    params_path = output_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved params.json to {params_path}")
    
    
def main():
    parser = argparse.ArgumentParser(
        description="Convert OLMoCore checkpoint to lingua format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--olmocore-ckpt-dir",
        required=True,
        help=(
            "Path to OLMoCore checkpoint directory "
            "(contains config.yaml plus model.pt/model.safetensors and optional "
            "optim.pt/optim.safetensors)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the lingua checkpoint",
    )
    parser.add_argument(
        "--dp-degree",
        type=int,
        default=32,
        help="Number of data parallel shards (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Path to the tokenizer",
    )
    parser.add_argument(
        "--no-optim",
        action="store_true",
        help="Do not read optim.pt or write optimizer state (model weights only)",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    olmocore_ckpt_dir = Path(args.olmocore_ckpt_dir)
    output_dir = Path(args.output_dir)
    dp_degree = args.dp_degree
    tokenizer_path = args.tokenizer_path
    with open(olmocore_ckpt_dir / "config.yaml") as f:
        olmocore_config = yaml.safe_load(f)
    
    logger.info("Loading OLMoCore checkpoint...")
    olmocore_sd = load_olmocore_model_checkpoint(olmocore_ckpt_dir)
    logger.info(f"Loaded {len(olmocore_sd)} keys total")
    
    n_heads = olmocore_config["model"]["n_heads"]
    n_kv_heads = olmocore_config["model"]["n_kv_heads"]
    if n_kv_heads is None:
        n_kv_heads = n_heads
    head_dim = olmocore_config["model"].get(
        "head_dim", olmocore_config["model"]["d_model"] // n_heads
    )
    weight_tying = olmocore_config["model"]["weight_tying"]
    
    logger.info(f"Number of heads: {n_heads}")
    logger.info(f"Number of key-value heads: {n_kv_heads}")
    logger.info(f"Head dimension: {head_dim}")
    logger.info(f"Weight tying: {weight_tying}")
    
    logger.info("Converting OLMoCore keys to lingua format...")
    backbone_sd = convert_olmocore_to_lingua(olmocore_sd, n_heads, n_kv_heads, head_dim, weight_tying)
    logger.info(f"Converted {len(backbone_sd)} keys to lingua format")

    model_inner = backbone_sd_to_model_inner(backbone_sd)
    del backbone_sd

    optim_sd = None
    optim_path = olmocore_ckpt_dir / "optim.pt"
    if not args.no_optim:
        if optim_path.is_file():
            olmocore_optim = load_olmocore_optim_checkpoint(optim_path)
            optim_sd = build_lingua_optimizer_state_dict(
                olmocore_optim,
                n_heads,
                n_kv_heads,
                head_dim,
                weight_tying,
                olmocore_config,
            )
            logger.info("Converted optimizer state to lingua / DCP format")
        elif (olmocore_ckpt_dir / "optim.safetensors").is_file():
            olmocore_optim = load_olmocore_optim_checkpoint(
                olmocore_ckpt_dir / "optim.safetensors"
            )
            optim_sd = build_lingua_optimizer_state_dict(
                olmocore_optim,
                n_heads,
                n_kv_heads,
                head_dim,
                weight_tying,
                olmocore_config,
            )
            logger.info("Converted optimizer state to lingua / DCP format")
        else:
            logger.warning(
                "No optim.pt or optim.safetensors in %s; saving model-only checkpoint",
                olmocore_ckpt_dir,
            )

    logger.info("Saving DCP (train.py-compatible layout)...")
    save_lingua_dcp(model_inner, output_dir, optim_sd=optim_sd)
    del model_inner
    logger.info("DCP save complete")
    
    logger.info("Creating params.json...")
    create_params_json(output_dir, olmocore_config, dp_degree, tokenizer_path)

    if dist.is_initialized():
        dist.destroy_process_group()

    logger.info("")
    logger.info(f"Lingua checkpoint saved to: {output_dir}")
    logger.info(f"  DCP dir:  {output_dir} (__0_0.distcp + .metadata)")
    logger.info(f"  Config:   {output_dir}/params.json")
    if optim_sd is not None:
        logger.info("  Optimizer state included for continue_training_from_init + load_from_checkpoint")


if __name__ == "__main__":
    main()
    
