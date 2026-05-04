"""CLI for STEM knowledge-editing experiments.

Example:

    python -m apps.main.knowledge_editing.run_stem_knowledge_edit \
        --config apps/main/configs/stem_dag_llama3_1B_midfine.yaml \
        --checkpoint-dir /path/to/dump/checkpoints/0000100000 \
        --output-dir /tmp/stem_ke \
        --source-entity Spain \
        --target-entity Germany \
        --top-k 4 \
        --max-new-tokens 100 \
        --temperature 0.0 \
        --seed 0 \
        --device cuda \
        --dtype bfloat16

Use --dry-run-tokenization to validate prompt construction and entity spans
without loading a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from omegaconf import OmegaConf

from apps.main.knowledge_editing.experiment import (
    GenerationResult,
    TokenizedPrompt,
    TopKResult,
    make_stem_embedding_override_fn,
    plot_topk_probabilities,
    replace_last_entity,
    run_generation,
    run_next_token_topk,
    sanitize_for_path,
    save_results,
    set_deterministic_seed,
    tokenize_with_entity_span,
    write_json,
)

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone STEM/STEM-DAG knowledge-editing experiment."
    )
    parser.add_argument("--config", required=True, help="Training config YAML or checkpoint params.json.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Checkpoint step directory, dump/checkpoints directory, dump directory, "
            "or consolidated directory. Required unless --dry-run-tokenization is set."
        ),
    )
    parser.add_argument("--output-dir", default="knowledge_edit_outputs")
    parser.add_argument("--source-entity", required=True)
    parser.add_argument("--target-entity", required=True)
    parser.add_argument("--top-k", type=int, default=4, help="Top-k next-token probabilities to save and plot.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--sampling-top-k",
        "--generation-top-k",
        dest="sampling_top_k",
        type=int,
        default=None,
        help="Optional top-k filter for generation sampling. Distinct from --top-k plot size.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"),
    )
    parser.add_argument(
        "--edit-mode",
        default="auto",
        choices=("auto", "one_to_one", "average", "copy", "left_pad", "right_pad"),
    )
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--add-bos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override prompt BOS handling. By default uses config data.add_bos if present.",
    )
    parser.add_argument(
        "--add-eos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override prompt EOS handling. By default uses config data.add_eos if present.",
    )
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument(
        "--dry-run-tokenization",
        action="store_true",
        help="Build prompts, tokenize source/target entity spans, print diagnostics, and exit.",
    )
    return parser.parse_args()


def configure_logging(run_dir: Optional[Path]) -> None:
    handlers = [logging.StreamHandler()]
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(run_dir / "run.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def torch_load(path: Path, *, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_dtype(dtype_arg: str, cfg: Any) -> torch.dtype:
    if dtype_arg == "auto":
        cfg_dtype = str(OmegaConf.select(cfg, "distributed.model_dtype", default="fp32")).lower()
    else:
        cfg_dtype = dtype_arg.lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if cfg_dtype not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_arg!r} resolved to {cfg_dtype!r}")
    return mapping[cfg_dtype]


def resolve_bool_from_config(cfg: Any, key: str, override: Optional[bool], default: bool) -> bool:
    if override is not None:
        return bool(override)
    value = OmegaConf.select(cfg, key, default=default)
    return bool(value)


def build_tokenizer_from_config(
    cfg: Any,
    *,
    tokenizer_name: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> Any:
    from lingua.tokenizer import build_tokenizer

    name = tokenizer_name or OmegaConf.select(cfg, "data.tokenizer.name", default=None)
    path = tokenizer_path or OmegaConf.select(cfg, "data.tokenizer.path", default=None)
    if name is None:
        raise ValueError("Tokenizer name missing. Set data.tokenizer.name or pass --tokenizer-name.")
    return build_tokenizer(name, path)


def _numeric_step_dirs(path: Path) -> list[Path]:
    return sorted(
        [child for child in path.iterdir() if child.is_dir() and re.fullmatch(r"\d{10}", child.name)],
        key=lambda p: int(p.name),
    )


def resolve_checkpoint_paths(checkpoint_dir: str, config_path: str) -> Tuple[Path, Path]:
    """Return (step_dir, consolidated_dir), consolidating backbone if needed."""

    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    from lingua.checkpoint import CONSOLIDATE_FOLDER, CONSOLIDATE_NAME, CONFIG_NAME, consolidate_checkpoints

    path = Path(checkpoint_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if path.name == CONSOLIDATE_FOLDER and (path / CONSOLIDATE_NAME).exists():
        return path.parent, path

    if (path / "checkpoints").is_dir():
        return resolve_checkpoint_paths(str(path / "checkpoints"), config_path)

    steps = _numeric_step_dirs(path) if path.is_dir() else []
    if steps:
        latest = steps[-1]
        LOG.info("Resolved checkpoint collection %s to latest step %s", path, latest)
        return resolve_checkpoint_paths(str(latest), config_path)

    consolidated = path / CONSOLIDATE_FOLDER
    if (consolidated / CONSOLIDATE_NAME).exists():
        if not (consolidated / CONFIG_NAME).exists():
            LOG.warning(
                "Checkpoint consolidated params.json is missing; using CLI config %s for model construction.",
                config_path,
            )
        return path, consolidated

    if (path / ".metadata").exists():
        if (path / CONFIG_NAME).exists():
            return path, Path(consolidate_checkpoints(str(path)))

        consolidated.mkdir(exist_ok=True)
        output = consolidated / CONSOLIDATE_NAME
        if not output.exists():
            LOG.info("Consolidating DCP backbone checkpoint at %s", path)
            dcp_to_torch_save(str(path), str(output))
        LOG.warning(
            "Checkpoint step params.json was missing; used CLI config at %s for model construction.",
            config_path,
        )
        return path, consolidated

    raise FileNotFoundError(
        "Could not resolve checkpoint. Expected one of: a step directory with "
        ".metadata or consolidated/consolidated.pth, a consolidated directory, "
        "a checkpoints directory containing 10-digit step folders, or a dump "
        "directory containing checkpoints/."
    )


def register_stem_model_types() -> Mapping[str, Any]:
    from apps.main.stem import STEM_MODEL_REGISTRY
    from apps.main.stem_dag import DAG_STEM_MODEL_REGISTRY

    STEM_MODEL_REGISTRY.update(DAG_STEM_MODEL_REGISTRY)
    return STEM_MODEL_REGISTRY


def _normalize_backbone_keys(backbone_dict: Dict[str, torch.Tensor], model: Any) -> Dict[str, torch.Tensor]:
    if "model" in backbone_dict and isinstance(backbone_dict["model"], dict):
        backbone_dict = backbone_dict["model"]

    if not backbone_dict:
        raise ValueError("Backbone checkpoint state_dict is empty")

    keys = list(backbone_dict.keys())
    first_key = keys[0]
    if first_key.startswith("model."):
        return {key.replace("model.", "lm_transformer.", 1): value for key, value in backbone_dict.items()}

    if hasattr(model, "lm_transformer"):
        already_prefixed = all(key.startswith("lm_transformer.") or key.startswith("stem_embeddings.") for key in keys)
        if not already_prefixed:
            likely_lm_keys = (
                "layers.",
                "tok_embeddings.",
                "norm.",
                "output.",
                "rope_embeddings.",
            )
            if any(key.startswith(likely_lm_keys) for key in keys):
                return {f"lm_transformer.{key}": value for key, value in backbone_dict.items()}

    return backbone_dict


def _load_stem_weights_if_needed(model: Any, step_dir: Path, consolidated_dir: Path, missing_keys: list[str]) -> None:
    from torch import nn
    from lingua.stem_checkpoint import CONSOLIDATE_STEM_NAME, consolidate_stem_shards
    from lingua.stem_dist_utils import ParallelEmbedding

    stem_layers = list(getattr(model, "stem_layers", []))
    if not stem_layers:
        raise RuntimeError("No STEM layers exist in the loaded model; cannot run STEM editing.")
    if not hasattr(model, "stem_embeddings"):
        raise RuntimeError("Loaded model has no model.stem_embeddings; cannot run STEM editing.")

    expected_stem_keys = {f"stem_embeddings.{idx}.weight" for idx in range(len(stem_layers))}
    missing_stem = expected_stem_keys & set(missing_keys)
    if missing_stem and missing_stem != expected_stem_keys:
        raise RuntimeError(
            "Checkpoint is missing only part of the STEM embedding table set: "
            f"{sorted(missing_stem)}"
        )
    if not missing_stem:
        LOG.info("STEM embeddings were loaded from the backbone checkpoint.")
        return

    stem_file = consolidated_dir / CONSOLIDATE_STEM_NAME
    if (step_dir / "stem_shards").exists():
        consolidate_stem_shards(str(step_dir))
    if not stem_file.exists():
        raise FileNotFoundError(
            f"Missing STEM shards for checkpoint {step_dir}. Expected either "
            f"{step_dir / 'stem_shards'} with stem_model_mp*.pt files or "
            f"{stem_file}."
        )

    stem_dict = torch_load(stem_file, map_location="cpu")
    loaded = []
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, ParallelEmbedding)):
                weight_key = f"{module_name}.weight" if module_name else "weight"
                if weight_key in stem_dict:
                    module.weight.copy_(stem_dict[weight_key].to(device=module.weight.device, dtype=module.weight.dtype))
                    loaded.append(weight_key)
    if not loaded:
        raise RuntimeError(
            f"Found {stem_file}, but no matching model.stem_embeddings.*.weight keys were loaded."
        )
    LOG.info("Loaded %d STEM embedding tensors from %s", len(loaded), stem_file)


def load_stem_model_and_tokenizer(
    *,
    consolidated_dir: Path,
    step_dir: Path,
    config_path: str,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer_name: Optional[str],
    tokenizer_path: Optional[str],
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    from lingua.args import dataclass_from_dict

    registry = register_stem_model_types()
    checkpoint_params = consolidated_dir / "params.json"
    cfg_source = checkpoint_params if checkpoint_params.exists() else Path(config_path)
    cfg = OmegaConf.load(cfg_source)
    model_type = str(OmegaConf.select(cfg, "model_type", default="llama"))
    if model_type not in registry:
        raise ValueError(
            f"Unknown STEM model_type {model_type!r}. Available model types: {sorted(registry)}"
        )

    model_cls, model_args_cls = registry[model_type][:2]
    model_args = dataclass_from_dict(model_args_cls, cfg.model, strict=False)
    tokenizer = build_tokenizer_from_config(
        cfg,
        tokenizer_name=tokenizer_name,
        tokenizer_path=tokenizer_path,
    )

    LOG.info("Building %s from %s", model_cls.__name__, cfg_source)
    model = model_cls(model_args)
    backbone_file = consolidated_dir / "consolidated.pth"
    if not backbone_file.exists():
        raise FileNotFoundError(f"Missing consolidated backbone checkpoint: {backbone_file}")
    backbone_dict = torch_load(backbone_file, map_location="cpu")
    backbone_dict = _normalize_backbone_keys(backbone_dict, model)
    missing_keys, unexpected_keys = model.load_state_dict(backbone_dict, strict=False)

    stem_layers = list(getattr(model, "stem_layers", []))
    expected_alpha_missing = {
        f"lm_transformer.layers.{idx}.feed_forward.alpha" for idx in stem_layers
    }
    expected_stem_missing = {f"stem_embeddings.{idx}.weight" for idx in range(len(stem_layers))}
    unaccounted_missing = set(missing_keys) - expected_stem_missing - expected_alpha_missing
    if unaccounted_missing:
        raise RuntimeError(f"Unexpected missing checkpoint keys: {sorted(unaccounted_missing)}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys}")

    _load_stem_weights_if_needed(model, step_dir, consolidated_dir, list(missing_keys))
    model.to(device=device)
    model.to(dtype=dtype)
    model.eval()

    loader_metadata = {
        "config_used_for_model": str(cfg_source),
        "cli_config": str(Path(config_path).resolve()),
        "checkpoint_step_dir": str(step_dir),
        "checkpoint_consolidated_dir": str(consolidated_dir),
        "model_type": model_type,
        "model_class": model_cls.__name__,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
    }
    return model, tokenizer, cfg, loader_metadata


def get_special_token_id(tokenizer: Any, name: str) -> Optional[int]:
    value = getattr(tokenizer, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def build_prompts_and_tokenization(
    tokenizer: Any,
    *,
    source_entity: str,
    target_entity: str,
    add_bos: bool,
    add_eos: bool,
) -> Tuple[Dict[str, str], Dict[str, TokenizedPrompt]]:
    original_prompt = build_country_capital_prompt(source_entity)
    target_prompt = replace_last_entity(original_prompt, source_entity, target_entity)
    original_tokens = tokenize_with_entity_span(
        tokenizer,
        original_prompt,
        source_entity,
        add_bos=add_bos,
        add_eos=add_eos,
    )
    target_tokens = tokenize_with_entity_span(
        tokenizer,
        target_prompt,
        target_entity,
        add_bos=add_bos,
        add_eos=add_eos,
    )
    return (
        {"original": original_prompt, "target": target_prompt},
        {"original": original_tokens, "target": target_tokens},
    )


def detect_model_mode_metadata(model: Any, cfg: Any) -> Dict[str, Any]:
    model_type = str(OmegaConf.select(cfg, "model_type", default="unknown"))
    config_alpha_mode = OmegaConf.select(cfg, "model.alpha_mode", default=None)
    module_alpha_modes = {}
    for layer_idx in list(getattr(model, "stem_layers", [])):
        try:
            feed_forward = model.lm_transformer.layers[layer_idx].feed_forward
            module_alpha_modes[str(layer_idx)] = getattr(feed_forward, "alpha_mode", None)
        except Exception:
            module_alpha_modes[str(layer_idx)] = None
    inferred_alpha_modes = sorted({mode for mode in module_alpha_modes.values() if mode is not None})
    dag_indicator = model_type.endswith("_dag") or bool(inferred_alpha_modes)
    sum_indicator = config_alpha_mode == "sum" or inferred_alpha_modes == ["sum"]
    warning = None
    if not dag_indicator:
        warning = "No obvious DAG indicator detected; running because valid STEM embeddings exist."
        LOG.warning(warning)
    elif not sum_indicator:
        warning = "DAG model detected, but alpha_mode='sum' was not clearly detected."
        LOG.warning(warning)
    return {
        "model_type": model_type,
        "config_model_alpha_mode": config_alpha_mode,
        "module_alpha_modes_by_layer": module_alpha_modes,
        "detected_dag": dag_indicator,
        "detected_sum_mode": sum_indicator,
        "warning": warning,
    }


def dry_run(args: argparse.Namespace) -> None:
    configure_logging(None)
    cfg = OmegaConf.load(args.config)
    tokenizer = build_tokenizer_from_config(
        cfg,
        tokenizer_name=args.tokenizer_name,
        tokenizer_path=args.tokenizer_path,
    )
    add_bos = resolve_bool_from_config(cfg, "data.add_bos", args.add_bos, default=False)
    add_eos = resolve_bool_from_config(cfg, "data.add_eos", args.add_eos, default=False)
    prompts, tokenization = build_prompts_and_tokenization(
        tokenizer,
        source_entity=args.source_entity,
        target_entity=args.target_entity,
        add_bos=add_bos,
        add_eos=add_eos,
    )
    payload = {
        "source_entity": args.source_entity,
        "target_entity": args.target_entity,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "prompt_original": prompts["original"],
        "prompt_target": prompts["target"],
        "source_span": tokenization["original"].entity_span,
        "target_span": tokenization["target"].entity_span,
    }
    print(json.dumps(json.loads(json.dumps(payload, default=lambda obj: getattr(obj, "__dict__", str(obj)))), indent=2))


def run(args: argparse.Namespace) -> Path:
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required unless --dry-run-tokenization is set")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"knowledge_edit_{sanitize_for_path(args.source_entity)}_"
        f"to_{sanitize_for_path(args.target_entity)}_{timestamp}"
    )
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    configure_logging(run_dir)
    set_deterministic_seed(args.seed)

    cfg_for_flags = OmegaConf.load(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false.")

    step_dir, consolidated_dir = resolve_checkpoint_paths(args.checkpoint_dir, args.config)
    dtype = parse_dtype(args.dtype, cfg_for_flags)
    model, tokenizer, cfg, loader_metadata = load_stem_model_and_tokenizer(
        consolidated_dir=consolidated_dir,
        step_dir=step_dir,
        config_path=args.config,
        device=device,
        dtype=dtype,
        tokenizer_name=args.tokenizer_name,
        tokenizer_path=args.tokenizer_path,
    )

    add_bos = resolve_bool_from_config(cfg, "data.add_bos", args.add_bos, default=False)
    add_eos = resolve_bool_from_config(cfg, "data.add_eos", args.add_eos, default=False)
    prompts, tokenization = build_prompts_and_tokenization(
        tokenizer,
        source_entity=args.source_entity,
        target_entity=args.target_entity,
        add_bos=add_bos,
        add_eos=add_eos,
    )

    source_span = tokenization["original"].entity_span
    target_span = tokenization["target"].entity_span
    override = make_stem_embedding_override_fn(
        model,
        source_span.token_positions,
        target_span.token_ids,
        source_token_ids=source_span.token_ids,
        edit_mode=args.edit_mode,
        pad_token_id=get_special_token_id(tokenizer, "pad_id"),
        eos_token_id=get_special_token_id(tokenizer, "eos_id"),
        bos_token_id=get_special_token_id(tokenizer, "bos_id"),
    )
    for warning in override.plan.warnings:
        LOG.warning(warning)

    metadata: Dict[str, Any] = {
        "source_entity_text": args.source_entity,
        "target_entity_text": args.target_entity,
        "source_token_ids": source_span.token_ids,
        "source_decoded_pieces": source_span.decoded_pieces,
        "target_token_ids": target_span.token_ids,
        "target_decoded_pieces": target_span.decoded_pieces,
        "source_token_positions_in_prompt": source_span.token_positions,
        "requested_edit_mode": args.edit_mode,
        "selected_edit_mode": override.plan.resolved_mode,
        "edit_plan": override.plan,
        "stem_layers_used": list(getattr(model, "stem_layers", [])),
        "add_bos": add_bos,
        "add_eos": add_eos,
        "seed": args.seed,
        "top_k_next_token": args.top_k,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_top_k": args.sampling_top_k,
        },
        "loader": loader_metadata,
        "model_mode": detect_model_mode_metadata(model, cfg),
    }

    (run_dir / "prompt_original.txt").write_text(prompts["original"])
    (run_dir / "prompt_target.txt").write_text(prompts["target"])

    LOG.info("Running next-token probability experiment")
    topk_results: Dict[str, TopKResult] = {
        "original": run_next_token_topk(
            model,
            tokenizer,
            tokenization["original"].token_ids,
            case="original",
            prompt=prompts["original"],
            top_k=args.top_k,
            device=device,
            attn_impl=args.attn_impl,
        ),
        "target": run_next_token_topk(
            model,
            tokenizer,
            tokenization["target"].token_ids,
            case="target",
            prompt=prompts["target"],
            top_k=args.top_k,
            device=device,
            attn_impl=args.attn_impl,
        ),
        "intervened": run_next_token_topk(
            model,
            tokenizer,
            tokenization["original"].token_ids,
            case="intervened",
            prompt=prompts["original"],
            top_k=args.top_k,
            device=device,
            stem_embeddings_fn=override,
            attn_impl=args.attn_impl,
        ),
    }

    LOG.info("Running qualitative generation experiment")
    eos_id = get_special_token_id(tokenizer, "eos_id")
    generations: Dict[str, GenerationResult] = {
        "original": run_generation(
            model,
            tokenizer,
            tokenization["original"].token_ids,
            case="original",
            prompt=prompts["original"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.sampling_top_k,
            seed=args.seed,
            device=device,
            eos_token_id=eos_id,
            attn_impl=args.attn_impl,
        ),
        "target": run_generation(
            model,
            tokenizer,
            tokenization["target"].token_ids,
            case="target",
            prompt=prompts["target"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.sampling_top_k,
            seed=args.seed,
            device=device,
            eos_token_id=eos_id,
            attn_impl=args.attn_impl,
        ),
        "intervened": run_generation(
            model,
            tokenizer,
            tokenization["original"].token_ids,
            case="intervened",
            prompt=prompts["original"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.sampling_top_k,
            seed=args.seed,
            device=device,
            eos_token_id=eos_id,
            stem_embeddings_fn=override,
            attn_impl=args.attn_impl,
        ),
    }

    plot_topk_probabilities(
        topk_results,
        output_png=run_dir / "knowledge_edit_topk_probs.png",
        output_pdf=run_dir / "knowledge_edit_topk_probs.pdf",
        source_entity=args.source_entity,
        target_entity=args.target_entity,
    )
    save_results(
        run_dir,
        metadata=metadata,
        prompts=prompts,
        tokenization=tokenization,
        topk_results=topk_results,
        generations=generations,
    )
    write_json(run_dir / "metadata.json", metadata)
    LOG.info("Knowledge-editing experiment complete: %s", run_dir)
    return run_dir


def main() -> None:
    args = parse_args()
    if args.dry_run_tokenization:
        dry_run(args)
        return
    run_dir = run(args)
    print(f"Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
