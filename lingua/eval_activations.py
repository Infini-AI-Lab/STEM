"""Task- and sample-aligned eval-time activation diagnostics.

Round 3 of the unified STEM diagnostics pipeline.  Reuses
:class:`lingua.diagnostics.StemDiagnosticsCollector` in ``mode="eval"`` to
capture forward-only per-layer / per-path metrics on the *actual* lm-eval
samples that produced reported task scores.

Design constraints
------------------

* No backward, no optimizer touch.  Hooks are forward-only in eval mode.
* No mutation of model outputs.  We rerun a forward pass on the exact
  prompt-target / prompt-generation tokens used by lm-eval; the lm-eval
  result objects are not modified.
* No huge activation all-gather.  Rank 0 (or single-process) does all
  detailed bookkeeping.  When ``rank0_only=False`` non-zero ranks write
  shard files alongside their data; aggregation is left to downstream
  tools.
* Bounded memory.  Per-sample token capping and per-task sample capping
  are configured by ``DiagnosticsArgs.eval_activation_max_*``.

Top-level entry point::

    from lingua.eval_activations import capture_eval_activations
    if cfg.diagnostics.enabled and cfg.diagnostics.collect_eval_activations:
        capture_eval_activations(
            model=model,
            tokenizer=tokenizer,
            results=results,
            args=cfg.diagnostics,
            output_dir=diag_dir,
            run_id=cfg.diagnostics.run_id or cfg.name,
        )

The function is safe to call when ``results`` is missing the ``samples``
key, when no tokenizer is available, and when the model has no STEM
layers — it simply records what is available and notes the rest.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from lingua.diagnostic_records import (
    LayerPathMetricRecord,
    append_jsonl,
    classify_task_group,
    classify_token_role,
    make_sample_id,
    write_json_atomic,
)
from lingua.diagnostics import (
    BASELINE_COMPARISON_CKA_JSON,
    GEOMETRY_BY_FREQUENCY_BUCKET_JSON,
    GEOMETRY_BY_TASK_LAYER_ROLE_JSON,
    RICHER_GEOMETRY_SUMMARY_JSON,
    StemDiagnosticsCollector,
    _iter_all_ffn_layers,
)
from lingua.eval_sample_capture import (
    iter_lm_eval_samples,
    _extract_generation,
    _extract_prompt,
    _extract_target,
    _safe_doc_id,
)

logger = logging.getLogger(__name__)


LAYER_PATH_METRICS_JSONL = "layer_path_metrics.jsonl"
EVAL_ACTIVATION_SUMMARY = "eval_activation_summary.json"
EVAL_GEOMETRY_SUMMARY = "eval_geometry_summary.json"
CONSOLIDATE_FOLDER = "consolidated"


# ---------------------------------------------------------------------------
# KV-cache neutralisation
# ---------------------------------------------------------------------------

@contextmanager
def disable_kv_cache(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily detach any ``kv_cache`` attribute from attention modules.

    The generator path attaches a ``kv_cache`` to each attention block so
    decode-time forwards reuse the cache.  For the diagnostic forward pass
    we need a clean prefill-style invocation; if we don't strip the
    attribute the attention path will try to ``update`` the cache with a
    ``tok_idx=None`` and crash, or worse, append to stale state.  This
    context manager restores everything on exit.
    """
    saved: List[Tuple[torch.nn.Module, Any]] = []
    for module in model.modules():
        if hasattr(module, "kv_cache"):
            saved.append((module, module.kv_cache))
            try:
                delattr(module, "kv_cache")
            except AttributeError:
                pass
    try:
        yield
    finally:
        for module, cache in saved:
            module.kv_cache = cache


# ---------------------------------------------------------------------------
# Tokenisation helpers (best-effort)
# ---------------------------------------------------------------------------

def _safe_encode(tokenizer: Any, text: str, *, add_bos: bool = True) -> Optional[List[int]]:
    if tokenizer is None or text is None:
        return None
    try:
        ids = tokenizer.encode(text, add_bos=add_bos, add_eos=False)
    except TypeError:
        try:
            ids = tokenizer.encode(text)
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(ids, list):
        try:
            ids = list(ids)
        except Exception:
            return None
    return ids


def _safe_decode_tokens(tokenizer: Any, ids: List[int]) -> Optional[List[str]]:
    if tokenizer is None or not ids:
        return None
    try:
        if hasattr(tokenizer, "get_token_offsets"):
            text = tokenizer.decode(ids)
            substrs, _ = tokenizer.get_token_offsets(text, tokens=ids)
            if isinstance(substrs, list) and len(substrs) >= len(ids):
                return substrs[: len(ids)]
    except Exception:
        pass
    try:
        return [tokenizer.decode([tid]) for tid in ids]
    except Exception:
        return None


def _build_input_ids(
    tokenizer: Any,
    prompt: Optional[str],
    completion: Optional[str],
    *,
    max_seqlen: int,
    max_tokens_per_sample: int,
) -> Tuple[Optional[torch.Tensor], Optional[List[int]], Optional[List[str]], Optional[List[str]], Optional[int]]:
    """Encode prompt + completion and return aligned token info.

    Returns
    -------
    (input_ids, token_ids_list, tokens, roles, prompt_len)
    where ``prompt_len`` is the number of token ids in the prompt portion
    (used for marking ``token_role == "prompt"`` vs ``"completion"`` in
    metadata).  Returns ``(None, ...)`` when no usable text/tokens exist.
    """
    if tokenizer is None:
        return None, None, None, None, None

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    if prompt is not None and prompt:
        ids = _safe_encode(tokenizer, prompt, add_bos=True) or []
        prompt_ids = list(ids)
    if completion is not None and completion:
        # Don't add a second BOS for the completion.
        ids = _safe_encode(tokenizer, completion, add_bos=False) or []
        completion_ids = list(ids)

    full = prompt_ids + completion_ids
    if not full:
        return None, None, None, None, None

    cap = max(1, min(int(max_seqlen), int(max_tokens_per_sample)))
    if len(full) > cap:
        # Prefer keeping the tail (completion) intact.
        keep_completion = min(len(completion_ids), max(1, cap // 2))
        keep_prompt = max(0, cap - keep_completion)
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        completion_ids = completion_ids[:keep_completion]
        full = prompt_ids + completion_ids

    if not full:
        return None, None, None, None, None

    tokens_list = _safe_decode_tokens(tokenizer, full)
    roles: Optional[List[str]] = None
    if tokens_list is not None:
        roles = [classify_token_role(t) for t in tokens_list]
    input_ids = torch.tensor([full], dtype=torch.long)
    return input_ids, full, tokens_list, roles, len(prompt_ids)


# ---------------------------------------------------------------------------
# Distributed safety helpers
# ---------------------------------------------------------------------------

def _is_rank0() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _resolve_consolidated_reference_path(path: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Return a consolidated checkpoint path or a non-fatal skip reason."""
    if not path.exists():
        return None, "reference_checkpoint_missing"
    if (path / "params.json").exists():
        return path, None
    consolidated = path / CONSOLIDATE_FOLDER
    if (consolidated / "params.json").exists():
        return consolidated, None
    return None, "reference_checkpoint_not_consolidated"


def load_reference_model_for_geometry(
    args: Any,
) -> Tuple[Optional[torch.nn.Module], Optional[Any], Dict[str, Any]]:
    """Best-effort reference model loader for baseline CKA/SVCCA diagnostics.

    Missing paths, unconsolidated checkpoints, unsupported model types, CPU-only
    hosts, and loader failures all return ``(None, None, status)`` rather than
    raising.  The caller writes the status into ``baseline_comparison_cka.json``.
    """
    ref_path_raw = getattr(args, "reference_checkpoint_path", None)
    status: Dict[str, Any] = {
        "requested": bool(ref_path_raw),
        "reference_checkpoint_path": ref_path_raw,
    }
    if not ref_path_raw:
        status.update({"skipped": True, "reason": "reference_checkpoint_path not provided"})
        return None, None, status

    ref_path = Path(str(ref_path_raw)).expanduser()
    consolidated_path, reason = _resolve_consolidated_reference_path(ref_path)
    if consolidated_path is None:
        status.update({"skipped": True, "reason": reason})
        return None, None, status

    try:
        from omegaconf import OmegaConf
        ckpt_cfg = OmegaConf.load(consolidated_path / "params.json")
        ckpt_model_type = str(getattr(ckpt_cfg, "model_type", "llama"))
    except Exception:
        ckpt_model_type = "llama"
    model_type = getattr(args, "reference_model_type", None) or ckpt_model_type
    status.update(
        {
            "consolidated_path": str(consolidated_path),
            "reference_model_type": model_type,
        }
    )

    try:
        from apps.main.qwen3 import Qwen3LMTransformer, Qwen3LMTransformerArgs
        from apps.main.olmo3 import OLMo3LMTransformer, OLMo3LMTransformerArgs
        from apps.main.transformer import LMTransformer, LMTransformerArgs

        dense_registry = {
            "llama": (LMTransformer, LMTransformerArgs),
            "qwen3": (Qwen3LMTransformer, Qwen3LMTransformerArgs),
            "olmo3": (OLMo3LMTransformer, OLMo3LMTransformerArgs),
        }

        stem_registry: Dict[str, Any] = {}
        try:
            from apps.main.stem import STEM_MODEL_REGISTRY
            from apps.main.stem_dag import DAG_STEM_MODEL_REGISTRY
            stem_registry.update(STEM_MODEL_REGISTRY)
            stem_registry.update(DAG_STEM_MODEL_REGISTRY)
        except Exception:
            stem_registry = {}

        if model_type in stem_registry:
            from apps.main.stem_generate import load_consolidated_model_and_tokenizer
            model_cls, model_args_cls = stem_registry[model_type][:2]
            model, tokenizer, _cfg = load_consolidated_model_and_tokenizer(
                str(consolidated_path),
                model_cls=model_cls,
                model_args_cls=model_args_cls,
            )
            status.update({"loaded": True, "model_family": "stem"})
            return model, tokenizer, status

        if model_type not in dense_registry:
            status.update(
                {
                    "skipped": True,
                    "reason": "unsupported_reference_model_type",
                    "available_dense_model_types": sorted(dense_registry),
                    "available_stem_model_types": sorted(stem_registry),
                }
            )
            return None, None, status

        from apps.main.generate import load_consolidated_model_and_tokenizer
        model_cls, model_args_cls = dense_registry[model_type]
        model, tokenizer, _cfg = load_consolidated_model_and_tokenizer(
            str(consolidated_path),
            model_cls=model_cls,
            model_args_cls=model_args_cls,
        )
        status.update({"loaded": True, "model_family": "dense"})
        return model, tokenizer, status
    except Exception as exc:
        status.update(
            {
                "skipped": True,
                "reason": "reference_load_failed",
                "error": str(exc),
            }
        )
        return None, None, status


# ---------------------------------------------------------------------------
# Top-level capture
# ---------------------------------------------------------------------------

def _select_completion(sample: Dict[str, Any]) -> Optional[str]:
    """Pick the completion text used for the diagnostic forward pass.

    Preference order: model generation (tokens the model actually
    produced), then gold target.  Either gives us the same shape the
    metric was computed on for generate-until / loglikelihood tasks.
    """
    gen = _extract_generation(sample)
    if isinstance(gen, str) and gen:
        return gen
    target = _extract_target(sample)
    if isinstance(target, str) and target:
        return target
    return None


def capture_eval_activations(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    results: Optional[Dict[str, Any]],
    args: Any,                 # DiagnosticsArgs
    output_dir: Path,
    run_id: str,
    rank: Optional[int] = None,
    model_max_seqlen: Optional[int] = None,
) -> Dict[str, Any]:
    """Run forward-only activation diagnostics on lm-eval samples.

    Parameters
    ----------
    model:
        The eval-time model.  Must accept ``model(input_ids)`` for a
        single-prompt forward pass.  STEM / DAG variants are supported
        natively.
    tokenizer:
        The tokenizer used by the eval harness.  When unavailable we skip
        capture and return an empty summary with a metadata note.
    results:
        The dict returned by lm-eval's ``simple_evaluate``.  Must contain
        ``samples`` with ``log_samples=True`` for capture to do anything.
    args:
        :class:`DiagnosticsArgs`-shaped object.
    output_dir, run_id:
        Where artifacts are written and the run identifier propagated
        into every record.

    Returns
    -------
    dict
        The summary dict that was written to
        ``eval_activation_summary.json``.  Empty when the feature is
        disabled or no samples were processed.
    """
    enabled = bool(getattr(args, "enabled", False))
    activations_on = bool(getattr(args, "collect_eval_activations", False)) or bool(
        getattr(args, "collect_richer_geometry", False)
    )
    if not enabled or not activations_on:
        return {}
    if results is None:
        logger.info("diagnostics: eval activations skipped — no lm-eval results")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank0_only = bool(getattr(args, "rank0_only", True))
    is_rank0 = (rank is None) or (rank == 0)
    if rank0_only and not is_rank0:
        # Distributed safety: only rank 0 writes and runs the extra forward
        # passes.  Avoid duplicate compute on every rank.
        return {}

    if tokenizer is None:
        logger.warning(
            "diagnostics: eval activations skipped — tokenizer is None"
        )
        return {"missing_fields": ["no_tokenizer_available"]}

    # Determine where the model expects to live.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve max sequence length.
    if model_max_seqlen is None:
        model_max_seqlen = getattr(model, "max_seqlen", None)
        if model_max_seqlen is None:
            lm = getattr(model, "lm_transformer", None)
            model_max_seqlen = getattr(lm, "max_seqlen", 2048) if lm is not None else 2048
    max_tokens_per_sample = int(getattr(args, "eval_activation_max_tokens_per_sample", 256))

    # Build collector in eval mode and register forward hooks.
    collector = StemDiagnosticsCollector(
        model,
        args,
        output_dir=output_dir,
        prefix="diag/eval",
        mode="eval",
        run_id=run_id,
    )
    collector.register()

    reference_model: Optional[torch.nn.Module] = None
    reference_collector: Optional[StemDiagnosticsCollector] = None
    reference_status: Dict[str, Any] = {}
    reference_failures = 0
    if bool(getattr(args, "collect_richer_geometry", False)) and getattr(args, "reference_checkpoint_path", None):
        reference_model, _reference_tokenizer, reference_status = load_reference_model_for_geometry(args)
        if reference_model is not None:
            reference_collector = StemDiagnosticsCollector(
                reference_model,
                args,
                output_dir=output_dir,
                prefix="diag/reference",
                mode="eval",
                run_id=f"{run_id}:reference",
            )
            reference_collector.register()

    tasks_filter = getattr(args, "tasks", None)
    max_per_task = getattr(args, "eval_activation_max_samples_per_task", None)
    write_records = bool(getattr(args, "write_layer_path_records", False))

    layer_path_records: List[LayerPathMetricRecord] = []
    samples_per_task: Dict[str, int] = {}
    samples_skipped: int = 0
    samples_failed: int = 0
    samples_processed: int = 0

    was_training = model.training
    model.eval()
    ref_was_training = bool(reference_model.training) if reference_model is not None else False
    if reference_model is not None:
        reference_model.eval()

    def _clear_collector_context(c: StemDiagnosticsCollector) -> None:
        c._active = False
        c.current_sample_records = []
        c.current_sample_id = None
        c.current_task = None
        c.current_task_group = None
        c.current_token_ids_list = None
        c.current_tokens_list = None
        c.current_token_roles = None
        c.current_include_richer_geometry = False

    try:
        with ExitStack() as stack:
            stack.enter_context(disable_kv_cache(model))
            if reference_model is not None:
                stack.enter_context(disable_kv_cache(reference_model))
            for task_name, sample in iter_lm_eval_samples(
                results, tasks_filter=tasks_filter, max_per_task=max_per_task
            ):
                # Per-task sample cap is enforced redundantly inside the
                # collector so we still bail early if iter_lm_eval_samples
                # didn't filter (e.g. when ``max_per_task`` is None).
                if (
                    max_per_task is not None
                    and samples_per_task.get(task_name, 0) >= max_per_task
                ):
                    continue

                prompt = _extract_prompt(sample)
                completion = _select_completion(sample)
                target = _extract_target(sample)
                doc_id = _safe_doc_id(sample)
                # Keep the join key identical to eval_sample_capture.  The
                # diagnostic forward may use the model generation when
                # available, but downstream joins key records by prompt/target.
                sample_id = make_sample_id(task_name, prompt, target, doc_idx=doc_id)
                task_group = classify_task_group(task_name)

                input_ids, tok_ids, tok_strs, tok_roles, prompt_len = _build_input_ids(
                    tokenizer,
                    prompt,
                    completion,
                    max_seqlen=int(model_max_seqlen),
                    max_tokens_per_sample=max_tokens_per_sample,
                )
                if input_ids is None:
                    samples_skipped += 1
                    continue

                input_ids = input_ids.to(device)

                collector.set_sample_context(
                    task=task_name,
                    sample_id=sample_id,
                    task_group=task_group,
                    token_ids=tok_ids,
                    tokens=tok_strs,
                    token_roles=tok_roles,
                )
                if not collector._active:
                    # Sample budget exceeded inside the collector.
                    continue

                try:
                    with torch.no_grad():
                        _ = model(input_ids)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning(
                        "diagnostics: forward pass failed for task=%s sample=%s: %s",
                        task_name, sample_id, exc,
                    )
                    samples_failed += 1
                    _clear_collector_context(collector)
                    continue

                include_reference = (
                    reference_model is not None
                    and reference_collector is not None
                    and collector.current_include_richer_geometry
                )
                if include_reference:
                    reference_collector.set_sample_context(
                        task=task_name,
                        sample_id=sample_id,
                        task_group=task_group,
                        token_ids=tok_ids,
                        tokens=tok_strs,
                        token_roles=tok_roles,
                    )
                    if reference_collector._active:
                        try:
                            ref_device = next(reference_model.parameters()).device
                            ref_input_ids = input_ids.to(ref_device)
                            with torch.no_grad():
                                _ = reference_model(ref_input_ids)
                            reference_collector.flush_sample()
                        except Exception as exc:  # pragma: no cover — defensive
                            logger.warning(
                                "diagnostics: reference forward failed for task=%s sample=%s: %s",
                                task_name,
                                sample_id,
                                exc,
                            )
                            reference_failures += 1
                            _clear_collector_context(reference_collector)

                sample_records = collector.flush_sample()
                if write_records:
                    # Tag each record with phase metadata so downstream
                    # analyses can split prompt vs completion contributions.
                    for rec in sample_records:
                        if (
                            prompt_len is not None
                            and rec.token_position is not None
                        ):
                            phase = "prompt" if rec.token_position < prompt_len else "completion"
                            rec.metadata = dict(rec.metadata or {})
                            rec.metadata["phase"] = phase
                    layer_path_records.extend(sample_records)
                samples_per_task[task_name] = samples_per_task.get(task_name, 0) + 1
                samples_processed += 1
    finally:
        collector.close()
        if reference_collector is not None:
            reference_collector.close()
        if was_training:
            model.train()
        if reference_model is not None and ref_was_training:
            reference_model.train()

    # Write artifacts.  We always go through diagnostic_records' helpers
    # which gate on rank-0.
    if reference_status:
        reference_status["forward_failures"] = int(reference_failures)
    summary = collector.write_eval_artifacts(
        output_dir=output_dir,
        layer_path_records=layer_path_records if write_records else None,
        reference_collector=reference_collector,
        reference_status=reference_status,
    )
    if isinstance(summary, dict):
        summary.setdefault("samples_processed", samples_processed)
        summary.setdefault("samples_per_task", samples_per_task)
        summary.setdefault("samples_skipped_no_text", samples_skipped)
        summary.setdefault("samples_failed", samples_failed)
        summary.setdefault("model_max_seqlen", int(model_max_seqlen))
        summary.setdefault(
            "max_tokens_per_sample", int(max_tokens_per_sample)
        )
        # Re-write summary so the operational counters are durable.
        write_json_atomic(output_dir / EVAL_ACTIVATION_SUMMARY, summary)

    logger.info(
        "diagnostics: eval activations done — processed=%d skipped=%d failed=%d "
        "tasks=%d output=%s",
        samples_processed,
        samples_skipped,
        samples_failed,
        len(samples_per_task),
        output_dir,
    )
    return summary if isinstance(summary, dict) else {}


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "BASELINE_COMPARISON_CKA_JSON",
    "LAYER_PATH_METRICS_JSONL",
    "EVAL_ACTIVATION_SUMMARY",
    "EVAL_GEOMETRY_SUMMARY",
    "GEOMETRY_BY_FREQUENCY_BUCKET_JSON",
    "GEOMETRY_BY_TASK_LAYER_ROLE_JSON",
    "RICHER_GEOMETRY_SUMMARY_JSON",
    "capture_eval_activations",
    "disable_kv_cache",
    "load_reference_model_for_geometry",
]
