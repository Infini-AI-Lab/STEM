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
from contextlib import contextmanager
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
from lingua.diagnostics import StemDiagnosticsCollector, _iter_all_ffn_layers
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
    activations_on = bool(getattr(args, "collect_eval_activations", False))
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

    try:
        with disable_kv_cache(model):
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
                doc_id = _safe_doc_id(sample)
                sample_id = make_sample_id(task_name, prompt, completion, doc_idx=doc_id)
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
                    collector._active = False
                    collector.current_sample_records = []
                    collector.current_sample_id = None
                    collector.current_task = None
                    continue

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
        if was_training:
            model.train()

    # Write artifacts.  We always go through diagnostic_records' helpers
    # which gate on rank-0.
    summary = collector.write_eval_artifacts(
        output_dir=output_dir,
        layer_path_records=layer_path_records if write_records else None,
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
    "LAYER_PATH_METRICS_JSONL",
    "EVAL_ACTIVATION_SUMMARY",
    "EVAL_GEOMETRY_SUMMARY",
    "capture_eval_activations",
    "disable_kv_cache",
]
