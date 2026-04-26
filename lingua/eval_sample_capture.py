"""Task-aligned lm-eval sample capture.

Round 2 of the unified STEM diagnostics pipeline.  This module is purely
additive: it converts lm-eval ``results["samples"]`` into
:class:`lingua.diagnostic_records.DiagnosticSampleRecord` rows and writes them
as JSONL alongside a small summary JSON.

It does *not* run hooks, interventions, or extra forward passes.  Tokenisation
of prompts/generations is best-effort and bounded; if the tokenizer is
unavailable or fails, ``token_ids`` / ``tokens`` / ``token_roles`` are left
``None`` and a metadata note is recorded.

Typical wire-up (rank-0 path)::

    from lingua.eval_sample_capture import capture_eval_samples
    if cfg.diagnostics.enabled and cfg.diagnostics.collect_eval_samples:
        capture_eval_samples(
            results=results,
            args=cfg.diagnostics,
            output_dir=diag_dir,
            run_id=cfg.diagnostics.run_id or cfg.name,
            checkpoint_path=cfg.ckpt_dir,
            model_id=cfg.model_type,
            tokenizer=tokenizer,
        )
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from lingua.diagnostic_records import (
    DiagnosticSampleRecord,
    append_jsonl,
    classify_task_group,
    classify_token_role,
    make_sample_id,
    read_jsonl,
    record_to_dict,
    sanitize_for_json,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


# Output filenames live next to other diagnostics artifacts.
EVAL_SAMPLES_JSONL = "diagnostics_eval_samples.jsonl"
EVAL_SAMPLES_SHARD_TEMPLATE = "diagnostics_eval_samples.shard{rank:02d}.jsonl"
EVAL_SAMPLE_SUMMARY = "eval_sample_summary.json"


# ---------------------------------------------------------------------------
# Helpers — extracting fields from an lm-eval sample dict
# ---------------------------------------------------------------------------

def _is_correct(sample: Dict[str, Any]) -> Optional[bool]:
    """Best-effort correctness extraction.

    Tries common metric keys (``acc``, ``acc_norm``, ``exact_match``,
    ``pass@1``).  Returns None when no recognised metric is present.
    """
    for key in ("acc", "acc_norm", "exact_match", "pass@1", "f1", "rouge1"):
        v = sample.get(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v >= 0.5)
    return None


def _extract_metric(sample: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    """Pick a single (metric_name, metric_value) pair for the record."""
    metric_keys = sample.get("metrics") or []
    preferred = ["pass@1", "exact_match", "acc_norm", "acc", "f1", "rougeL", "rouge1"]
    candidates: List[str] = []
    if isinstance(metric_keys, list):
        candidates.extend([k for k in metric_keys if isinstance(k, str)])
    for k in preferred:
        if k in candidates and isinstance(sample.get(k), (int, float, bool)):
            return k, float(sample[k])
    for k in candidates:
        if isinstance(sample.get(k), (int, float, bool)):
            return k, float(sample[k])
    # Fall back to any common metric value present
    for k in preferred:
        if isinstance(sample.get(k), (int, float, bool)):
            return k, float(sample[k])
    return None, None


def _extract_prompt(sample: Dict[str, Any]) -> Optional[str]:
    """Pull the prompt text from ``arguments`` / ``doc`` / known keys."""
    args = sample.get("arguments")
    if isinstance(args, list) and args:
        first = args[0]
        if isinstance(first, (list, tuple)) and first:
            cand = first[0]
            if isinstance(cand, str):
                return cand
        elif isinstance(first, str):
            return first
    doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
    for key in ("prompt", "text", "question", "input", "passage"):
        v = doc.get(key)
        if isinstance(v, str):
            return v
    return None


def _extract_target(sample: Dict[str, Any]) -> Optional[str]:
    """Pull the gold target."""
    t = sample.get("target")
    if isinstance(t, str):
        return t
    if isinstance(t, (list, tuple)) and t and isinstance(t[0], str):
        return t[0]
    doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
    v = doc.get("target") or doc.get("answer") or doc.get("output")
    if isinstance(v, str):
        return v
    if t is not None:
        try:
            return str(t)
        except Exception:
            return None
    return None


def _extract_generation(sample: Dict[str, Any]) -> Optional[str]:
    """Pull the model generation, preferring the post-filter response."""
    fr = sample.get("filtered_resps")
    if isinstance(fr, list) and fr:
        first = fr[0]
        if isinstance(first, str):
            return first
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
            return first[0]
    rs = sample.get("resps")
    if isinstance(rs, list) and rs:
        first = rs[0]
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
            return first[0]
        if isinstance(first, str):
            return first
    pred = sample.get("prediction")
    if isinstance(pred, str):
        return pred
    return None


def _extract_split(sample: Dict[str, Any]) -> Optional[str]:
    doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
    v = doc.get("split") or sample.get("split")
    if isinstance(v, str):
        return v
    return None


def _safe_doc_id(sample: Dict[str, Any]) -> Optional[str]:
    did = sample.get("doc_id")
    if did is None:
        return None
    try:
        return str(did)
    except Exception:
        return None


def _safe_metadata(sample: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pick a small, safe-to-serialise subset of the sample dict.

    Avoids including huge ``doc`` blobs (image bytes, very long passages); we
    only retain hashes and small bookkeeping keys.  Anything passed via
    *extra* is merged in unchanged (still sanitised at write time).
    """
    keep_keys = (
        "doc_hash",
        "prompt_hash",
        "target_hash",
        "filter",
        "metrics",
    )
    md: Dict[str, Any] = {}
    for k in keep_keys:
        if k in sample:
            md[k] = sample[k]
    if extra:
        md.update(extra)
    return md


# ---------------------------------------------------------------------------
# Tokenisation (best-effort)
# ---------------------------------------------------------------------------

def _try_tokenize(
    text: Optional[str],
    tokenizer: Any,
    *,
    max_tokens: int = 2048,
) -> tuple[Optional[List[int]], Optional[List[str]], Optional[List[str]]]:
    """Best-effort encode + role classification.

    Returns (token_ids, tokens, token_roles) or (None, None, None) on failure.
    """
    if text is None or tokenizer is None:
        return None, None, None
    try:
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    except TypeError:
        try:
            ids = tokenizer.encode(text)
        except Exception:
            return None, None, None
    except Exception:
        return None, None, None
    if not isinstance(ids, list):
        try:
            ids = list(ids)
        except Exception:
            return None, None, None
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    # Per-token decoded strings — prefer get_token_offsets when available
    tokens: Optional[List[str]] = None
    try:
        if hasattr(tokenizer, "get_token_offsets"):
            substrs, _offsets = tokenizer.get_token_offsets(text, tokens=ids)
            if isinstance(substrs, list) and len(substrs) >= len(ids):
                tokens = substrs[: len(ids)]
    except Exception:
        tokens = None
    if tokens is None:
        try:
            tokens = [tokenizer.decode([tid]) for tid in ids]
        except Exception:
            tokens = None
    roles: Optional[List[str]] = None
    if tokens is not None:
        try:
            roles = [classify_token_role(t) for t in tokens]
        except Exception:
            roles = None
    return ids, tokens, roles


# ---------------------------------------------------------------------------
# Adapter — convert lm-eval sample → DiagnosticSampleRecord
# ---------------------------------------------------------------------------

def lm_eval_sample_to_record(
    task: str,
    sample: Dict[str, Any],
    *,
    run_id: str,
    model_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    capture_prompts: bool = True,
    capture_generations: bool = True,
    capture_token_ids: bool = False,
    tokenizer: Any = None,
    max_text_chars: int = 4096,
    max_tokens: int = 2048,
) -> DiagnosticSampleRecord:
    """Convert a single lm-eval sample dict to a :class:`DiagnosticSampleRecord`.

    Missing fields are filled with ``None`` and a note in ``metadata``.  This
    function never raises on a malformed sample (it captures the error in
    ``metadata['conversion_error']``).
    """
    metadata_extra: Dict[str, Any] = {}
    notes: List[str] = []

    try:
        prompt = _extract_prompt(sample) if capture_prompts else None
        target = _extract_target(sample)
        generation = _extract_generation(sample) if capture_generations else None
        correct = _is_correct(sample)
        metric_name, metric_value = _extract_metric(sample)
        doc_id = _safe_doc_id(sample)
        split = _extract_split(sample)

        if prompt is None and capture_prompts:
            notes.append("prompt_unavailable")
        if generation is None and capture_generations:
            notes.append("generation_unavailable")
        if target is None:
            notes.append("target_unavailable")
        if correct is None:
            notes.append("correctness_unavailable")

        sample_id = make_sample_id(task, prompt, target, doc_idx=doc_id)

        token_ids: Optional[List[int]] = None
        tokens: Optional[List[str]] = None
        token_roles: Optional[List[str]] = None
        sequence_length: Optional[int] = None
        if capture_token_ids and tokenizer is not None:
            tokenize_text = None
            if prompt is not None and generation is not None:
                tokenize_text = prompt + generation
            elif prompt is not None and target is not None:
                tokenize_text = prompt + target
            elif prompt is not None:
                tokenize_text = prompt
            elif generation is not None:
                tokenize_text = generation
            if tokenize_text is not None:
                token_ids, tokens, token_roles = _try_tokenize(
                    tokenize_text, tokenizer, max_tokens=max_tokens
                )
                if token_ids is not None:
                    sequence_length = len(token_ids)
                else:
                    notes.append("tokenization_failed")
        elif capture_token_ids and tokenizer is None:
            notes.append("no_tokenizer_available")

        if notes:
            metadata_extra["missing_fields"] = notes

        record = DiagnosticSampleRecord(
            run_id=run_id,
            task=task,
            sample_id=sample_id,
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            task_group=classify_task_group(task),
            doc_id=doc_id,
            split=split,
            prompt=prompt,
            target=target,
            generation=generation,
            correct=correct,
            metric_name=metric_name,
            metric_value=metric_value,
            token_ids=token_ids,
            tokens=tokens,
            token_roles=token_roles,
            sequence_length=sequence_length,
            metadata=_safe_metadata(sample, metadata_extra),
        )
        return record
    except Exception as exc:
        # Never crash the eval pipeline because of diagnostics.
        return DiagnosticSampleRecord(
            run_id=run_id,
            task=task,
            sample_id=make_sample_id(task, str(sample.get("doc_id", "")), doc_idx=sample.get("doc_id")),
            task_group=classify_task_group(task),
            metadata={"conversion_error": str(exc)},
        )


def iter_lm_eval_samples(
    results: Dict[str, Any],
    *,
    tasks_filter: Optional[List[str]] = None,
    max_per_task: Optional[int] = None,
) -> Iterable[tuple[str, Dict[str, Any]]]:
    """Yield ``(task_name, sample_dict)`` pairs from lm-eval results.

    ``results["samples"]`` is expected to map task name → list of sample dicts.
    If ``samples`` is missing or not a dict, yields nothing.
    """
    samples_root = results.get("samples") if isinstance(results, dict) else None
    if not isinstance(samples_root, dict):
        return
    filt = set(t.lower() for t in tasks_filter) if tasks_filter else None
    for task_name, task_samples in samples_root.items():
        if filt is not None and task_name.lower() not in filt:
            continue
        if not isinstance(task_samples, list):
            continue
        if max_per_task is not None:
            task_samples = task_samples[:max_per_task]
        for sample in task_samples:
            if isinstance(sample, dict):
                yield task_name, sample


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_records(records: Iterable[DiagnosticSampleRecord]) -> Dict[str, Any]:
    """Compute a small summary across DiagnosticSampleRecord rows.

    Counts per task, correct/incorrect counts, average sequence length,
    number with generation/token ids, and counts of records that report
    missing fields.
    """
    per_task: Dict[str, Counter] = defaultdict(Counter)
    seqlen_sum: Dict[str, int] = defaultdict(int)
    seqlen_count: Dict[str, int] = defaultdict(int)
    total = 0
    correct = 0
    incorrect = 0
    unknown_correct = 0
    with_generation = 0
    with_token_ids = 0
    missing_field_counts: Counter = Counter()

    for rec in records:
        total += 1
        per_task[rec.task]["count"] += 1
        if rec.correct is True:
            per_task[rec.task]["correct"] += 1
            correct += 1
        elif rec.correct is False:
            per_task[rec.task]["incorrect"] += 1
            incorrect += 1
        else:
            per_task[rec.task]["correctness_unknown"] += 1
            unknown_correct += 1
        if rec.generation is not None:
            with_generation += 1
            per_task[rec.task]["with_generation"] += 1
        if rec.token_ids is not None:
            with_token_ids += 1
            per_task[rec.task]["with_token_ids"] += 1
        if rec.sequence_length is not None:
            seqlen_sum[rec.task] += int(rec.sequence_length)
            seqlen_count[rec.task] += 1
        missing = rec.metadata.get("missing_fields") if isinstance(rec.metadata, dict) else None
        if isinstance(missing, list):
            for m in missing:
                missing_field_counts[m] += 1

    by_task: Dict[str, Dict[str, Any]] = {}
    for task, counts in per_task.items():
        avg_seqlen = (
            seqlen_sum[task] / seqlen_count[task] if seqlen_count[task] > 0 else None
        )
        by_task[task] = {
            "count": counts["count"],
            "correct": counts.get("correct", 0),
            "incorrect": counts.get("incorrect", 0),
            "correctness_unknown": counts.get("correctness_unknown", 0),
            "with_generation": counts.get("with_generation", 0),
            "with_token_ids": counts.get("with_token_ids", 0),
            "avg_sequence_length": avg_seqlen,
            "task_group": classify_task_group(task),
        }

    return sanitize_for_json(
        {
            "total_records": total,
            "total_correct": correct,
            "total_incorrect": incorrect,
            "total_correctness_unknown": unknown_correct,
            "total_with_generation": with_generation,
            "total_with_token_ids": with_token_ids,
            "missing_field_counts": dict(missing_field_counts),
            "by_task": by_task,
        }
    )


# ---------------------------------------------------------------------------
# Top-level capture entry point
# ---------------------------------------------------------------------------

def _resolve_jsonl_path(output_dir: Path, rank: Optional[int]) -> Path:
    """Single file when rank in {None, 0}; sharded path when rank > 0."""
    if rank is None or rank == 0:
        return output_dir / EVAL_SAMPLES_JSONL
    return output_dir / EVAL_SAMPLES_SHARD_TEMPLATE.format(rank=rank)


def capture_eval_samples(
    *,
    results: Dict[str, Any],
    args: Any,                            # DiagnosticsArgs
    output_dir: Path,
    run_id: str,
    checkpoint_path: Optional[str] = None,
    model_id: Optional[str] = None,
    tokenizer: Any = None,
    rank: Optional[int] = None,           # global rank for shard naming
    write_summary: bool = True,
) -> Dict[str, Any]:
    """Convert lm-eval samples to records, write JSONL, and emit summary.

    Returns a dict containing the summary that was written (or computed even
    when no records were produced).  Safe to call when ``results`` has no
    ``samples`` key — it will simply produce an empty summary.

    Behaviour
    ---------
    * No-op when ``args.enabled`` is False or ``args.collect_eval_samples``
      is False; returns an empty dict.
    * Writes records to ``<output_dir>/diagnostics_eval_samples.jsonl`` on
      rank 0 (or when ``rank`` is None / single-process).  Non-zero ranks
      write to a shard file ``diagnostics_eval_samples.shard{NN}.jsonl``
      so downstream tools can union shards.
    * Summary is written only on rank 0 (or single-process).
    """
    if not getattr(args, "enabled", False) or not getattr(args, "collect_eval_samples", False):
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_filter = getattr(args, "tasks", None)
    max_per_task = getattr(args, "max_eval_samples_per_task", None)
    capture_prompts = getattr(args, "capture_prompts", True)
    capture_generations = getattr(args, "capture_generations", True)
    capture_token_ids = getattr(args, "capture_token_ids", False)
    max_text_chars = getattr(args, "max_text_chars", 4096)
    rank0_only = getattr(args, "rank0_only", True)

    is_rank0 = (rank is None) or (rank == 0)
    # Honour user-requested rank0_only by skipping non-rank-0 writes entirely.
    if rank0_only and not is_rank0:
        return {}

    records: List[DiagnosticSampleRecord] = []
    for task_name, sample in iter_lm_eval_samples(
        results, tasks_filter=tasks_filter, max_per_task=max_per_task
    ):
        rec = lm_eval_sample_to_record(
            task=task_name,
            sample=sample,
            run_id=run_id,
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            capture_prompts=capture_prompts,
            capture_generations=capture_generations,
            capture_token_ids=capture_token_ids,
            tokenizer=tokenizer,
            max_text_chars=max_text_chars,
        )
        records.append(rec)

    jsonl_path = _resolve_jsonl_path(output_dir, rank)
    if records:
        # ``rank0_only=False`` in append_jsonl because we've already gated
        # above; we want the chosen file to actually be written.
        append_jsonl(
            jsonl_path,
            records,
            rank0_only=False,
            max_prompt_chars=max_text_chars,
            max_generation_chars=max_text_chars,
        )
        logger.info(
            "diagnostics: wrote %d eval-sample records → %s", len(records), jsonl_path
        )
    else:
        logger.info(
            "diagnostics: no eval samples found in lm-eval results "
            "(check harness.log_samples=True and that diagnostics.tasks matches a real task)"
        )

    summary: Dict[str, Any] = {}
    if write_summary and is_rank0:
        summary = summarize_records(records)
        summary["run_id"] = run_id
        summary["jsonl_path"] = str(jsonl_path)
        write_json_atomic(
            output_dir / EVAL_SAMPLE_SUMMARY,
            summary,
            rank0_only=False,
        )
        logger.info(
            "diagnostics: wrote eval sample summary → %s",
            output_dir / EVAL_SAMPLE_SUMMARY,
        )
    return summary


# ---------------------------------------------------------------------------
# Round-trip / reuse — feed code-failure analysis from the new JSONL
# ---------------------------------------------------------------------------

def load_eval_sample_records(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all per-sample records from the canonical JSONL plus any shards."""
    output_dir = Path(output_dir)
    rows = read_jsonl(output_dir / EVAL_SAMPLES_JSONL)
    for shard in sorted(output_dir.glob("diagnostics_eval_samples.shard*.jsonl")):
        rows.extend(read_jsonl(shard))
    return rows


def records_to_results_samples_dict(
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Re-shape captured records into the lm-eval ``samples`` dict shape.

    Useful for feeding the existing ``analyze_eval_samples`` flow from the
    new JSONL artifacts.  Each row is wrapped to expose ``filtered_resps``,
    ``target``, and ``doc`` keys so the legacy code paths keep working.
    """
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = row.get("task") or "unknown"
        out[task].append(
            {
                "doc_id": row.get("doc_id"),
                "doc": {"target": row.get("target")},
                "target": row.get("target"),
                "filtered_resps": [row.get("generation") or ""],
                "resps": [[row.get("generation") or ""]],
                "prediction": row.get("generation"),
            }
        )
    return dict(out)
