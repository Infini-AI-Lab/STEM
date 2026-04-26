"""Shared schema, serialisation utilities, and lightweight classifiers for the
task-aligned STEM diagnostics pipeline.

Nothing in this module changes model behaviour or default training/eval flow.
All record types are plain dataclasses that serialise to JSON-safe dicts.  The
IO helpers are safe to call from any rank but only actually write from rank 0.

Typical import pattern::

    from lingua.diagnostic_records import (
        DiagnosticSampleRecord,
        LayerPathMetricRecord,
        InterventionRecord,
        TokenEffectRecord,
        CodeFailureRecord,
        DebuggabilityRecord,
        append_jsonl,
        read_jsonl,
        write_json_atomic,
        sanitize_for_json,
        make_sample_id,
        classify_token_role,
        classify_task_group,
    )
"""

from __future__ import annotations

import hashlib
import json
import keyword
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Record dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticSampleRecord:
    """One sample's worth of eval/generation diagnostics.

    Produced during eval; can be written to a JSONL file and later joined with
    :class:`LayerPathMetricRecord` rows on ``(run_id, task, sample_id)``.
    """

    run_id: str
    task: str
    sample_id: str

    model_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    task_group: Optional[str] = None          # reasoning, math, code, commonsense, …
    doc_id: Optional[str] = None
    split: Optional[str] = None

    prompt: Optional[str] = None
    prompt_truncated: bool = False
    target: Optional[str] = None
    generation: Optional[str] = None

    correct: Optional[bool] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None

    baseline_loss: Optional[float] = None
    model_loss: Optional[float] = None
    per_token_nll: Optional[List[float]] = None

    token_ids: Optional[List[int]] = None
    tokens: Optional[List[str]] = None
    token_roles: Optional[List[str]] = None
    sequence_length: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerPathMetricRecord:
    """Per-layer, optionally per-token-position, path norms and geometry.

    Produced by forward-hook collection in :class:`lingua.diagnostics.StemDiagnosticsCollector`
    or equivalent eval-time collection.  Join with :class:`DiagnosticSampleRecord`
    on ``(run_id, task, sample_id)``.
    """

    run_id: str
    task: str
    sample_id: str
    layer_idx: int

    token_position: Optional[int] = None
    token_id: Optional[int] = None
    token: Optional[str] = None
    token_role: Optional[str] = None

    stem_norm: Optional[float] = None
    up_norm: Optional[float] = None
    combined_norm: Optional[float] = None
    stem_up_cos: Optional[float] = None
    ffn_out_norm: Optional[float] = None
    gate_alpha: Optional[float] = None
    w1_act_norm: Optional[float] = None
    silu_act_norm: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionRecord:
    """Result of a single causal-path intervention (ablation / replacement / gate forcing).

    Produced by :func:`lingua.diagnostics.run_intervention_suite` and related
    helpers.  Richer than the existing flat row dicts; preserves per-token NLL
    deltas and path-relation classification.
    """

    run_id: str
    task: str
    sample_id: str

    intervention_name: str
    intervention_type: str   # ablate_stem | ablate_up | ablate_layer | force_gate | replace_mean | …
    target_path: str         # stem | up | gate | combined | layer

    loss_original: float
    loss_intervened: float
    delta_loss: float

    layer_idx: Optional[int] = None
    delta_per_token_nll: Optional[List[float]] = None
    token_ids: Optional[List[int]] = None
    token_roles: Optional[List[str]] = None
    path_relation: Optional[str] = None  # cooperative | redundant | destructive | dominant | unknown

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenEffectRecord:
    """Aggregated per-token-id effect statistics across a dataset or eval split.

    Produced by :class:`lingua.diagnostics.TokenStatsAggregator` or offline
    aggregation of :class:`DiagnosticSampleRecord` per-token lists.
    """

    run_id: str
    task: str
    token_id: int
    token: str
    token_role: str
    task_group: Optional[str] = None

    sample_id: Optional[str] = None
    layer_idx: Optional[int] = None
    frequency_bucket: Optional[str] = None
    count: int = 0

    stem_activation_norm_mean: Optional[float] = None
    up_activation_norm_mean: Optional[float] = None
    stem_update_norm: Optional[float] = None
    stem_cosine_drift: Optional[float] = None

    stem_ablation_delta_loss: Optional[float] = None
    up_ablation_delta_loss: Optional[float] = None
    combined_ablation_delta_loss: Optional[float] = None

    benefit_score: Optional[float] = None
    harm_score: Optional[float] = None
    ineffective_score: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeFailureRecord:
    """Taxonomy of a code-generation failure for one sample.

    Produced by :func:`lingua.diagnostics.analyze_code_failure` or offline
    analysis of :class:`DiagnosticSampleRecord` generations.
    """

    run_id: str
    task: str
    sample_id: str

    failure_category: str = "unknown"   # syntax_parse_error | indentation_formatting_error | …
    parse_ok: bool = False
    syntax_error: bool = False
    runtime_error: bool = False
    signature_error: bool = False
    test_failure: bool = False
    timeout: bool = False

    traceback_type: Optional[str] = None
    function_name_expected: Optional[str] = None
    function_name_found: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebuggabilityRecord:
    """High-level debuggability classification for a sample or task.

    Answers: is this failure likely debuggable (path/gate issue), possibly
    architectural (capability ceiling), or inconclusive?
    """

    run_id: str
    task: str

    classification: str          # likely_debuggable | possibly_architectural | inconclusive
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0

    sample_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _is_rank0() -> bool:
    """Return True when we should write to disk.

    If ``torch.distributed`` is initialised we only write from rank 0.  When
    it is not initialised (single-process) we always write.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-serialisable value.

    Handles:

    * ``torch.Tensor`` (detached, cast to float32, converted to Python scalars
      or lists; bfloat16/float16 are upcast first)
    * ``numpy.ndarray`` (same treatment)
    * ``float`` NaN / ±Inf → ``None``
    * ``pathlib.Path`` → ``str``
    * ``dict``, ``list``, ``tuple`` are traversed recursively
    """
    # Lazy imports so the module can be imported without torch/numpy installed.
    try:
        import torch
        _torch_available = True
    except ImportError:
        _torch_available = False
    try:
        import numpy as np
        _numpy_available = True
    except ImportError:
        _numpy_available = False

    if _torch_available:
        import torch as _torch
        try:
            from torch.distributed._tensor import DTensor
            if isinstance(obj, DTensor):
                obj = obj.to_local()
        except Exception:
            pass
        if isinstance(obj, _torch.Tensor):
            t = obj.detach()
            if t.dtype in (_torch.bfloat16, _torch.float16):
                t = t.float()
            if t.numel() == 1:
                val = t.item()
                return None if (isinstance(val, float) and not math.isfinite(val)) else val
            return sanitize_for_json(t.cpu().tolist())

    if _numpy_available:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            if obj.dtype in (_np.float16,):
                obj = obj.astype(_np.float32)
            if obj.size == 1:
                val = obj.item()
                return None if (isinstance(val, float) and not math.isfinite(val)) else val
            return sanitize_for_json(obj.tolist())
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            val = float(obj)
            return None if not math.isfinite(val) else val
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)

    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    # int, bool, str, None pass through unchanged
    return obj


def _truncate_text(
    text: Optional[str],
    max_chars: int,
) -> tuple[Optional[str], bool]:
    """Return ``(truncated_text, was_truncated)``."""
    if text is None:
        return None, False
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def record_to_dict(
    record: Any,
    *,
    max_prompt_chars: int = 4096,
    max_generation_chars: int = 4096,
) -> Dict[str, Any]:
    """Convert a record dataclass to a JSON-safe dict.

    Truncates ``prompt`` and ``generation`` fields to *max_prompt_chars* /
    *max_generation_chars* and sets ``prompt_truncated = True`` when prompt
    is trimmed.
    """
    d = asdict(record)
    # Truncate large text fields
    for fld, limit in (("prompt", max_prompt_chars), ("generation", max_generation_chars)):
        if fld in d:
            truncated, was = _truncate_text(d[fld], limit)
            d[fld] = truncated
            if fld == "prompt" and was and "prompt_truncated" in d:
                d["prompt_truncated"] = True
    return sanitize_for_json(d)


# ---------------------------------------------------------------------------
# Stable sample IDs
# ---------------------------------------------------------------------------

def make_sample_id(
    task: str,
    prompt: Optional[str] = None,
    target: Optional[str] = None,
    doc_idx: Optional[int] = None,
) -> str:
    """Return a stable, deterministic hex sample ID.

    The ID is a truncated SHA-256 over the concatenation of *task*, *prompt*,
    *target*, and *doc_idx*.  Callers that have an upstream id should prefer
    that; this is a fallback for cases where none is available.
    """
    parts = [task or "", prompt or "", target or "", str(doc_idx) if doc_idx is not None else ""]
    blob = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# JSONL / JSON IO
# ---------------------------------------------------------------------------

def append_jsonl(
    path: "os.PathLike[str] | str",
    rows: "Sequence[Any]",
    *,
    rank0_only: bool = True,
    max_prompt_chars: int = 4096,
    max_generation_chars: int = 4096,
) -> None:
    """Append *rows* (dicts or record dataclasses) to a JSONL file.

    When *rank0_only* is True (the default) and ``torch.distributed`` is
    initialised this is a no-op on all ranks except rank 0.  Writing is safe
    under concurrent appends within a single process.

    The parent directory is created if it does not exist.
    """
    if not rows:
        return
    if rank0_only and not _is_rank0():
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                row = record_to_dict(
                    row,
                    max_prompt_chars=max_prompt_chars,
                    max_generation_chars=max_generation_chars,
                )
            elif isinstance(row, dict):
                row = sanitize_for_json(row)
            print(json.dumps(row), file=fh)


def read_jsonl(
    path: "os.PathLike[str] | str",
    *,
    skip_errors: bool = True,
) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file and return a list of dicts."""
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not skip_errors:
                    raise
    return rows


def write_json_atomic(
    path: "os.PathLike[str] | str",
    data: Any,
    *,
    rank0_only: bool = True,
    indent: int = 2,
) -> None:
    """Write *data* as JSON to *path* atomically via a ``.tmp`` rename.

    Atomic write prevents readers from seeing a partial file.  On failure the
    ``.tmp`` file is left for inspection.
    """
    if rank0_only and not _is_rank0():
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    safe_data = sanitize_for_json(data)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(safe_data, fh, indent=indent)
    tmp.rename(p)


# ---------------------------------------------------------------------------
# Token-role classifier
# ---------------------------------------------------------------------------

# Python keywords as a frozenset for O(1) lookup.
_PYTHON_KEYWORDS: frozenset[str] = frozenset(keyword.kwlist)

# Patterns compiled once at import time.
_RE_NUMERAL = re.compile(
    r"^[+-]?"
    r"("
    r"0[xX][0-9a-fA-F]+"          # hex literal
    r"|0[oO][0-7]+"                # octal literal
    r"|0[bB][01]+"                 # binary literal
    r"|\d+(\.\d*)?([eE][+-]?\d+)?" # decimal / float / scientific
    r"|\.\d+([eE][+-]?\d+)?"       # .5, .5e3
    r")"
    r"[jJ]?$"                      # optional complex suffix
)
_RE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RE_STRING_LIKE = re.compile(r'^[bBfFrRuU]{0,2}["\']')


def classify_token_role(token: str) -> str:
    """Classify *token* into a coarse syntactic role.

    The classifier is intentionally tokenizer-agnostic and heuristic.  It uses
    only the Python standard library.

    Categories
    ----------
    newline
        The token is exactly ``"\\n"`` or ``"\\\\n"`` (escaped repr).
    whitespace
        The token consists entirely of whitespace characters (excluding
        lone newlines, which are classified as ``newline``).
    python_keyword
        The token is one of Python's reserved keywords.
    identifier
        The token matches ``[A-Za-z_][A-Za-z0-9_]*``.
    numeral
        The token looks like a numeric literal (int, float, hex, …).
    bracket
        One of ``( ) [ ] { }``.
    operator
        Common arithmetic, comparison, or assignment operators.
    punctuation
        ``:  ,  .  ;  @  ->  =>  ...``
    string_like
        The token starts with an optional string prefix followed by a quote
        character (``'`` or ``"``).
    natural_language
        Anything with alphabetic characters not matching the above.
    unknown
        Catch-all.
    """
    # Newline check (before whitespace so "\n" is classified as newline)
    if token in {"\n", "\\n", "\r\n", "\r"}:
        return "newline"
    # Pure whitespace
    if token.strip() == "":
        return "whitespace"
    # Python keyword (exact match on the stripped token)
    stripped = token.strip()
    if stripped in _PYTHON_KEYWORDS:
        return "python_keyword"
    # Numeral
    if _RE_NUMERAL.match(stripped):
        return "numeral"
    # Brackets
    if stripped in {"(", ")", "[", "]", "{", "}"}:
        return "bracket"
    # Operators
    if stripped in {
        "+", "-", "*", "**", "/", "//", "%", "@",
        "=", "==", "!=", "<", ">", "<=", ">=",
        "+=", "-=", "*=", "/=", "//=", "%=", "**=", "@=",
        "&", "|", "^", "~", "<<", ">>",
        "&=", "|=", "^=", "<<=", ">>=",
        "->", "=>", ":=",
    }:
        return "operator"
    # Punctuation
    if stripped in {":", ",", ".", ";", "...", "\\"}:
        return "punctuation"
    # String-like fragment (starts with quote, possibly with prefix)
    if _RE_STRING_LIKE.match(stripped):
        return "string_like"
    # Identifier
    if _RE_IDENTIFIER.match(stripped):
        return "identifier"
    # Natural language: contains alphabetic characters
    if any(ch.isalpha() for ch in stripped):
        return "natural_language"
    return "unknown"


# ---------------------------------------------------------------------------
# Task-group classifier
# ---------------------------------------------------------------------------

# Ordered list of (group_name, substring_keywords).  First match wins.
_TASK_GROUP_RULES: List[tuple[str, tuple[str, ...]]] = [
    ("code", ("mbpp", "humaneval", "humaneval_plus", "apps", "code", "python")),
    ("math", ("gsm8k", "math", "minerva", "amc", "aime")),
    ("knowledge_reasoning", ("mmlu", "bbh", "triviaqa", "naturalqa", "nq_open")),
    ("commonsense", ("arc", "hellaswag", "piqa", "winogrande", "boolq", "openbookqa", "race")),
]


def classify_task_group(task_name: str) -> str:
    """Map a task name to a coarse task group.

    Groups
    ------
    code
        ``mbpp``, ``humaneval``, ``humaneval_plus``, ``apps``, ``code``, ``python``
    math
        ``gsm8k``, ``math``, ``minerva``, ``amc``, ``aime``
    knowledge_reasoning
        ``mmlu``, ``bbh``, ``triviaqa``, ``naturalqa``, ``nq_open``
    commonsense
        ``arc``, ``hellaswag``, ``piqa``, ``winogrande``, ``boolq``,
        ``openbookqa``, ``race``
    other
        Anything else.
    """
    lower = task_name.lower()
    for group, keywords in _TASK_GROUP_RULES:
        if any(kw in lower for kw in keywords):
            return group
    return "other"
