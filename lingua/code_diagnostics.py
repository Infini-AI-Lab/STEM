"""MBPP/HumanEval-specific causal failure analysis.

Offline, artifact-based.  Does *not* execute generated code.

Connects::

    sample outcome → code failure category → token roles
                   → STEM/up/gate intervention deltas

Entry point::

    from lingua.code_diagnostics import run_code_causal_analysis
    summary = run_code_causal_analysis(
        output_dir=Path("diagnostics"),
        run_id="my_run",
    )

The module reads two JSONL artifacts produced by earlier pipeline rounds:

* ``diagnostics_eval_samples.jsonl`` — one :class:`DiagnosticSampleRecord`
  per line, written by :func:`lingua.eval_sample_capture.capture_eval_samples`.
* ``interventions_task_aligned.jsonl`` — one :class:`InterventionRecord` per
  line, written by
  :func:`lingua.diagnostics.run_task_aligned_interventions`.

Both files are optional; the module gracefully degrades if either is missing.

Static analysis limitations
----------------------------
* ``logic_error_likely`` is inferred when a generation parses without error
  but the harness marks it incorrect (``correct=False``).  It cannot
  distinguish semantic errors from subtle API misuse.
* Wildcard-import detection is a heuristic; non-wildcard bad imports are not
  detected without execution.
* ``signature_error`` compares positional arg count only; keyword-only,
  ``*args``, and ``**kwargs`` parameters are not counted against the expected
  positional count when they appear in the generated code.
* Fenced-code extraction tries ``python`` and ``py`` fences first, then any
  fence.  If no fence is found the whole generation is treated as code.

Interpreting intervention deltas
---------------------------------
Sign convention (inherited from :func:`lingua.diagnostics.run_task_aligned_interventions`):

* ``delta_loss = intervened_loss - original_loss``
* Positive delta: the ablated path *helped* the model (removing it raised loss).
* Negative delta: the ablated path *hurt* the model (removing it lowered loss).

The summary field ``stem_helpful_examples`` lists samples where
``ablate_stem`` produced a *negative* delta_loss — i.e. where STEM was
harmful and removing it improved the prediction.  ``gate_up_helpful_examples``
lists samples where ``force_gate_1`` (pure up-path) produced a lower delta
than ``force_gate_0`` (pure stem-path), indicating the up path was preferable.
"""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lingua.diagnostic_records import (
    CodeFailureRecord,
    append_jsonl,
    read_jsonl,
    record_to_dict,
    sanitize_for_json,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public failure category constants
# ---------------------------------------------------------------------------

CAT_PASS                 = "pass"
CAT_SYNTAX_ERROR         = "syntax_error"
CAT_INDENTATION_ERROR    = "indentation_error"
CAT_UNMATCHED_BRACKET    = "unmatched_bracket_or_quote"
CAT_SIGNATURE_ERROR      = "signature_error"
CAT_MISSING_FUNCTION     = "missing_function"
CAT_WRONG_FUNCTION_NAME  = "wrong_function_name"
CAT_IMPORT_ERROR         = "import_error_or_api_misuse"
CAT_RUNTIME_ERROR        = "runtime_error"
CAT_TIMEOUT              = "timeout"
CAT_TEST_FAILURE         = "test_failure"
CAT_LOGIC_ERROR          = "logic_error_likely"
CAT_PROMPT_NONCOMPLIANCE = "prompt_noncompliance"
CAT_EMPTY_OR_TRUNCATED   = "empty_or_truncated_generation"
CAT_UNKNOWN              = "unknown_failure"

ALL_CATEGORIES: List[str] = [
    CAT_PASS, CAT_SYNTAX_ERROR, CAT_INDENTATION_ERROR, CAT_UNMATCHED_BRACKET,
    CAT_SIGNATURE_ERROR, CAT_MISSING_FUNCTION, CAT_WRONG_FUNCTION_NAME,
    CAT_IMPORT_ERROR, CAT_RUNTIME_ERROR, CAT_TIMEOUT, CAT_TEST_FAILURE,
    CAT_LOGIC_ERROR, CAT_PROMPT_NONCOMPLIANCE, CAT_EMPTY_OR_TRUNCATED,
    CAT_UNKNOWN,
]

# Output file names (relative to output_dir / "diagnostics")
CODE_FAILURES_JSONL         = "code_failures.jsonl"
CODE_CAUSAL_ANALYSIS_JSON   = "code_causal_failure_analysis.json"
CODE_FAILURE_EXAMPLES_JSONL = "code_failure_examples.jsonl"

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for efficiency)
# ---------------------------------------------------------------------------

_RE_FENCED_PYTHON = re.compile(
    r"```(?:python|py)?\s*\n(.*?)(?:```|$)",
    re.DOTALL | re.IGNORECASE,
)
_RE_FENCED_ANY = re.compile(r"```\w*\s*\n(.*?)(?:```|$)", re.DOTALL)
_RE_DEF = re.compile(r"(?:^|\n)\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_RE_PROSE_LEAD = re.compile(
    r"^\s*(?:The\s|Here\s|This\s|I\s|We\s|Sure\s|Below\s|To\s)",
    re.IGNORECASE,
)
_RE_IMPORT_STAR = re.compile(r"\bfrom\s+\w[\w.]*\s+import\s+\*")


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

def extract_fenced_code(text: str) -> Optional[str]:
    """Return the first fenced code block, preferring ```python``` / ```py```.

    Returns ``None`` when no fence is found; the caller should then treat the
    whole text as code.
    """
    m = _RE_FENCED_PYTHON.search(text)
    if m:
        return m.group(1)
    m = _RE_FENCED_ANY.search(text)
    if m:
        return m.group(1)
    return None


def _has_unbalanced_delimiters(text: str) -> bool:
    """Return True when brackets or quotes are unbalanced.

    Handles triple-quoted strings, escape sequences, and ``#`` comments.
    """
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: List[str] = []
    i = 0
    in_string: Optional[str] = None
    triple = False

    while i < len(text):
        ch = text[i]

        # Triple-quote open / close
        if in_string is None and text[i:i + 3] in ('"""', "'''"):
            in_string = text[i:i + 3]
            triple = True
            i += 3
            continue
        if in_string is not None and triple and text[i:i + 3] == in_string:
            in_string = None
            triple = False
            i += 3
            continue

        # Single-char string open / close
        if in_string is None and ch in ('"', "'"):
            in_string = ch
            triple = False
            i += 1
            continue
        if in_string is not None and not triple and ch == in_string:
            in_string = None
            i += 1
            continue

        # Inside a string
        if in_string is not None:
            if ch == "\\" and not triple:
                i += 2
                continue
            i += 1
            continue

        # Comment — skip to end of line
        if ch == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue

        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack[-1] != ch:
                return True
            stack.pop()
        i += 1

    return bool(stack)


def _extract_fn_defs_from_tree(tree: ast.AST) -> List[Tuple[str, int]]:
    """Return (name, positional_arg_count) for every top-level FunctionDef."""
    results: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Count positional + positional-only args; exclude *args, **kwargs,
            # keyword-only args so we compare apples to apples with the prompt.
            nargs = len(node.args.posonlyargs) + len(node.args.args)
            results.append((node.name, nargs))
    return results


def _extract_expected_fn_name(
    prompt: Optional[str],
    target: Optional[str],
) -> Optional[str]:
    """Best-effort: find the expected function name from prompt or target."""
    for text in (prompt, target):
        if not text:
            continue
        m = _RE_DEF.search(text)
        if m:
            return m.group(1)
    return None


def _count_prompt_args(prompt: Optional[str], fn_name: Optional[str]) -> Optional[int]:
    """Count positional parameters in the function stub inside *prompt*."""
    if not prompt or not fn_name:
        return None
    # Find the def for the expected function name
    pat = re.compile(
        r"\bdef\s+" + re.escape(fn_name) + r"\s*\(([^)]*)\)"
    )
    m = pat.search(prompt)
    if not m:
        return None
    params_str = m.group(1).strip()
    if not params_str:
        return 0
    # Split on commas at depth 0
    depth = 0
    commas = 0
    for ch in params_str:
        if ch in ("(", "[", "{"):
            depth += 1
        elif ch in (")", "]", "}"):
            depth -= 1
        elif ch == "," and depth == 0:
            commas += 1
    return commas + 1


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_static_failure(
    generated_text: str,
    *,
    expected_fn_name: Optional[str] = None,
    expected_arg_count: Optional[int] = None,
    prompt: Optional[str] = None,
    correct: Optional[bool] = None,
) -> Dict[str, Any]:
    """Classify a code generation using static analysis only.

    Parameters
    ----------
    generated_text:
        Raw model output; may contain fenced code blocks.
    expected_fn_name:
        Function name the task expects (e.g. from ``def add(`` in prompt).
    expected_arg_count:
        Number of positional parameters expected, if known.
    prompt:
        Original task prompt; used to detect function expectations and
        prompt-noncompliance.
    correct:
        Harness correctness label.  When ``False`` and no structural error is
        found, the category is ``logic_error_likely``.

    Returns
    -------
    dict with keys: ``static_category``, ``parse_ok``, ``syntax_error_flag``,
    ``indentation_error_flag``, ``unmatched_delimiter``, ``missing_function``,
    ``wrong_function_name``, ``signature_error``, ``import_error``,
    ``empty_or_truncated``, ``prompt_noncompliance``,
    ``function_name_found``, ``detail``.
    """
    out: Dict[str, Any] = {
        "static_category": CAT_UNKNOWN,
        "parse_ok": False,
        "syntax_error_flag": False,
        "indentation_error_flag": False,
        "unmatched_delimiter": False,
        "missing_function": False,
        "wrong_function_name": False,
        "signature_error": False,
        "import_error": False,
        "empty_or_truncated": False,
        "prompt_noncompliance": False,
        "function_name_found": None,
        "detail": "",
    }

    text = generated_text or ""
    stripped = text.strip()

    # ------------------------------------------------------------------
    # 1. Empty / truncated
    # ------------------------------------------------------------------
    if len(stripped) < 5:
        out["empty_or_truncated"] = True
        out["static_category"] = CAT_EMPTY_OR_TRUNCATED
        return out

    # ------------------------------------------------------------------
    # 2. Extract code — prefer fenced block, fall back to whole text
    # ------------------------------------------------------------------
    fenced = extract_fenced_code(text)
    code = fenced.strip() if fenced is not None else stripped

    if len(code) < 5:
        out["empty_or_truncated"] = True
        out["static_category"] = CAT_EMPTY_OR_TRUNCATED
        return out

    # ------------------------------------------------------------------
    # 3. Parse with ast
    # ------------------------------------------------------------------
    tree: Optional[ast.AST] = None
    try:
        tree = ast.parse(code)
        out["parse_ok"] = True
    except IndentationError as exc:
        out["indentation_error_flag"] = True
        out["detail"] = str(exc)
        out["static_category"] = CAT_INDENTATION_ERROR
        return out
    except SyntaxError as exc:
        msg = str(exc).lower()
        out["detail"] = str(exc)
        if any(
            kw in msg
            for kw in (
                "eol while scanning",
                "unterminated string",
                "eof while scanning",
                "unexpected eof",
                "unmatched",
                "unbalanced",
            )
        ):
            out["unmatched_delimiter"] = True
            out["static_category"] = CAT_UNMATCHED_BRACKET
        else:
            out["syntax_error_flag"] = True
            out["static_category"] = CAT_SYNTAX_ERROR
        return out

    # ------------------------------------------------------------------
    # 4. Unbalanced delimiter check (handles cases ast.parse may accept)
    # ------------------------------------------------------------------
    if fenced is None and _has_unbalanced_delimiters(stripped):
        out["unmatched_delimiter"] = True
        out["static_category"] = CAT_UNMATCHED_BRACKET
        return out

    # ------------------------------------------------------------------
    # 5. Function definition extraction
    # ------------------------------------------------------------------
    assert tree is not None
    fn_defs = _extract_fn_defs_from_tree(tree)
    if fn_defs:
        out["function_name_found"] = fn_defs[0][0]

    fn_expected = expected_fn_name is not None or (
        isinstance(prompt, str) and "def " in prompt
    )

    # ------------------------------------------------------------------
    # 6. Missing function def (when context implies one is required)
    # ------------------------------------------------------------------
    if fn_expected and not fn_defs:
        out["missing_function"] = True
        out["static_category"] = CAT_MISSING_FUNCTION
        return out

    # ------------------------------------------------------------------
    # 7. Wrong function name
    # ------------------------------------------------------------------
    if expected_fn_name and fn_defs:
        found_names = {n for n, _ in fn_defs}
        if expected_fn_name not in found_names:
            out["wrong_function_name"] = True
            out["static_category"] = CAT_WRONG_FUNCTION_NAME
            return out

    # ------------------------------------------------------------------
    # 8. Signature error (positional arg count mismatch)
    # ------------------------------------------------------------------
    if (
        expected_fn_name
        and expected_arg_count is not None
        and expected_arg_count >= 0
        and fn_defs
    ):
        for name, nargs in fn_defs:
            if name == expected_fn_name and nargs >= 0 and nargs != expected_arg_count:
                out["signature_error"] = True
                out["static_category"] = CAT_SIGNATURE_ERROR
                return out

    # ------------------------------------------------------------------
    # 9. Wildcard import detection
    # ------------------------------------------------------------------
    if _RE_IMPORT_STAR.search(code):
        out["import_error"] = True
        # Treat wildcard imports as import_error category only when the
        # generation is otherwise structurally correct but not passing.
        if correct is not True:
            out["static_category"] = CAT_IMPORT_ERROR
            return out

    # ------------------------------------------------------------------
    # 10. Prompt noncompliance (prose response when code was expected)
    # ------------------------------------------------------------------
    if not fn_defs and not fn_expected and _RE_PROSE_LEAD.match(stripped):
        out["prompt_noncompliance"] = True
        out["static_category"] = CAT_PROMPT_NONCOMPLIANCE
        return out

    # ------------------------------------------------------------------
    # 11. Logic error (structurally OK but harness marks incorrect)
    # ------------------------------------------------------------------
    if correct is False:
        out["static_category"] = CAT_LOGIC_ERROR
        return out

    # ------------------------------------------------------------------
    # 12. Pass (structurally OK and correct or unknown correctness)
    # ------------------------------------------------------------------
    out["static_category"] = CAT_PASS
    return out


# ---------------------------------------------------------------------------
# Harness execution result integration
# ---------------------------------------------------------------------------

def _classify_harness_failure(sample_row: Dict[str, Any]) -> Optional[str]:
    """Extract a failure category from harness execution metadata when present.

    Returns ``None`` when no execution metadata is available so the caller can
    fall back to the static category.
    """
    # Explicit pass signal from lm-eval metrics
    for key in ("pass@1", "acc", "exact_match"):
        v = sample_row.get(key)
        if isinstance(v, (int, float)):
            if float(v) >= 1.0:
                return CAT_PASS
            # explicit fail — don't return yet; try to be more specific
            break

    # Traceback / stderr from execution harness
    metadata = sample_row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    tb = metadata.get("traceback") or metadata.get("stderr") or ""
    if isinstance(tb, str) and tb.strip():
        tbl = tb.lower()
        if "timeouterror" in tbl or "timed out" in tbl:
            return CAT_TIMEOUT
        if "syntaxerror" in tbl:
            return CAT_SYNTAX_ERROR
        if "indentationerror" in tbl:
            return CAT_INDENTATION_ERROR
        if "importerror" in tbl or "modulenotfounderror" in tbl:
            return CAT_IMPORT_ERROR
        if "assertionerror" in tbl or "test failed" in tbl:
            return CAT_TEST_FAILURE
        return CAT_RUNTIME_ERROR

    # Generic status field used by some custom harnesses
    status = metadata.get("status") or metadata.get("result")
    if isinstance(status, str):
        sl = status.lower()
        if "pass" in sl:
            return CAT_PASS
        if "timeout" in sl:
            return CAT_TIMEOUT
        if "syntax" in sl:
            return CAT_SYNTAX_ERROR
        if "import" in sl:
            return CAT_IMPORT_ERROR
        if "fail" in sl or "error" in sl:
            return CAT_TEST_FAILURE

    return None


# ---------------------------------------------------------------------------
# Build CodeFailureRecord from a DiagnosticSampleRecord dict
# ---------------------------------------------------------------------------

def build_failure_record(
    sample_row: Dict[str, Any],
    *,
    run_id: str,
) -> CodeFailureRecord:
    """Convert a DiagnosticSampleRecord dict to a :class:`CodeFailureRecord`.

    Runs static analysis on the ``generation`` field and, if harness
    execution metadata is present in the row, overlays a ``harness_category``.
    The resolved ``failure_category`` equals ``harness_category`` when
    available, otherwise ``static_category``.
    """
    task       = sample_row.get("task")       or "unknown"
    sample_id  = sample_row.get("sample_id")  or "unknown"
    generation = sample_row.get("generation") or ""
    prompt     = sample_row.get("prompt")
    target     = sample_row.get("target")
    correct    = sample_row.get("correct")
    metric_val = sample_row.get("metric_value")

    expected_fn  = _extract_expected_fn_name(prompt, target)
    expected_argc = _count_prompt_args(prompt, expected_fn)

    static_result = classify_static_failure(
        generation,
        expected_fn_name=expected_fn,
        expected_arg_count=expected_argc,
        prompt=prompt,
        correct=correct,
    )

    harness_cat = _classify_harness_failure(sample_row)

    # Prefer execution-aware harness result; fall back to static.
    failure_category = harness_cat if harness_cat is not None else static_result["static_category"]

    return CodeFailureRecord(
        run_id=run_id,
        task=task,
        sample_id=sample_id,
        failure_category=failure_category,
        static_category=static_result["static_category"],
        harness_category=harness_cat,
        parse_ok=static_result["parse_ok"],
        syntax_error=static_result["syntax_error_flag"],
        indentation_error=static_result["indentation_error_flag"],
        unmatched_delimiter=static_result["unmatched_delimiter"],
        missing_function=static_result["missing_function"],
        wrong_function_name=static_result["wrong_function_name"],
        signature_error=static_result["signature_error"],
        import_error=static_result["import_error"],
        runtime_error=(failure_category == CAT_RUNTIME_ERROR),
        test_failure=(failure_category == CAT_TEST_FAILURE),
        timeout=(failure_category == CAT_TIMEOUT),
        empty_or_truncated=static_result["empty_or_truncated"],
        prompt_noncompliance=static_result["prompt_noncompliance"],
        traceback_type=None,
        function_name_expected=expected_fn,
        function_name_found=static_result["function_name_found"],
        correct=correct,
        metric_value=float(metric_val) if metric_val is not None else None,
        metadata={"detail": static_result["detail"]},
    )


# ---------------------------------------------------------------------------
# Intervention index
# ---------------------------------------------------------------------------

def _index_interventions(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Index intervention rows by (run_id, task, sample_id)."""
    idx: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("run_id")    or ""),
            str(row.get("task")      or ""),
            str(row.get("sample_id") or ""),
        )
        idx[key].append(row)
    return idx


def _safe_mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Join a failure record with its interventions
# ---------------------------------------------------------------------------

def _join_record(
    rec: CodeFailureRecord,
    interventions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Produce a merged dict from a CodeFailureRecord and its interventions."""
    base = record_to_dict(rec)
    if not interventions:
        return base

    by_type: Dict[str, List[float]] = defaultdict(list)
    # per-layer: layer_idx -> {itype: [deltas]}
    by_layer: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    role_deltas: Dict[str, List[float]] = defaultdict(list)
    tid_deltas: Dict[int, List[float]] = defaultdict(list)

    for row in interventions:
        itype = str(row.get("intervention_type") or "")
        delta = row.get("delta_loss")
        if delta is None:
            continue
        delta = float(delta)
        by_type[itype].append(delta)

        layer_idx = row.get("layer_idx")
        if isinstance(layer_idx, int):
            by_layer[layer_idx][itype].append(delta)

        roles     = row.get("token_roles")
        per_tok   = row.get("delta_per_token_nll")
        tok_ids   = row.get("token_ids")

        if isinstance(roles, list) and isinstance(per_tok, list) and len(roles) == len(per_tok):
            for role, td in zip(roles, per_tok):
                if isinstance(role, str) and isinstance(td, (int, float)):
                    role_deltas[role].append(float(td))

        if isinstance(tok_ids, list) and isinstance(per_tok, list) and len(tok_ids) == len(per_tok):
            for tid, td in zip(tok_ids, per_tok):
                if isinstance(tid, int) and isinstance(td, (int, float)):
                    tid_deltas[tid].append(float(td))

    base["intervention_summary"] = {
        itype: {
            "mean_delta_loss": _safe_mean(vals),
            "count": len(vals),
        }
        for itype, vals in by_type.items()
    }

    # Harmful layers: layers where ablate_stem delta < -0.01 (STEM hurt the model)
    harmful_layers = sorted({
        layer
        for layer, type_vals in by_layer.items()
        for itype, vals in type_vals.items()
        if itype in {"ablate_stem", "ablate_layer_stem"}
        and _safe_mean(vals) is not None
        and (_safe_mean(vals) or 0.0) < -0.01
    })
    base["harmful_layers"] = harmful_layers

    # Harmful roles: mean per-token delta < -0.01 when STEM is ablated
    base["harmful_token_roles"] = sorted(
        role for role, deltas in role_deltas.items()
        if (_safe_mean(deltas) or 0.0) < -0.01
    )

    # Most-harmful token ids
    harmful_tids = sorted(
        (
            (tid, _safe_mean(deltas) or 0.0)
            for tid, deltas in tid_deltas.items()
            if (_safe_mean(deltas) or 0.0) < -0.01
        ),
        key=lambda x: x[1],
    )
    base["harmful_token_ids"] = [tid for tid, _ in harmful_tids[:20]]

    return base


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def aggregate_summaries(joined_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce summary statistics over joined failure + intervention records.

    Returns a JSON-safe dict suitable for writing to
    ``code_causal_failure_analysis.json``.
    """
    counts_by_cat: Counter[str] = Counter()
    counts_by_task: Dict[str, Counter[str]] = defaultdict(Counter)

    # delta_loss lists: category -> intervention_type -> [floats]
    delta_acc: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    harm_layers: Dict[str, List[int]] = defaultdict(list)
    harm_roles:  Dict[str, List[str]] = defaultdict(list)
    harm_tids:   Dict[str, List[int]] = defaultdict(list)

    stem_helpful:    List[Dict[str, Any]] = []
    gate_up_helpful: List[Dict[str, Any]] = []

    for rec in joined_records:
        cat  = rec.get("failure_category") or CAT_UNKNOWN
        task = rec.get("task")             or "unknown"
        counts_by_cat[cat]        += 1
        counts_by_task[task][cat] += 1

        intv = rec.get("intervention_summary") or {}
        for itype, stats in intv.items():
            md = stats.get("mean_delta_loss")
            if md is not None:
                delta_acc[cat][itype].append(float(md))

        harm_layers[cat].extend(rec.get("harmful_layers")      or [])
        harm_roles[cat].extend(rec.get("harmful_token_roles")  or [])
        harm_tids[cat].extend(rec.get("harmful_token_ids")     or [])

        # Samples where STEM was harmful (ablating it lowered loss)
        ablate_stem = intv.get("ablate_stem") or intv.get("ablate_layer_stem")
        if ablate_stem and len(stem_helpful) < 10:
            md_s = ablate_stem.get("mean_delta_loss")
            if isinstance(md_s, float) and md_s < -0.01:
                stem_helpful.append({
                    "sample_id":       rec.get("sample_id"),
                    "task":            task,
                    "failure_category": cat,
                    "ablate_stem_mean_delta": md_s,
                    "generation_snippet": (rec.get("generation") or "")[:300],
                })

        # Samples where forcing gate toward up path (force_gate_1) was better
        d0 = (intv.get("force_gate_0") or {}).get("mean_delta_loss")
        d1 = (intv.get("force_gate_1") or {}).get("mean_delta_loss")
        if (
            isinstance(d0, float)
            and isinstance(d1, float)
            and d1 < d0 - 0.01
            and len(gate_up_helpful) < 10
        ):
            gate_up_helpful.append({
                "sample_id":         rec.get("sample_id"),
                "task":              task,
                "failure_category":  cat,
                "force_gate_0_delta": d0,
                "force_gate_1_delta": d1,
            })

    def _top_counter(items: List[Any], n: int = 10) -> List[Any]:
        return [item for item, _ in Counter(items).most_common(n)]

    avg_delta_by_cat: Dict[str, Dict[str, Optional[float]]] = {
        cat: {itype: _safe_mean(vals) for itype, vals in by_itype.items()}
        for cat, by_itype in delta_acc.items()
    }

    return sanitize_for_json({
        "failure_counts_by_category": dict(counts_by_cat),
        "failure_counts_by_task": {
            task: dict(c) for task, c in counts_by_task.items()
        },
        "avg_delta_loss_by_category": avg_delta_by_cat,
        "harmful_layers_by_category":      {c: _top_counter(v) for c, v in harm_layers.items()},
        "harmful_token_roles_by_category": {c: _top_counter(v) for c, v in harm_roles.items()},
        "harmful_token_ids_by_category":   {c: _top_counter(v) for c, v in harm_tids.items()},
        "stem_helpful_examples":    stem_helpful,
        "gate_up_helpful_examples": gate_up_helpful,
        "total_samples":            len(joined_records),
    })


# ---------------------------------------------------------------------------
# Task filter
# ---------------------------------------------------------------------------

_CODE_TASK_KEYWORDS: frozenset[str] = frozenset(
    ["mbpp", "humaneval", "humaneval_plus", "apps", "code", "python"]
)


def _is_code_task(task_name: str, code_tasks: Optional[List[str]]) -> bool:
    lower = task_name.lower()
    if code_tasks:
        return any(ct.lower() in lower or lower.startswith(ct.lower()) for ct in code_tasks)
    return any(kw in lower for kw in _CODE_TASK_KEYWORDS)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_code_causal_analysis(
    output_dir: "os.PathLike[str] | str",
    *,
    run_id: str = "unknown",
    code_tasks: Optional[List[str]] = None,
    sample_jsonl: Optional["os.PathLike[str] | str"] = None,
    intervention_jsonl: Optional["os.PathLike[str] | str"] = None,
    save_examples: bool = True,
    max_examples: int = 200,
    max_example_text_chars: int = 1000,
) -> Dict[str, Any]:
    """Run MBPP/HumanEval causal failure analysis from artifact files.

    Reads ``diagnostics_eval_samples.jsonl`` and
    ``interventions_task_aligned.jsonl``, classifies each code sample into a
    failure category, joins with intervention deltas, aggregates summaries,
    and writes output artifacts.

    This function is safe to call even when either input file is absent; it
    returns ``{"skipped": True, "reason": ...}`` in that case.

    Parameters
    ----------
    output_dir:
        Directory to read artifacts from and write results to.
    run_id:
        Identifier propagated into records.
    code_tasks:
        Task-name keywords to include; defaults to the built-in MBPP/HumanEval
        keyword set.
    sample_jsonl:
        Override path for the eval-sample JSONL.
    intervention_jsonl:
        Override path for the intervention JSONL.
    save_examples:
        Write ``code_failure_examples.jsonl`` with bounded prompt/generation.
    max_examples:
        Cap on the number of examples to write.
    max_example_text_chars:
        Max characters for prompt/generation in the examples file.

    Returns
    -------
    Summary dict (also written to ``code_causal_failure_analysis.json``).
    """
    import os as _os  # local import keeps module importable without os shim

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = (
        Path(sample_jsonl)
        if sample_jsonl
        else output_dir / "diagnostics_eval_samples.jsonl"
    )
    intv_path = (
        Path(intervention_jsonl)
        if intervention_jsonl
        else output_dir / "interventions_task_aligned.jsonl"
    )

    sample_rows    = read_jsonl(samples_path)
    intv_rows      = read_jsonl(intv_path)

    if not sample_rows:
        logger.info(
            "code_diagnostics: no sample records at %s; skipping", samples_path
        )
        return {"skipped": True, "reason": "no_sample_records"}

    code_rows = [
        r for r in sample_rows
        if _is_code_task(str(r.get("task") or ""), code_tasks)
    ]

    if not code_rows:
        all_tasks = sorted({r.get("task") for r in sample_rows})
        logger.info(
            "code_diagnostics: no code-task samples (tasks present: %s); skipping",
            all_tasks,
        )
        return {"skipped": True, "reason": "no_code_task_samples", "tasks_present": all_tasks}

    logger.info(
        "code_diagnostics: analysing %d code samples from %d total",
        len(code_rows),
        len(sample_rows),
    )

    intv_index = _index_interventions(intv_rows)
    harness_data_available = False

    failure_records: List[CodeFailureRecord] = []
    joined_records:  List[Dict[str, Any]] = []
    example_rows:    List[Dict[str, Any]] = []

    for row in code_rows:
        row_run_id = str(row.get("run_id") or run_id)
        rec = build_failure_record(row, run_id=row_run_id)

        if rec.harness_category is not None:
            harness_data_available = True

        failure_records.append(rec)

        intv_key = (row_run_id, rec.task, rec.sample_id)
        sample_intv = intv_index.get(intv_key, [])
        joined = _join_record(rec, sample_intv)
        # Attach text for examples (trimmed separately below)
        joined["generation"] = row.get("generation")
        joined_records.append(joined)

        if save_examples and len(example_rows) < max_examples:
            mc = max_example_text_chars
            example_rows.append(sanitize_for_json({
                "run_id":               row_run_id,
                "task":                 rec.task,
                "sample_id":            rec.sample_id,
                "failure_category":     rec.failure_category,
                "static_category":      rec.static_category,
                "harness_category":     rec.harness_category,
                "correct":              rec.correct,
                "metric_value":         rec.metric_value,
                "function_name_expected": rec.function_name_expected,
                "function_name_found":    rec.function_name_found,
                "prompt_snippet":       (row.get("prompt") or "")[:mc],
                "generation_snippet":   (row.get("generation") or "")[:mc],
                "intervention_summary": joined.get("intervention_summary"),
            }))

    # Write failure records JSONL
    failures_path = output_dir / CODE_FAILURES_JSONL
    append_jsonl(failures_path, failure_records, rank0_only=False)
    logger.info(
        "code_diagnostics: wrote %d failure records → %s",
        len(failure_records),
        failures_path,
    )

    # Write examples JSONL
    if save_examples and example_rows:
        examples_path = output_dir / CODE_FAILURE_EXAMPLES_JSONL
        append_jsonl(examples_path, example_rows, rank0_only=False)
        logger.info(
            "code_diagnostics: wrote %d examples → %s",
            len(example_rows),
            examples_path,
        )

    # Aggregate and write summary
    summary = aggregate_summaries(joined_records)
    summary["run_id"]                  = run_id
    summary["sample_jsonl"]            = str(samples_path)
    summary["intervention_jsonl"]      = str(intv_path)
    summary["harness_data_available"]  = harness_data_available
    summary["intervention_data_available"] = bool(intv_rows)

    analysis_path = output_dir / CODE_CAUSAL_ANALYSIS_JSON
    write_json_atomic(analysis_path, summary, rank0_only=False)
    logger.info("code_diagnostics: wrote summary → %s", analysis_path)

    return summary
