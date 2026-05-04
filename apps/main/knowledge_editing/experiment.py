"""Core utilities for STEM knowledge-editing experiments.

The intervention implemented here is intentionally narrow: the textual prompt
and token IDs remain those of the source entity, while the STEM embedding
lookup vectors at the source entity token positions are replaced by vectors
looked up for the target entity. Model weights are never modified.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger(__name__)


EDIT_MODES = ("auto", "one_to_one", "average", "copy", "left_pad", "right_pad")
DEFAULT_MATH_TEXT_OPERATORS = ("add", "subtract", "multiply", "divide")


@dataclass(frozen=True)
class CountryCapitalExample:
    country: str
    capital_sentence: str


@dataclass(frozen=True)
class MathTextExample:
    problem: str
    answer: str


DEFAULT_COUNTRY_CAPITAL_EXAMPLES: Tuple[CountryCapitalExample, ...] = (
    CountryCapitalExample(
        country="United States of America",
        capital_sentence=(
            "Washington D.C. is the capital city of the United States. It is "
            "located on the Potomac River and serves as the seat of the federal "
            "government."
        ),
    ),
    CountryCapitalExample(
        country="United Kingdom of Great Britain",
        capital_sentence=(
            "London is the capital city of the United Kingdom. It is a major "
            "global city known for government, finance, culture, and history."
        ),
    ),
    CountryCapitalExample(
        country="France",
        capital_sentence=(
            "Paris is the capital city of France. It is located on the Seine "
            "River and is known for politics, culture, education, and landmarks."
        ),
    ),
)


DEFAULT_MATH_TEXT_EXAMPLES: Tuple[MathTextExample, ...] = (
    MathTextExample(problem="three subtract by one", answer="two"),
    MathTextExample(problem="one add by ten", answer="eleven"),
    MathTextExample(problem="two multiply by four", answer="eight"),
    MathTextExample(problem="six divide by three", answer="two"),
)


@dataclass(frozen=True)
class EntityTokenSpan:
    entity_text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_positions: List[int]
    token_ids: List[int]
    decoded_pieces: List[str]


@dataclass(frozen=True)
class TokenizedPrompt:
    text: str
    token_ids: List[int]
    entity_span: EntityTokenSpan
    add_bos: bool = False
    add_eos: bool = False


@dataclass(frozen=True)
class StemEditPlan:
    requested_mode: str
    resolved_mode: str
    source_positions: List[int]
    source_token_ids: List[int]
    target_token_ids: List[int]
    replacement_token_ids: Optional[List[int]] = None
    pad_token_id: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TopKToken:
    token_id: int
    token_str: str
    logit: float
    probability: float


@dataclass(frozen=True)
class TopKResult:
    case: str
    prompt: str
    prompt_token_ids: List[int]
    top_k: int
    tokens: List[TopKToken]


@dataclass(frozen=True)
class GenerationResult:
    case: str
    prompt: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    continuation_text: str
    full_text: str
    stop_reason: str


@dataclass(frozen=True)
class InterventionLayerDiagnostic:
    layer_idx: int
    stem_embedding_index: int
    output_shape: List[int]
    output_dtype: str
    output_device: str
    source_positions: List[int]
    source_token_ids_at_positions: List[int]
    target_token_ids: List[int]
    replacement_token_ids: Optional[List[int]]
    max_abs_diff_unedited_vs_baseline: float
    max_abs_diff_edited_vs_expected: float
    max_abs_diff_edit_vs_baseline: float
    unedited_positions_match_baseline: bool
    edited_positions_match_expected: bool
    output_is_distinct_tensor: bool
    selected_weight_rows_unchanged: bool
    weight_version_unchanged: bool
    passed: bool
    failures: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class InterventionDiagnostics:
    passed: bool
    prompt_text_unchanged: bool
    prompt_token_ids_unchanged: bool
    source_positions_match_plan: bool
    source_token_ids_match_plan: bool
    stem_layers_expected: List[int]
    stem_layers_checked: List[int]
    forward_path_checked: bool
    forward_path_called_layers: List[int]
    forward_path_calls_match_stem_layers: bool
    edit_mode: str
    layer_diagnostics: List[InterventionLayerDiagnostic]
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def build_country_capital_prompt(
    query_country: str,
    examples: Optional[Sequence[CountryCapitalExample]] = None,
) -> str:
    """Build the few-shot country-capital retrieval prompt."""

    shots = examples if examples is not None else DEFAULT_COUNTRY_CAPITAL_EXAMPLES
    blocks = [
        f"Country: {example.country}\nCapital: {example.capital_sentence}"
        for example in shots
    ]
    blocks.append(f"Country: {query_country}\nCapital:")
    return "\n\n".join(blocks)


def build_math_text_prompt(
    query_operator: str,
    *,
    query_left_operand: str = "nine",
    query_right_operand: str = "two",
    examples: Optional[Sequence[MathTextExample]] = None,
) -> str:
    """Build the few-shot text-arithmetic prompt.

    The query operator is intentionally isolated in the final problem so the
    same entity-span and STEM-vector replacement machinery can edit only that
    operator while leaving all prompt token IDs unchanged in the intervened
    case.
    """

    shots = examples if examples is not None else DEFAULT_MATH_TEXT_EXAMPLES
    blocks = [
        f"Problem: {example.problem}\nAnswer: {example.answer}"
        for example in shots
    ]
    blocks.append(
        f"Problem: {query_left_operand} {query_operator} by {query_right_operand}\nAnswer:"
    )
    return "\n\n".join(blocks)


def replace_last_entity(text: str, source_entity: str, target_entity: str) -> str:
    """Replace only the last literal source entity occurrence in text."""

    start = text.rfind(source_entity)
    if start < 0:
        raise ValueError(f"Source entity {source_entity!r} was not found in prompt")
    end = start + len(source_entity)
    return f"{text[:start]}{target_entity}{text[end:]}"


def _encode(tokenizer: Any, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
    return list(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos))


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return str(tokenizer.decode(list(token_ids)))


def _decode_piece(tokenizer: Any, token_id: int) -> str:
    try:
        return _decode(tokenizer, [int(token_id)])
    except Exception:
        return f"<token:{int(token_id)}>"


def _find_overlapping_token_positions(
    text: str,
    token_offsets: Sequence[int],
    char_start: int,
    char_end: int,
) -> List[int]:
    positions: List[int] = []
    for idx, start in enumerate(token_offsets):
        end = token_offsets[idx + 1] if idx + 1 < len(token_offsets) else len(text)
        start = max(0, int(start))
        end = max(start, int(end))
        if max(start, char_start) < min(end, char_end):
            positions.append(idx)
    return positions


def _longest_common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    upto = min(len(left), len(right))
    for idx in range(upto):
        if left[idx] != right[idx]:
            return idx
    return upto


def _prefix_difference_span(
    tokenizer: Any,
    text: str,
    char_start: int,
    char_end: int,
    content_token_ids: Sequence[int],
) -> List[int]:
    """Fallback span detector using prefix tokenization differences."""

    prefix_ids = _encode(tokenizer, text[:char_start], add_bos=False, add_eos=False)
    prefix_entity_ids = _encode(tokenizer, text[:char_end], add_bos=False, add_eos=False)
    token_start = _longest_common_prefix_len(content_token_ids, prefix_ids)
    token_end = _longest_common_prefix_len(content_token_ids, prefix_entity_ids)
    if token_end > token_start:
        return list(range(token_start, token_end))
    return []


def find_last_entity_token_span(
    text: str,
    entity_text: str,
    tokenizer: Any,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> EntityTokenSpan:
    """Find the token span for the last exact entity occurrence in a full prompt.

    The primary path uses tokenizer offsets from the full prompt, which handles
    leading-space BPE pieces such as " Spain". A prefix-difference fallback is
    used for tokenizers without usable offsets.
    """

    if not entity_text:
        raise ValueError("entity_text must be non-empty")
    char_start = text.rfind(entity_text)
    if char_start < 0:
        raise ValueError(f"Entity {entity_text!r} was not found in prompt")
    char_end = char_start + len(entity_text)

    content_token_ids = _encode(tokenizer, text, add_bos=False, add_eos=False)
    full_token_ids = _encode(tokenizer, text, add_bos=add_bos, add_eos=add_eos)
    bos_shift = len(full_token_ids) - len(content_token_ids) - int(add_eos)
    if bos_shift < 0:
        bos_shift = 1 if add_bos else 0

    content_positions: List[int] = []
    try:
        _pieces, offsets = tokenizer.get_token_offsets(text, content_token_ids)
        if len(offsets) == len(content_token_ids):
            content_positions = _find_overlapping_token_positions(
                text, offsets, char_start, char_end
            )
    except Exception as exc:
        logger.debug("Tokenizer offset lookup failed; using fallback: %s", exc)

    if not content_positions:
        content_positions = _prefix_difference_span(
            tokenizer, text, char_start, char_end, content_token_ids
        )

    if not content_positions:
        entity_ids = _encode(tokenizer, entity_text, add_bos=False, add_eos=False)
        raise ValueError(
            "Could not locate entity token span for "
            f"{entity_text!r}. Standalone token IDs={entity_ids}; "
            "try checking the prompt text and tokenizer."
        )

    token_positions = [pos + bos_shift for pos in content_positions]
    token_start = token_positions[0]
    token_end = token_positions[-1] + 1
    token_ids = [int(full_token_ids[pos]) for pos in token_positions]
    decoded_pieces = [_decode_piece(tokenizer, token_id) for token_id in token_ids]

    return EntityTokenSpan(
        entity_text=entity_text,
        char_start=char_start,
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
        token_positions=token_positions,
        token_ids=token_ids,
        decoded_pieces=decoded_pieces,
    )


def tokenize_with_entity_span(
    tokenizer: Any,
    text: str,
    entity_text: str,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> TokenizedPrompt:
    token_ids = _encode(tokenizer, text, add_bos=add_bos, add_eos=add_eos)
    span = find_last_entity_token_span(
        text,
        entity_text,
        tokenizer,
        add_bos=add_bos,
        add_eos=add_eos,
    )
    return TokenizedPrompt(
        text=text,
        token_ids=token_ids,
        entity_span=span,
        add_bos=add_bos,
        add_eos=add_eos,
    )


def _resolve_actual_edit_mode(mode: str, source_len: int, target_len: int) -> str:
    if mode not in EDIT_MODES:
        raise ValueError(f"Unknown edit mode {mode!r}; expected one of {EDIT_MODES}")
    if source_len <= 0:
        raise ValueError("source_len must be positive")
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if mode == "auto":
        return "one_to_one" if source_len == target_len else "average"
    if mode == "one_to_one" and source_len != target_len:
        raise ValueError(
            "edit-mode one_to_one requires equal tokenized lengths; "
            f"source_len={source_len}, target_len={target_len}"
        )
    return mode


def _choose_pad_token_id(
    explicit_pad_token_id: Optional[int],
    eos_token_id: Optional[int],
    bos_token_id: Optional[int],
    warnings: List[str],
) -> int:
    if explicit_pad_token_id is not None and int(explicit_pad_token_id) >= 0:
        return int(explicit_pad_token_id)
    if eos_token_id is not None and int(eos_token_id) >= 0:
        warnings.append("No pad token is available; using eos token for padding.")
        return int(eos_token_id)
    if bos_token_id is not None and int(bos_token_id) >= 0:
        warnings.append("No pad/eos token is available; using bos token for padding.")
        return int(bos_token_id)
    raise ValueError("left_pad/right_pad requires a pad token, eos fallback, or bos fallback")


def build_replacement_token_ids(
    source_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    *,
    edit_mode: str = "auto",
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
) -> StemEditPlan:
    """Resolve edit mode and replacement token IDs when a token sequence is used."""

    source_ids = [int(x) for x in source_token_ids]
    target_ids = [int(x) for x in target_token_ids]
    warnings: List[str] = []
    resolved = _resolve_actual_edit_mode(edit_mode, len(source_ids), len(target_ids))
    replacement_ids: Optional[List[int]]

    if resolved == "average":
        replacement_ids = None
    elif resolved == "one_to_one":
        replacement_ids = list(target_ids)
    elif resolved == "copy":
        replacement_ids = [target_ids[idx % len(target_ids)] for idx in range(len(source_ids))]
    elif resolved in ("left_pad", "right_pad"):
        pad_id = _choose_pad_token_id(pad_token_id, eos_token_id, bos_token_id, warnings)
        if len(target_ids) >= len(source_ids):
            if len(target_ids) > len(source_ids):
                warnings.append(
                    f"Target token sequence is longer than source for {resolved}; "
                    "truncating to source length."
                )
            if resolved == "left_pad":
                replacement_ids = list(target_ids[-len(source_ids) :])
            else:
                replacement_ids = list(target_ids[: len(source_ids)])
        else:
            pad_count = len(source_ids) - len(target_ids)
            if resolved == "left_pad":
                replacement_ids = [pad_id] * pad_count + list(target_ids)
            else:
                replacement_ids = list(target_ids) + [pad_id] * pad_count
        pad_token_id = pad_id
    else:
        raise AssertionError(f"Unhandled resolved edit mode {resolved!r}")

    return StemEditPlan(
        requested_mode=edit_mode,
        resolved_mode=resolved,
        source_positions=[],
        source_token_ids=source_ids,
        target_token_ids=target_ids,
        replacement_token_ids=replacement_ids,
        pad_token_id=pad_token_id,
        warnings=warnings,
    )


class StemEmbeddingOverride:
    """Callable STEM embedding override with attached edit metadata."""

    def __init__(self, model: Any, plan: StemEditPlan):
        self.model = model
        self.plan = plan

    def _embedding_for_layer(self, layer_idx: int) -> Any:
        if not hasattr(self.model, "_layer_to_stem_idx"):
            raise RuntimeError("model._layer_to_stem_idx is required for STEM editing")
        if not hasattr(self.model, "stem_embeddings"):
            raise RuntimeError("model.stem_embeddings is required for STEM editing")
        layer_to_stem_idx = getattr(self.model, "_layer_to_stem_idx")
        if layer_idx not in layer_to_stem_idx:
            raise RuntimeError(
                f"Layer {layer_idx} is not a STEM layer; available={sorted(layer_to_stem_idx)}"
            )
        return self.model.stem_embeddings[layer_to_stem_idx[layer_idx]]

    def _lookup_target_vectors(
        self,
        embedding: Any,
        token_values: torch.Tensor,
        output_dtype: torch.dtype,
        output_device: torch.device,
    ) -> torch.Tensor:
        batch_size = int(token_values.shape[0])
        if self.plan.resolved_mode == "average":
            target_ids = torch.tensor(
                self.plan.target_token_ids,
                dtype=torch.long,
                device=token_values.device,
            ).view(1, -1)
            target_ids = target_ids.expand(batch_size, -1)
            target_vectors = embedding(target_ids).to(device=output_device, dtype=output_dtype)
            avg = target_vectors.mean(dim=1, keepdim=True)
            return avg.expand(batch_size, len(self.plan.source_positions), -1)

        if not self.plan.replacement_token_ids:
            raise RuntimeError(
                f"Resolved edit mode {self.plan.resolved_mode!r} requires replacement_token_ids"
            )
        replacement_ids = torch.tensor(
            self.plan.replacement_token_ids,
            dtype=torch.long,
            device=token_values.device,
        ).view(1, -1)
        replacement_ids = replacement_ids.expand(batch_size, -1)
        return embedding(replacement_ids).to(device=output_device, dtype=output_dtype)

    @torch.no_grad()
    def __call__(self, layer_idx: int, token_values: torch.Tensor) -> torch.Tensor:
        embedding = self._embedding_for_layer(layer_idx)
        y = embedding(token_values)
        out = y.clone()
        positions = list(self.plan.source_positions)
        if not positions:
            raise RuntimeError("No source positions configured for STEM editing")
        if max(positions) >= out.shape[1]:
            raise RuntimeError(
                "Configured source position exceeds current sequence length: "
                f"max_position={max(positions)}, seqlen={out.shape[1]}"
            )
        replacement = self._lookup_target_vectors(
            embedding,
            token_values,
            output_dtype=out.dtype,
            output_device=out.device,
        )
        if replacement.shape[1] != len(positions):
            raise RuntimeError(
                f"Replacement length {replacement.shape[1]} does not match "
                f"source length {len(positions)}"
            )
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=out.device)
        out[:, pos_tensor, :] = replacement
        return out


class TracedStemEmbeddingOverride:
    """Wrap a STEM override and record which layers the LM forward requested."""

    def __init__(self, base: StemEmbeddingOverride):
        self.base = base
        self.calls: List[int] = []

    @property
    def plan(self) -> StemEditPlan:
        return self.base.plan

    def __call__(self, layer_idx: int, token_values: torch.Tensor) -> torch.Tensor:
        self.calls.append(int(layer_idx))
        return self.base(layer_idx, token_values)


def make_stem_embedding_override_fn(
    model: Any,
    source_token_positions: Sequence[int],
    target_token_ids: Sequence[int],
    *,
    source_token_ids: Optional[Sequence[int]] = None,
    edit_mode: str = "auto",
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
) -> StemEmbeddingOverride:
    """Create the non-mutating STEM embedding intervention callback."""

    positions = [int(pos) for pos in source_token_positions]
    if not positions:
        raise ValueError("source_token_positions must be non-empty")
    if positions != sorted(positions):
        raise ValueError(f"source_token_positions must be sorted; got {positions}")
    source_ids = [int(x) for x in (source_token_ids if source_token_ids is not None else positions)]
    plan = build_replacement_token_ids(
        source_ids,
        target_token_ids,
        edit_mode=edit_mode,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    )
    plan = StemEditPlan(
        requested_mode=plan.requested_mode,
        resolved_mode=plan.resolved_mode,
        source_positions=positions,
        source_token_ids=source_ids,
        target_token_ids=plan.target_token_ids,
        replacement_token_ids=plan.replacement_token_ids,
        pad_token_id=plan.pad_token_id,
        warnings=plan.warnings,
    )
    return StemEmbeddingOverride(model=model, plan=plan)


def _max_abs_or_zero(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().abs().max().item())


def _selected_weight_row_snapshot(
    embedding: Any,
    token_ids: Sequence[int],
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    weight = getattr(embedding, "weight", None)
    if weight is None:
        return None, None
    unique_ids = sorted({int(token_id) for token_id in token_ids})
    if not unique_ids:
        return None, int(getattr(weight, "_version", -1))
    rows = torch.tensor(unique_ids, dtype=torch.long, device=weight.device)
    return weight.detach().index_select(0, rows).clone(), int(getattr(weight, "_version", -1))


def _selected_weight_rows_unchanged(
    embedding: Any,
    token_ids: Sequence[int],
    before_rows: Optional[torch.Tensor],
) -> bool:
    weight = getattr(embedding, "weight", None)
    if weight is None or before_rows is None:
        return True
    unique_ids = sorted({int(token_id) for token_id in token_ids})
    if not unique_ids:
        return True
    rows = torch.tensor(unique_ids, dtype=torch.long, device=weight.device)
    after_rows = weight.detach().index_select(0, rows)
    return bool(torch.equal(before_rows, after_rows))


@torch.no_grad()
def run_intervention_diagnostics(
    *,
    model: Any,
    original_prompt: str,
    intervened_prompt: str,
    original_token_ids: Sequence[int],
    intervened_token_ids: Sequence[int],
    override: StemEmbeddingOverride,
    device: Union[str, torch.device],
    expected_source_positions: Sequence[int],
    expected_stem_layers: Optional[Sequence[int]] = None,
    attn_impl: str = "sdpa",
    run_forward_path_check: bool = True,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> InterventionDiagnostics:
    """Validate that the STEM intervention edits only the intended vectors.

    This is a production safety gate for the central causal claim of the
    experiment: the intervened case must keep the original prompt text and
    token IDs, while replacing only the STEM lookup vectors at the configured
    source positions for every STEM layer.
    """

    failures: List[str] = []
    warnings: List[str] = []
    original_ids = [int(token_id) for token_id in original_token_ids]
    intervened_ids = [int(token_id) for token_id in intervened_token_ids]
    plan_positions = list(override.plan.source_positions)
    expected_positions = [int(pos) for pos in expected_source_positions]
    prompt_text_unchanged = original_prompt == intervened_prompt
    prompt_token_ids_unchanged = original_ids == intervened_ids
    source_positions_match_plan = plan_positions == expected_positions
    source_token_ids_match_plan = True

    if not prompt_text_unchanged:
        failures.append("Intervened prompt text differs from original prompt text.")
    if not prompt_token_ids_unchanged:
        failures.append("Intervened token IDs differ from original token IDs.")
    if not source_positions_match_plan:
        failures.append(
            "Override source positions do not match detected source entity span: "
            f"plan={plan_positions}, detected={expected_positions}."
        )
    if not original_ids:
        failures.append("Original token IDs are empty.")
    if any(pos < 0 or pos >= len(original_ids) for pos in plan_positions):
        failures.append(
            f"Source positions {plan_positions} are not all within prompt length {len(original_ids)}."
        )

    stem_layers = list(expected_stem_layers if expected_stem_layers is not None else getattr(model, "stem_layers", []))
    stem_layers = [int(layer_idx) for layer_idx in stem_layers]
    if not stem_layers:
        failures.append("No STEM layers were provided or detected.")
    if not hasattr(model, "_layer_to_stem_idx"):
        failures.append("Model has no _layer_to_stem_idx mapping.")
    if not hasattr(model, "stem_embeddings"):
        failures.append("Model has no stem_embeddings module list.")

    input_tensor = torch.tensor([original_ids], dtype=torch.long, device=device)
    source_tokens_at_positions = [
        int(original_ids[pos]) for pos in plan_positions if 0 <= pos < len(original_ids)
    ]
    source_token_ids_match_plan = source_tokens_at_positions == list(override.plan.source_token_ids)
    if not source_token_ids_match_plan:
        failures.append(
            "Source token IDs at edit positions do not match the override plan: "
            f"prompt={source_tokens_at_positions}, plan={override.plan.source_token_ids}."
        )
    selected_weight_ids = (
        list(override.plan.source_token_ids)
        + list(override.plan.target_token_ids)
        + list(override.plan.replacement_token_ids or [])
        + source_tokens_at_positions
    )

    layer_diagnostics: List[InterventionLayerDiagnostic] = []
    for layer_idx in stem_layers:
        layer_failures: List[str] = []
        try:
            embedding = override._embedding_for_layer(layer_idx)
            stem_idx = int(getattr(model, "_layer_to_stem_idx")[layer_idx])
            before_rows, before_version = _selected_weight_row_snapshot(embedding, selected_weight_ids)

            baseline = embedding(input_tensor)
            edited = override(layer_idx, input_tensor)
            replacement = override._lookup_target_vectors(
                embedding,
                input_tensor,
                output_dtype=edited.dtype,
                output_device=edited.device,
            )
            selected_rows_unchanged = _selected_weight_rows_unchanged(
                embedding, selected_weight_ids, before_rows
            )
            weight = getattr(embedding, "weight", None)
            after_version = int(getattr(weight, "_version", -1)) if weight is not None else before_version
            weight_version_unchanged = before_version == after_version

            pos_tensor = torch.tensor(plan_positions, dtype=torch.long, device=edited.device)
            unedited_mask = torch.ones(edited.shape[1], dtype=torch.bool, device=edited.device)
            if len(plan_positions) > 0:
                unedited_mask[pos_tensor] = False
            unedited_diff = _max_abs_or_zero(edited[:, unedited_mask, :] - baseline[:, unedited_mask, :])
            edited_expected_diff = _max_abs_or_zero(edited[:, pos_tensor, :] - replacement)
            edit_vs_baseline_diff = _max_abs_or_zero(edited[:, pos_tensor, :] - baseline[:, pos_tensor, :])
            unedited_ok = bool(torch.allclose(
                edited[:, unedited_mask, :],
                baseline[:, unedited_mask, :],
                atol=atol,
                rtol=rtol,
            ))
            edited_ok = bool(torch.allclose(
                edited[:, pos_tensor, :],
                replacement,
                atol=atol,
                rtol=rtol,
            ))
            output_is_distinct = edited.data_ptr() != baseline.data_ptr()

            if not unedited_ok:
                layer_failures.append(
                    "Edited output differs from baseline outside source positions "
                    f"(max_abs_diff={unedited_diff})."
                )
            if not edited_ok:
                layer_failures.append(
                    "Edited source-position vectors do not match expected target vectors "
                    f"(max_abs_diff={edited_expected_diff})."
                )
            if not output_is_distinct:
                layer_failures.append("Override returned the same tensor storage as the baseline lookup.")
            if not selected_rows_unchanged:
                layer_failures.append("Selected STEM embedding weight rows changed during diagnostic.")
            if not weight_version_unchanged:
                layer_failures.append("STEM embedding weight version changed during diagnostic.")

            layer_diagnostics.append(
                InterventionLayerDiagnostic(
                    layer_idx=layer_idx,
                    stem_embedding_index=stem_idx,
                    output_shape=[int(dim) for dim in edited.shape],
                    output_dtype=str(edited.dtype).replace("torch.", ""),
                    output_device=str(edited.device),
                    source_positions=plan_positions,
                    source_token_ids_at_positions=source_tokens_at_positions,
                    target_token_ids=list(override.plan.target_token_ids),
                    replacement_token_ids=override.plan.replacement_token_ids,
                    max_abs_diff_unedited_vs_baseline=unedited_diff,
                    max_abs_diff_edited_vs_expected=edited_expected_diff,
                    max_abs_diff_edit_vs_baseline=edit_vs_baseline_diff,
                    unedited_positions_match_baseline=unedited_ok,
                    edited_positions_match_expected=edited_ok,
                    output_is_distinct_tensor=output_is_distinct,
                    selected_weight_rows_unchanged=selected_rows_unchanged,
                    weight_version_unchanged=weight_version_unchanged,
                    passed=not layer_failures,
                    failures=layer_failures,
                )
            )
            failures.extend([f"Layer {layer_idx}: {message}" for message in layer_failures])
        except Exception as exc:
            message = f"Layer {layer_idx} diagnostic failed with exception: {exc}"
            failures.append(message)
            layer_diagnostics.append(
                InterventionLayerDiagnostic(
                    layer_idx=layer_idx,
                    stem_embedding_index=-1,
                    output_shape=[],
                    output_dtype="unknown",
                    output_device=str(device),
                    source_positions=plan_positions,
                    source_token_ids_at_positions=source_tokens_at_positions,
                    target_token_ids=list(override.plan.target_token_ids),
                    replacement_token_ids=override.plan.replacement_token_ids,
                    max_abs_diff_unedited_vs_baseline=float("nan"),
                    max_abs_diff_edited_vs_expected=float("nan"),
                    max_abs_diff_edit_vs_baseline=float("nan"),
                    unedited_positions_match_baseline=False,
                    edited_positions_match_expected=False,
                    output_is_distinct_tensor=False,
                    selected_weight_rows_unchanged=False,
                    weight_version_unchanged=False,
                    passed=False,
                    failures=[message],
                )
            )

    forward_path_checked = False
    forward_path_called_layers: List[int] = []
    forward_path_calls_match_stem_layers = False
    if run_forward_path_check and not failures:
        forward_path_checked = True
        traced = TracedStemEmbeddingOverride(override)
        try:
            _ = _call_model(model, input_tensor, traced, attn_impl=attn_impl)
            forward_path_called_layers = list(traced.calls)
            forward_path_calls_match_stem_layers = forward_path_called_layers == stem_layers
            if not forward_path_calls_match_stem_layers:
                failures.append(
                    "LM forward did not call the override exactly once for each STEM layer: "
                    f"called={forward_path_called_layers}, expected={stem_layers}."
                )
        except Exception as exc:
            failures.append(f"Forward-path diagnostic failed with exception: {exc}")
    elif not run_forward_path_check:
        warnings.append("Forward-path diagnostic was disabled by configuration.")

    passed = not failures and all(layer.passed for layer in layer_diagnostics)
    return InterventionDiagnostics(
        passed=passed,
        prompt_text_unchanged=prompt_text_unchanged,
        prompt_token_ids_unchanged=prompt_token_ids_unchanged,
        source_positions_match_plan=source_positions_match_plan,
        source_token_ids_match_plan=source_token_ids_match_plan,
        stem_layers_expected=stem_layers,
        stem_layers_checked=[layer.layer_idx for layer in layer_diagnostics],
        forward_path_checked=forward_path_checked,
        forward_path_called_layers=forward_path_called_layers,
        forward_path_calls_match_stem_layers=forward_path_calls_match_stem_layers,
        edit_mode=override.plan.resolved_mode,
        layer_diagnostics=layer_diagnostics,
        failures=failures,
        warnings=warnings,
    )


def _call_model(
    model: Any,
    input_ids: torch.Tensor,
    stem_embeddings_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    *,
    attn_impl: str = "sdpa",
) -> torch.Tensor:
    if stem_embeddings_fn is None:
        return model(input_ids, attn_impl=attn_impl)
    lm_transformer = getattr(model, "lm_transformer", None)
    if lm_transformer is None:
        raise RuntimeError("Intervened STEM forward requires model.lm_transformer")
    return lm_transformer(
        token_values=input_ids,
        attn_impl=attn_impl,
        stem_embeddings_fn=stem_embeddings_fn,
    )


@torch.no_grad()
def run_next_token_topk(
    model: Any,
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    case: str,
    prompt: str,
    top_k: int = 4,
    device: Union[str, torch.device] = "cuda",
    stem_embeddings_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    attn_impl: str = "sdpa",
) -> TopKResult:
    """Run a full prompt forward pass and return final-position top-k tokens."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    input_tensor = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    logits = _call_model(model, input_tensor, stem_embeddings_fn, attn_impl=attn_impl)
    final_logits = logits[0, -1].float()
    probabilities = torch.softmax(final_logits, dim=-1)
    k = min(int(top_k), int(probabilities.shape[-1]))
    top_probs, top_ids = torch.topk(probabilities, k=k)
    top_logits = final_logits[top_ids]
    tokens = [
        TopKToken(
            token_id=int(token_id),
            token_str=_decode_piece(tokenizer, int(token_id)),
            logit=float(logit),
            probability=float(prob),
        )
        for token_id, logit, prob in zip(top_ids.tolist(), top_logits.tolist(), top_probs.tolist())
    ]
    return TopKResult(
        case=case,
        prompt=prompt,
        prompt_token_ids=[int(x) for x in token_ids],
        top_k=k,
        tokens=tokens,
    )


def _filter_logits(
    logits: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    filtered = logits.clone()
    if top_k is not None and top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, k=int(top_k)).values[-1]
        filtered[filtered < threshold] = -float("inf")
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative > float(top_p)
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        indices_to_remove = sorted_indices[remove_sorted]
        filtered[indices_to_remove] = -float("inf")
    return filtered


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    generator: Optional[torch.Generator],
) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    scaled = logits.float() / float(temperature)
    filtered = _filter_logits(scaled, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0.0:
        raise RuntimeError("Sampling distribution became invalid")
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


@torch.no_grad()
def run_generation(
    model: Any,
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    case: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: int = 0,
    device: Union[str, torch.device] = "cuda",
    eos_token_id: Optional[int] = None,
    stem_embeddings_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    attn_impl: str = "sdpa",
) -> GenerationResult:
    """Generate a continuation by recomputing the full forward pass each step."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    current_ids = [int(x) for x in token_ids]
    generated: List[int] = []
    stop_reason = "max_new_tokens"
    try:
        generator = torch.Generator(device=torch.device(device))
    except Exception:
        generator = torch.Generator()
    generator.manual_seed(int(seed))

    for _ in range(int(max_new_tokens)):
        input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
        logits = _call_model(model, input_tensor, stem_embeddings_fn, attn_impl=attn_impl)
        next_logits = logits[0, -1]
        next_token = _sample_next_token(
            next_logits,
            temperature=float(temperature),
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        generated.append(next_token)
        current_ids.append(next_token)
        if eos_token_id is not None and next_token == int(eos_token_id):
            stop_reason = "eos"
            break

    continuation_text = _decode(tokenizer, generated)
    full_text = _decode(tokenizer, current_ids)
    return GenerationResult(
        case=case,
        prompt=prompt,
        prompt_token_ids=[int(x) for x in token_ids],
        generated_token_ids=generated,
        continuation_text=continuation_text,
        full_text=full_text,
        stop_reason=stop_reason,
    )


def _token_label(text: str, token_id: int) -> str:
    if text == "":
        return f"<empty:{token_id}>"
    escaped = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
    escaped = escaped.replace("\r", "\\r")
    if escaped.strip() == "":
        escaped = repr(text)[1:-1]
    if len(escaped) > 24:
        escaped = escaped[:21] + "..."
    return escaped


def plot_topk_probabilities(
    topk_results: Mapping[str, TopKResult],
    *,
    output_png: Union[str, Path],
    output_pdf: Union[str, Path],
    source_entity: str,
    target_entity: str,
    prompt_type: str = "country-capital",
) -> None:
    """Save Figure-7-style side-by-side top-k probability bar charts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["original", "target", "intervened"]
    if prompt_type == "math-text":
        titles = {
            "original": f"Original: operator {source_entity}",
            "target": f"Target: operator {target_entity}",
            "intervened": f"Intervened: {source_entity} text + {target_entity} STEM",
        }
    else:
        titles = {
            "original": f"Original: Country: {source_entity}",
            "target": f"Target: Country: {target_entity}",
            "intervened": f"Intervened: {source_entity} text + {target_entity} STEM",
        }
    colors = {
        "original": "#4C78A8",
        "target": "#F58518",
        "intervened": "#54A24B",
    }

    max_prob = 0.0
    for result in topk_results.values():
        for token in result.tokens:
            max_prob = max(max_prob, token.probability)
    y_limit = min(1.0, max(0.05, max_prob * 1.18))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharey=True)
    for ax, case in zip(axes, order):
        result = topk_results[case]
        labels = [_token_label(tok.token_str, tok.token_id) for tok in result.tokens]
        probs = [tok.probability for tok in result.tokens]
        x_values = list(range(len(labels)))
        ax.bar(x_values, probs, color=colors[case], edgecolor="black", linewidth=0.6)
        ax.set_title(titles[case], fontsize=10.5, pad=10)
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(0, y_limit)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.tick_params(axis="y", labelsize=9)
        for idx, prob in enumerate(probs):
            ax.text(idx, prob + y_limit * 0.025, f"{prob:.3f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylabel("Next-token probability", fontsize=10)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj
    return str(obj)


def write_json(path: Union[str, Path], payload: Any) -> None:
    Path(path).write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def save_results(
    output_dir: Union[str, Path],
    *,
    metadata: Mapping[str, Any],
    prompts: Mapping[str, str],
    tokenization: Mapping[str, TokenizedPrompt],
    topk_results: Mapping[str, TopKResult],
    generations: Mapping[str, GenerationResult],
) -> None:
    """Write all structured experiment outputs except plots."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if "original" in prompts:
        (out / "prompt_original.txt").write_text(prompts["original"])
    if "target" in prompts:
        (out / "prompt_target.txt").write_text(prompts["target"])

    topk_payload = {case: result for case, result in topk_results.items()}
    generation_payload = {case: result for case, result in generations.items()}
    write_json(out / "metadata.json", metadata)
    write_json(out / "topk_next_token_probs.json", topk_payload)
    write_json(
        out / "full_results.json",
        {
            "metadata": metadata,
            "prompts": prompts,
            "tokenization": tokenization,
            "topk_next_token_probs": topk_payload,
            "generations": generation_payload,
        },
    )

    with (out / "generations.jsonl").open("w") as f:
        for case in ("original", "target", "intervened"):
            if case in generations:
                f.write(json.dumps(_jsonable(generations[case]), sort_keys=True) + "\n")

    lines: List[str] = []
    for case in ("original", "target", "intervened"):
        if case not in generations:
            continue
        result = generations[case]
        lines.append(f"## {case}")
        lines.append(result.full_text)
        lines.append("")
    (out / "generations.txt").write_text("\n".join(lines).rstrip() + "\n")


def sanitize_for_path(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "entity"


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
