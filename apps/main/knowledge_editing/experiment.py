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


@dataclass(frozen=True)
class CountryCapitalExample:
    country: str
    capital_sentence: str


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
) -> None:
    """Save Figure-7-style side-by-side top-k probability bar charts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["original", "target", "intervened"]
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
