# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Sliding-Window STEM language-model wrappers.

This variant replaces raw tokenwise STEM embeddings with a causal trailing
window average before feeding them to STEM FFN gates.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from lingua.stem_window import WindowStemTransformerArgs, SlidingWindowMemory
from lingua.stem_dist_utils import ParallelEmbedding

from apps.main.stem import (
    LMTransformer,
    Qwen3LMTransformer,
    OLMo3LMTransformer,
    build_stem_lm_fsdp_grouping_plan,
    build_qwen3_stem_lm_fsdp_grouping_plan,
    build_olmo3_stem_lm_fsdp_grouping_plan,
)
from apps.main.transformer import (
    LMTransformerArgs,
    get_no_recompute_ops as llama_get_no_recompute_ops,
    get_num_flop_per_token as llama_get_num_flop_per_token,
)
from apps.main.qwen3 import (
    get_no_recompute_ops as qwen3_get_no_recompute_ops,
    get_num_flop_per_token as qwen3_get_num_flop_per_token,
)
from apps.main.olmo3 import (
    get_no_recompute_ops as olmo3_get_no_recompute_ops,
    get_num_flop_per_token as olmo3_get_num_flop_per_token,
)


@dataclass
class WindowStemLMTransformerArgs(WindowStemTransformerArgs, LMTransformerArgs):
    """Combined args for sliding-window STEM language-model transformers."""

    pass


class WindowStemLMTransformer(nn.Module):
    """STEM LM wrapper with per-layer causal sliding-window memories."""

    _lm_transformer_cls = LMTransformer

    def __init__(self, args: WindowStemLMTransformerArgs):
        super().__init__()
        self.args = args

        self.lm_transformer = self._lm_transformer_cls(args)

        assert args.stem_embedding_dim is not None, "stem_embedding_dim must be provided"
        device = next(iter(self.lm_transformer.parameters())).device
        self.stem_embeddings = nn.ModuleList([
            ParallelEmbedding(
                args.vocab_size, args.stem_embedding_dim, device=device,
            )
            for _ in range(len(self.lm_transformer.stem_layers))
        ])

        self.window_memories = nn.ModuleList([
            SlidingWindowMemory(
                d_ff=args.stem_embedding_dim,
                window_size=args.stem_window_size,
            )
            for _ in range(len(self.lm_transformer.stem_layers))
        ])

        self._layer_to_stem_idx = {
            layer_idx: stem_idx
            for stem_idx, layer_idx in enumerate(self.lm_transformer.stem_layers)
        }

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        def stem_embeddings_fn(
            layer_idx: int, token_values: torch.Tensor,
        ) -> torch.Tensor:
            stem_idx = self._layer_to_stem_idx[layer_idx]
            e = self.stem_embeddings[stem_idx](token_values)
            return self.window_memories[stem_idx](e, tok_idx=tok_idx)

        return self.lm_transformer(
            token_values=token_values,
            target=target,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
            stem_embeddings_fn=stem_embeddings_fn,
        )

    @torch.no_grad()
    def reset_stem_embeddings(self):
        import logging

        logger = logging.getLogger()
        for i, embedding in enumerate(self.stem_embeddings):
            weight_device = embedding.weight.device
            if weight_device.type == "meta":
                logger.warning(
                    f"stem_embeddings[{i}].weight on meta device, skipping init"
                )
                continue
            embedding.reset_parameters()
            if embedding.weight.numel() > 0:
                weight_norm = embedding.weight.norm().item()
                is_zero = (embedding.weight.abs().max() == 0).item()
                if is_zero:
                    logger.error(
                        f"stem_embeddings[{i}].weight still zeros after reset!"
                    )
                else:
                    logger.debug(
                        f"stem_embeddings[{i}].weight initialised: "
                        f"norm={weight_norm:.6f}, device={weight_device}"
                    )

    def init_weights(self):
        self.lm_transformer.init_weights()
        self.reset_stem_embeddings()
        for window_mem in self.window_memories:
            window_mem.reset_parameters()

    def stem_parameters(self):
        yield from self.stem_embeddings.parameters()
        yield from self.window_memories.parameters()

    @property
    def layers(self):
        return self.lm_transformer.layers

    @property
    def stem_layers(self):
        return self.lm_transformer.stem_layers

    @property
    def rope_embeddings(self):
        return self.lm_transformer.rope_embeddings

    @property
    def max_seqlen(self):
        return self.lm_transformer.max_seqlen

    @property
    def dim(self):
        return self.lm_transformer.dim

    @property
    def weight_tying(self):
        return self.lm_transformer.weight_tying

    @property
    def sliding_window(self):
        return self.lm_transformer.sliding_window

    def set_requires_gradient_sync(self, requires_sync: bool):
        if hasattr(self.lm_transformer, "set_requires_gradient_sync"):
            self.lm_transformer.set_requires_gradient_sync(requires_sync)


class Qwen3WindowStemLMTransformer(WindowStemLMTransformer):
    _lm_transformer_cls = Qwen3LMTransformer


class OLMo3WindowStemLMTransformer(WindowStemLMTransformer):
    _lm_transformer_cls = OLMo3LMTransformer


def build_window_stem_lm_fsdp_grouping_plan(
    model_args: WindowStemLMTransformerArgs,
):
    return build_stem_lm_fsdp_grouping_plan(model_args)


def build_qwen3_window_stem_lm_fsdp_grouping_plan(
    model_args: WindowStemLMTransformerArgs,
):
    return build_qwen3_stem_lm_fsdp_grouping_plan(model_args)


def build_olmo3_window_stem_lm_fsdp_grouping_plan(
    model_args: WindowStemLMTransformerArgs,
):
    return build_olmo3_stem_lm_fsdp_grouping_plan(model_args)


WINDOW_STEM_MODEL_REGISTRY = {
    "llama_window": (
        WindowStemLMTransformer,
        WindowStemLMTransformerArgs,
        build_window_stem_lm_fsdp_grouping_plan,
        llama_get_no_recompute_ops,
        llama_get_num_flop_per_token,
    ),
    "qwen3_window": (
        Qwen3WindowStemLMTransformer,
        WindowStemLMTransformerArgs,
        build_qwen3_window_stem_lm_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops,
        qwen3_get_num_flop_per_token,
    ),
    "olmo3_window": (
        OLMo3WindowStemLMTransformer,
        WindowStemLMTransformerArgs,
        build_olmo3_window_stem_lm_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops,
        olmo3_get_num_flop_per_token,
    ),
}
