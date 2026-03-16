from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from lingua.stem_dist_utils import ParallelEmbedding
from lingua.tokenizer import CompressedTokenizer, Tokenizer
from apps.main.stem import (
    StemLMTransformerArgs,
    StemLMTransformer,
    LMTransformer,
    Qwen3LMTransformer,
    OLMo3LMTransformer,
    build_stem_lm_fsdp_grouping_plan,
    build_qwen3_stem_lm_fsdp_grouping_plan,
    build_olmo3_stem_lm_fsdp_grouping_plan,
    llama_get_no_recompute_ops,
    llama_get_num_flop_per_token,
    qwen3_get_no_recompute_ops,
    qwen3_get_num_flop_per_token,
    olmo3_get_no_recompute_ops,
    olmo3_get_num_flop_per_token,
)


@dataclass
class CompressedStemLMTransformerArgs(StemLMTransformerArgs):
    # Number of rows in stem embedding tables. If None, train wrapper sets it.
    stem_vocab_size: Optional[int] = None


class CompressedStemLMTransformer(StemLMTransformer):
    """
    STEM variant that indexes stem embedding tables using CompressedTokenizer IDs.
    Backbone token embeddings and logits still use original token IDs.
    """

    _lm_transformer_cls = LMTransformer
    _default_lookup_cpu: Optional[torch.Tensor] = None

    @classmethod
    def set_default_lookup_table(cls, lookup_table: Union[np.ndarray, torch.Tensor]):
        if isinstance(lookup_table, np.ndarray):
            lookup = torch.from_numpy(lookup_table.astype(np.int64, copy=False))
        elif isinstance(lookup_table, torch.Tensor):
            lookup = lookup_table.to(dtype=torch.long, device="cpu")
        else:
            raise TypeError(f"Unsupported lookup table type: {type(lookup_table)}")
        cls._default_lookup_cpu = lookup.contiguous()

    @classmethod
    def clear_default_lookup_table(cls):
        cls._default_lookup_cpu = None

    def __init__(self, args: CompressedStemLMTransformerArgs):
        nn.Module.__init__(self)
        self.args = args
        self.lm_transformer = self._lm_transformer_cls(args)

        assert args.stem_embedding_dim is not None
        stem_vocab_size = args.stem_vocab_size or args.vocab_size
        assert stem_vocab_size > 0

        device = next(iter(self.lm_transformer.parameters())).device
        self.stem_embeddings = nn.ModuleList(
            [
                ParallelEmbedding(stem_vocab_size, args.stem_embedding_dim, device=device)
                for _ in range(len(self.lm_transformer.stem_layers))
            ]
        )
        self._layer_to_stem_idx = {
            layer_idx: stem_idx
            for stem_idx, layer_idx in enumerate(self.lm_transformer.stem_layers)
        }

        self._compressed_lookup_cpu: Optional[torch.Tensor] = None
        self._compressed_lookup_device_cache: Optional[torch.Tensor] = None
        self._compressed_lookup_device: Optional[torch.device] = None

        default_lookup = type(self)._default_lookup_cpu
        if default_lookup is not None:
            self.set_compressed_lookup_table(default_lookup)

    def set_compressed_lookup_table(self, lookup_table: Union[np.ndarray, torch.Tensor]):
        if isinstance(lookup_table, np.ndarray):
            lookup = torch.from_numpy(lookup_table.astype(np.int64, copy=False))
        elif isinstance(lookup_table, torch.Tensor):
            lookup = lookup_table.to(dtype=torch.long, device="cpu")
        else:
            raise TypeError(f"Unsupported lookup table type: {type(lookup_table)}")

        if lookup.ndim != 1:
            raise ValueError("Compressed lookup table must be 1-D")
        if lookup.numel() != self.args.vocab_size:
            raise ValueError(
                f"Compressed lookup length ({lookup.numel()}) must equal "
                f"vocab_size ({self.args.vocab_size})"
            )

        self._compressed_lookup_cpu = lookup.contiguous()
        self._compressed_lookup_device_cache = None
        self._compressed_lookup_device = None

    def configure_stem_tokenizer(self, tokenizer: Tokenizer):
        compressed_tokenizer = CompressedTokenizer(tokenizer)
        compressed_vocab_size = len(compressed_tokenizer)
        expected = self.stem_embeddings[0].weight.shape[0]
        if expected != compressed_vocab_size:
            raise ValueError(
                f"stem_vocab_size mismatch: model has {expected}, "
                f"CompressedTokenizer has {compressed_vocab_size}. "
                "Set model.stem_vocab_size accordingly."
            )
        self.set_compressed_lookup_table(compressed_tokenizer.lookup_table)

    def _compressed_lookup_on(self, device: torch.device) -> torch.Tensor:
        if self._compressed_lookup_cpu is None:
            raise RuntimeError(
                "Compressed lookup table is not initialized. "
                "Use stem_compressed_train or call configure_stem_tokenizer()."
            )
        if (
            self._compressed_lookup_device_cache is None
            or self._compressed_lookup_device != device
        ):
            self._compressed_lookup_device_cache = self._compressed_lookup_cpu.to(
                device=device, non_blocking=True
            )
            self._compressed_lookup_device = device
        return self._compressed_lookup_device_cache

    def _compress_token_values(self, token_values: torch.Tensor) -> torch.Tensor:
        lookup = self._compressed_lookup_on(token_values.device)
        compressed = token_values.clone()
        mask = compressed >= 0
        compressed[mask] = lookup[compressed[mask]]
        return compressed

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        compressed_token_values = self._compress_token_values(token_values)

        def stem_embeddings_fn(layer_idx: int, _token_values: torch.Tensor) -> torch.Tensor:
            stem_idx = self._layer_to_stem_idx[layer_idx]
            return self.stem_embeddings[stem_idx](compressed_token_values)

        return self.lm_transformer(
            token_values=token_values,
            target=target,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
            stem_embeddings_fn=stem_embeddings_fn,
        )


class Qwen3CompressedStemLMTransformer(CompressedStemLMTransformer):
    _lm_transformer_cls = Qwen3LMTransformer


class OLMo3CompressedStemLMTransformer(CompressedStemLMTransformer):
    _lm_transformer_cls = OLMo3LMTransformer


def set_default_lookup_table_for_all(lookup_table: Union[np.ndarray, torch.Tensor]):
    for cls in (
        CompressedStemLMTransformer,
        Qwen3CompressedStemLMTransformer,
        OLMo3CompressedStemLMTransformer,
    ):
        cls.set_default_lookup_table(lookup_table)


def clear_default_lookup_table_for_all():
    for cls in (
        CompressedStemLMTransformer,
        Qwen3CompressedStemLMTransformer,
        OLMo3CompressedStemLMTransformer,
    ):
        cls.clear_default_lookup_table()


COMPRESSED_STEM_MODEL_REGISTRY = {
    "llama_compressed": (
        CompressedStemLMTransformer,
        CompressedStemLMTransformerArgs,
        build_stem_lm_fsdp_grouping_plan,
        llama_get_no_recompute_ops,
        llama_get_num_flop_per_token,
    ),
    "qwen3_compressed": (
        Qwen3CompressedStemLMTransformer,
        CompressedStemLMTransformerArgs,
        build_qwen3_stem_lm_fsdp_grouping_plan,
        qwen3_get_no_recompute_ops,
        qwen3_get_num_flop_per_token,
    ),
    "olmo3_compressed": (
        OLMo3CompressedStemLMTransformer,
        CompressedStemLMTransformerArgs,
        build_olmo3_stem_lm_fsdp_grouping_plan,
        olmo3_get_no_recompute_ops,
        olmo3_get_num_flop_per_token,
    ),
}
