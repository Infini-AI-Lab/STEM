from dataclasses import dataclass
from typing import Optional, List

import flashinfer
import torch
import torch.distributed as dist
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F



def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    scaling_factor: float = 1.0
    # llama 3.1 with high_freq_factor and low_freq_factor
    low_freq_factor: int = None  # added new
    high_freq_factor: int = None  # added new
    original_max_position_embeddings: int = None  # added new
    qkv_bias: bool = False
    # STEM
    stem_layers: Optional[List[int]] = None
    lfu_buffer_size: int = 2048

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])
    
    
transformer_configs = {
    "llama-3.1-8b": dict(
        block_size=131072,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000.0,
        scaling_factor=8,
        high_freq_factor=4,
        low_freq_factor=1,
        original_max_position_embeddings=8192,
    ),
    "llama-3.2-1b": dict(
        block_size=131072,
        n_layer=16,
        n_head=32,
        n_local_heads=8,
        dim=2048,
        intermediate_size=8192,
        vocab_size=128256,
        rope_base=500000.0,
        scaling_factor=32,
        high_freq_factor=4,
        low_freq_factor=1,
        original_max_position_embeddings=8192,
    ),
    "llama-3.2-1b": dict(
        block_size=131072,
        n_layer=16,
        n_head=32,
        n_local_heads=8,
        dim=2048,
        intermediate_size=8192,
        vocab_size=128256,
        rope_base=500000.0,
        scaling_factor=32,
        high_freq_factor=4,
        low_freq_factor=1,
        original_max_position_embeddings=8192,
        stem_layers=list(range(1, 16, 2)),
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_num_pages, page_size, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_num_pages, 2, page_size, n_heads, head_dim)
        self.register_buffer("kv_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(
        self, k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen
    ):
        torch.ops.mylib.update_kv(
            k,
            v,
            kv_append_indptr,
            self.kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        return self.kv_cache
    
    
    
    
class Transformer(nn.Module):
    
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.world_size = None
        self.rank = None
        self.process_group = None

    def setup_caches(self, num_pages, page_size):
        head_dim = self.config.dim // self.config.n_head
        dtype = (
            self.output.weight.dtype
            if self.output.weight.dtype == torch.float16
            else torch.bfloat16
        )

        if (self.config.high_freq_factor is not None) and (
            self.config.low_freq_factor is not None
        ):
            torch.library.define(
                "mylib::rope",
                "(Tensor q, Tensor k, Tensor indptr, Tensor offsets) -> (Tensor ropeq, Tensor ropek)",
            )

            @torch.library.impl("mylib::rope", "cuda")
            def rope(q, k, indptr, offsets):
                return flashinfer.rope.apply_llama31_rope(
                    q,
                    k,
                    indptr,
                    offsets,
                    interleave=True,
                    rope_scale=self.config.scaling_factor,
                    rope_theta=self.config.rope_base,
                    low_freq_factor=self.config.low_freq_factor,
                    high_freq_factor=self.config.high_freq_factor,
                    old_context_len=self.config.original_max_position_embeddings,
                )

            @torch.library.register_fake("mylib::rope")
            def rope_abstract(q, k, indptr, offsets):
                return torch.empty_like(q), torch.empty_like(k)

        else:
            torch.library.define(
                "mylib::rope",
                "(Tensor q, Tensor k, Tensor indptr, Tensor offsets) -> (Tensor ropeq, Tensor ropek)",
            )

            @torch.library.impl("mylib::rope", "cuda")
            def rope(q, k, indptr, offsets):
                return flashinfer.rope.apply_rope(
                    q,
                    k,
                    indptr,
                    offsets,
                    interleave=True,
                    rope_scale=self.config.scaling_factor,
                    rope_theta=self.config.rope_base,
                )

            @torch.library.register_fake("mylib::rope")
            def rope_abstract(q, k, indptr, offsets):
                return torch.empty_like(q), torch.empty_like(k)

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                num_pages, page_size, self.config.n_local_heads, head_dim, dtype
            )
            b.attention.attn_decode = torch.ops.mylib.decode
            b.attention.attn_prefill = torch.ops.mylib.prefill
            b.attention.rope = torch.ops.mylib.rope
            
    def forward(
        self,
        idx: Tensor,
        input_pos: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
    ) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                input_pos,
                kv_append_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_page_lastlen,
            )
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    def prefill(
        self,
        idx: Tensor,
        input_pos: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
    ) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.prefill(
                x,
                input_pos,
                kv_append_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_page_lastlen,
            )
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))
    
    
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        
    def forward(
        self,
        x: Tensor,
        offsets: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
    ) -> Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            offsets,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def prefill(
        self,
        x: Tensor,
        offsets: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
    ) -> Tensor:
        h = x + self.attention.prefill(
            self.attention_norm(x),
            offsets,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    
class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=config.qkv_bias)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.process_group = None
        self.attn_decode = None
        self.attn_prefill = None
        self.rope = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

        if prefix + "wq.bias" in state_dict:
            bq = state_dict.pop(prefix + "wq.bias")
            bk = state_dict.pop(prefix + "wk.bias")
            bv = state_dict.pop(prefix + "wv.bias")
            state_dict[prefix + "wqkv.bias"] = torch.cat([bq, bk, bv])

    def forward(
        self,
        x: Tensor,
        offsets: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update(
            k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen
        )
        y = self.attn_decode(q, kv_cache)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y

    def prefill(
        self,
        x: Tensor,
        offsets: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_head, self.head_dim)
        k = k.view(bsz * seqlen, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(bsz * seqlen, self.n_local_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update(
            k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen
        )
        y = self.attn_prefill(q, kv_cache)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y
    
    
class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.process_group = None

    def forward(self, x: Tensor) -> Tensor:
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight