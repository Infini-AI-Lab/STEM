from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from Engine.model import (
    Attention,
    FeedForward,
    ModelArgs,
    RMSNorm,
    Transformer as TransformerBase,
)


class StemTransformer(TransformerBase):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.stem_layer_ids = config.stem_layers
        self.num_stem_layers = len(self.stem_layer_ids)
        self.stem_buffer_size = config.lfu_buffer_size
        self.stem_embedding_dim = config.intermediate_size
        self.layers = nn.ModuleList(
            TransformerBlock(config, 
                             use_stem=i in self.stem_layer_ids) 
            for i in range(config.n_layer)
        )
        
    def setup_stem(self):
        self.cpu_stem_embeddings = torch.empty(
            (self.config.vocab_size, 
             self.config.intermediate_size * self.num_stem_layers),
            device="cpu",
            dtype=torch.bfloat16,
            pin_memory=True,
        )
        print("Allocated CPU STEM embeddings for each layer")
        
        # ---- Pinned CPU staging (dedup gather result) ----
        self._cpu_stem_stage = torch.empty(
            (self.num_stem_layers, self.stem_buffer_size, self.stem_embedding_dim),
            device="cpu",
            dtype=torch.bfloat16,
            pin_memory=True,
        )
        
        # ---- GPU double buffers (what the layer forward uses) ----
        self.stem_gpu_buf0 = torch.empty(
            (self.num_stem_layers, self.stem_buffer_size, self.stem_embedding_dim),
            dtype=torch.bfloat16,
        )
        self.stem_gpu_buf1 = torch.empty_like(self.stem_gpu_buf0)
        
        # ---- Prefetch stream + events (MUST be distinct events) ----
        self._prefetch_stream = torch.cuda.Stream()
        self._prefetch_events = [torch.cuda.Event(), torch.cuda.Event()]  # two distinct events
        self._cur_buf_idx = 0
        
        # ---- Persistent CPU worker thread for prefetching ----
        self._cpu_executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future: Optional[Future] = None
        
        print("Prepared STEM buffers")
        
    def _cur_gpu_buf(self) -> torch.Tensor:
        return self.stem_gpu_buf0 if self._cur_buf_idx == 0 else self.stem_gpu_buf1

    def _next_gpu_buf(self) -> torch.Tensor:
        return self.stem_gpu_buf1 if self._cur_buf_idx == 0 else self.stem_gpu_buf0

    def _swap_gpu_bufs(self):
        self._cur_buf_idx = 1 - self._cur_buf_idx

    def _cpu_gather_work(self, token_ids_cpu: Tensor, buf_idx: int):
        """Worker thread function: performs deduplication, indexing, and H2D copy entirely in worker thread."""
        # ---- Deduplicate on CPU ----
        unique_ids_cpu, inv_cpu = torch.unique(token_ids_cpu, sorted=False, return_inverse=True)
        U = int(unique_ids_cpu.numel())
        assert U <= self.stem_buffer_size, "Too many unique tokens for STEM buffer"
        
        # ---- Gather from CPU embedding table ----
        cpu_tbl = self.cpu_stem_embeddings.view(
            self.config.vocab_size, self.num_stem_layers, self.stem_embedding_dim
        )
        tmp = cpu_tbl.index_select(0, unique_ids_cpu)
        stage = self._cpu_stem_stage
        stage[:, :U, :].copy_(tmp.permute(1, 0, 2))
        
        # ---- Async H2D into the "next" GPU buffer (worker thread, CUDA context) ----
        dst = self.stem_gpu_buf1 if buf_idx == 0 else self.stem_gpu_buf0
        next_event = self._prefetch_events[1 - buf_idx]
        print(f"current buffer: {buf_idx}")
        
        with torch.cuda.stream(self._prefetch_stream):
            dst[:, :U, :].copy_(stage[:, :U, :], non_blocking=True)
            next_event.record(self._prefetch_stream)
        
        # Convert buffer_ids to GPU tensor
        buffer_ids = torch.as_tensor(inv_cpu, device=dst.device, dtype=torch.int32)
        
        return buffer_ids
    
    @torch._dynamo.disable()
    def prefetch_stem_async(self, token_ids: Tensor) -> Future:
        """Submit CPU gather + H2D work to persistent worker thread, returns Future for later consumption.
        
        Pattern: submit CPU gather + H2D for chunk i+1 immediately, then later consume the future
        to get buffer_ids. All work (indexing + H2D) happens in the worker thread.
        """
        token_ids_cpu = token_ids.contiguous()  # cpu ids
        
        # ---- Submit CPU work + H2D to persistent worker (non-blocking) ----
        # Pass current buffer index so worker knows which buffer to write to
        future = self._cpu_executor.submit(
            self._cpu_gather_work, token_ids_cpu, self._cur_buf_idx
        )
        return future
    
    @torch._dynamo.disable()
    def prefetch_stem_complete(self, future: Future) -> Tensor:
        """Complete prefetch by waiting for worker thread to finish (indexing + H2D)."""
        # ---- Wait for worker thread to complete (CPU indexing + H2D copy) ----
        buffer_ids = future.result()
        return buffer_ids
    
    @torch._dynamo.disable()
    def prefetch_stem(self, token_ids: Tensor):
        """Submit CPU gather work to persistent worker thread, then do H2D copy in main thread.
        
        Convenience method that combines async submission and completion.
        For better overlap, use prefetch_stem_async() + prefetch_stem_complete() separately.
        """
        future = self.prefetch_stem_async(token_ids)
        return self.prefetch_stem_complete(future)
    
    @torch._dynamo.disable()
    def _wait_prefetch(self):
        self._prefetch_events[1 - self._cur_buf_idx].wait(self._prefetch_stream)

    
    def forward(
        self, 
        idx: Tensor,
        buffer_ids: Tensor,
        offsets: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
    ) -> Tensor:
        x = self.tok_embeddings(idx)
        all_buffers = self._cur_gpu_buf()
        for i, layer in enumerate(self.layers):
            if i in self.stem_layer_ids:
                buffer = all_buffers[self.stem_layer_ids.index(i)]
                x = layer(x, buffer, buffer_ids, offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
            else:
                x = layer(x, None, None, offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        x = self.norm(x)
        logits = self.output(x)
        return logits
        
    def prefill(
        self,
        idx: Tensor,
        buffer_ids: Tensor,
        offsets: Tensor,
        kv_append_indptr: Tensor,
        kv_page_indices: Tensor,
        kv_page_indptr: Tensor,
        kv_page_lastlen: Tensor,
    ) -> Tensor:
        x = self.tok_embeddings(idx)
        all_buffers = self._cur_gpu_buf()
        for i, layer in enumerate(self.layers):
            if i in self.stem_layer_ids:
                buffer = all_buffers[self.stem_layer_ids.index(i)]
                x = layer.prefill(x, buffer, buffer_ids, offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
            else:
                x = layer.prefill(x, None, None, offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        x = self.norm(x)
        logits = self.output(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, use_stem: bool = False) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = STEMFeedForward(config) if use_stem else FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.use_stem = use_stem

    def forward(
        self,
        x: Tensor,
        buffer: Optional[Tensor],
        buffer_ids: Optional[Tensor],
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
        if self.use_stem:
            out = h + self.feed_forward(self.ffn_norm(h), buffer, buffer_ids)
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def prefill(
        self,
        x: Tensor,
        buffer: Optional[Tensor],
        buffer_ids: Optional[Tensor],
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
        if self.use_stem:
            out = h + self.feed_forward(self.ffn_norm(h), buffer, buffer_ids)
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    
class STEMFeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.process_group = None
        
    def forward(self, x: Tensor, buffer: Tensor, buffer_ids: Tensor) -> Tensor:
        h = F.silu(self.w1(x))
        stem = F.embedding(buffer_ids, buffer)
        y = self.w2(h * stem)
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y    
    
    
    