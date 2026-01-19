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
from Engine.setup import stem_gather_cpu


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
        
        self._layer_to_stem_slot = [-1] * config.n_layer
        for slot, lid in enumerate(self.stem_layer_ids):
            self._layer_to_stem_slot[lid] = slot
            
        self._cpu_executor: Optional[ThreadPoolExecutor] = None
        
    def setup_stem(self, max_batched_tokens: int):
        self.cpu_stem_embeddings = torch.empty(
            (self.config.vocab_size, 
             self.config.intermediate_size * self.num_stem_layers),
            device="cpu",
            dtype=torch.bfloat16,
            pin_memory=True,
        )
        print("Allocated CPU STEM embeddings for each layer")
        
        # ---- Pinned CPU staging (dedup gather result) ----
        self._cpu_stem_stage0 = torch.empty(
            (self.num_stem_layers, self.stem_buffer_size, self.stem_embedding_dim),
            device="cpu",
            dtype=torch.bfloat16,
            pin_memory=True,
        )
        self._cpu_stem_stage1 = torch.empty_like(self._cpu_stem_stage0)
        
        # ---- Pinned CPU inv tensor ----
        self._cpu_inv0 = torch.empty(
            (max_batched_tokens,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=True,
        )
        self._cpu_inv1 = torch.empty_like(self._cpu_inv0)
        
        # ---- GPU double buffers (what the layer forward uses) ----
        self.stem_gpu_buf0 = torch.empty(
            (self.num_stem_layers, self.stem_buffer_size, self.stem_embedding_dim),
            dtype=torch.bfloat16,
        )
        self.stem_gpu_buf1 = torch.empty_like(self.stem_gpu_buf0)
        
        # GPU inv tensors 
        self._gpu_inv0 = torch.empty((max_batched_tokens,), dtype=torch.int32)
        self._gpu_inv1 = torch.empty_like(self._gpu_inv0)
        
        # Metadata for each buf (so we can slice correct lengths)
        self._buf_T = [0, 0]  # number of tokens in chunk (T)
        self._buf_U = [0, 0]  # number of unique tokens (U)
        
        # ---- Prefetch stream + events (MUST be distinct events) ----
        self._prefetch_stream = torch.cuda.Stream()
        self._prefetch_events = [torch.cuda.Event(), torch.cuda.Event()]  # two distinct events
        self._cur_buf_idx = 0
        
        # ---- Persistent CPU worker thread for prefetching ----
        self._cpu_executor = ThreadPoolExecutor(max_workers=1)
        
        # ---- Fused gather bookkeeping buffers ----
        self._cpu_seen = torch.zeros((self.config.vocab_size,), device="cpu", dtype=torch.int32)
        self._cpu_slot = torch.empty_like(self._cpu_seen)
        self._cpu_epoch = 1    
        
        self._cpu_unique0 = torch.empty((self.stem_buffer_size,), device="cpu", dtype=torch.int32)
        self._cpu_unique1 = torch.empty_like(self._cpu_unique0)
        
        self._max_batched_tokens = int(max_batched_tokens)    
        
        print("Prepared STEM buffers")
        
    # ---------------------------
    # Buffer helpers
    # ---------------------------
    def _gpu_embed_buf(self, buf_idx: int) -> Tensor:
        return self.stem_gpu_buf0 if buf_idx == 0 else self.stem_gpu_buf1

    def _cpu_stage(self, buf_idx: int) -> Tensor:
        return self._cpu_stem_stage0 if buf_idx == 0 else self._cpu_stem_stage1

    def _cpu_inv_stage(self, buf_idx: int) -> Tensor:
        return self._cpu_inv0 if buf_idx == 0 else self._cpu_inv1

    def _gpu_inv_buf(self, buf_idx: int) -> Tensor:
        return self._gpu_inv0 if buf_idx == 0 else self._gpu_inv1
    
    def _cpu_unique_out(self, buf_idx: int) -> torch.Tensor:
        return self._cpu_unique0 if buf_idx == 0 else self._cpu_unique1

    def _cpu_gather_work(self, token_ids_cpu: Tensor, dst_buf_idx: int):
        cpu_tbl = self.cpu_stem_embeddings.view(
            self.config.vocab_size, self.num_stem_layers, self.stem_embedding_dim
        )  # [vocab, L, D] bfloat16 pinned

        stage = self._cpu_stage(dst_buf_idx)               # pinned [L, Umax, D]
        inv_stage = self._cpu_inv_stage(dst_buf_idx)       # pinned [Tmax]
        unique_out = self._cpu_unique_out(dst_buf_idx)     # [Umax] int32

        # Call fused C++ op
        U, T = stem_gather_cpu(
            token_ids_cpu,
            cpu_tbl,
            stage,
            inv_stage,
            unique_out,
            self._cpu_seen,
            self._cpu_slot,
            self._cpu_epoch,
        )

        # bump epoch (wrap around safely; very unlikely to wrap in practice)
        self._cpu_epoch += 1
        if self._cpu_epoch == 2**31 - 1:
            self._cpu_seen.zero_()
            self._cpu_epoch = 1

        return dst_buf_idx, int(U), int(T)
    

    def submit_cpu_gather(self, token_ids_cpu: Tensor, dst_buf_idx: int) -> Future:
        return self._cpu_executor.submit(
            self._cpu_gather_work, token_ids_cpu, dst_buf_idx
        )
     
    @torch._dynamo.disable()   
    def enqueue_h2d_from_stage(self, buf_idx: int, U: int, T: int) -> None:
        """
        Main thread: enqueue H2D copies on prefetch stream, then record event[buf_idx].
        """
        dst_embed = self._gpu_embed_buf(buf_idx)
        stage = self._cpu_stage(buf_idx)

        dst_inv = self._gpu_inv_buf(buf_idx)
        inv_stage = self._cpu_inv_stage(buf_idx)

        ev = self._prefetch_events[buf_idx]

        with torch.cuda.stream(self._prefetch_stream):
            dst_embed[:, :U, :].copy_(stage[:, :U, :], non_blocking=True)
            dst_inv[:T].copy_(inv_stage[:T], non_blocking=True)
            ev.record(self._prefetch_stream)

        self._buf_U[buf_idx] = U
        self._buf_T[buf_idx] = T
    
    @torch._dynamo.disable()   
    def wait_current_ready_nonblocking(self) -> None:
        """
        Insert a GPU dependency: compute stream waits for event[current_buf].
        Does NOT block CPU.
        """
        torch.cuda.current_stream().wait_event(self._prefetch_events[self._cur_buf_idx])

    def current_buffer_ids(self) -> Tensor:
        """
        Returns a view of GPU inv buffer for current chunk: shape [T].
        """
        buf = self._cur_buf_idx
        T = self._buf_T[buf]
        return self._gpu_inv_buf(buf)[:T]

    def swap_to(self, new_buf_idx: int) -> None:
        self._cur_buf_idx = new_buf_idx

    
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
        buffer_ids = buffer_ids.view(idx.shape)
        all_buffers = self._gpu_embed_buf(self._cur_buf_idx)
        
        for i, layer in enumerate(self.layers):
            slot = self._layer_to_stem_slot[i]
            if slot >= 0:
                buffer = all_buffers[slot]
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
        buffer_ids = buffer_ids.view(idx.shape)
        all_buffers = self._gpu_embed_buf(self._cur_buf_idx)

        for i, layer in enumerate(self.layers):
            slot = self._layer_to_stem_slot[i]
            if slot >= 0:
                buffer = all_buffers[slot]
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
    
    
    