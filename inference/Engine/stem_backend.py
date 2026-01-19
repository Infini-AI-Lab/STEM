import flashinfer
import torch
import torch.distributed as dist
from concurrent.futures import Future
from typing import Optional
from Engine.stem_model import StemTransformer
from Engine.backend import LMBackend
from Engine.utils import load_stem_model

class StemLMBackend(LMBackend):
    def __init__(self,
        dtype=torch.bfloat16,
        device: str = "cuda:0",
        dec_len: int = 1,
    ) -> None:
        super().__init__(dtype, device, dec_len)
        self.model_forward = lambda model, x, buffer_ids, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model(
            x,
            buffer_ids,
            input_pos,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        self.prefill = lambda model, x, buffer_ids, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model.prefill(
            x,
            buffer_ids,
            input_pos,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        
    def load_model(
        self,
        checkpoints: str,
        model_name: str,
        max_batched_tokens: int,
        use_tp: bool=False,
        rank_group=None,
        group=None,
    ):
        self.model: StemTransformer = load_stem_model(
            checkpoint_path=checkpoints,
            model_name=model_name,
            device=self.device,
            precision=self.dtype,
            max_batched_tokens=max_batched_tokens,
            use_tp=use_tp,
            rank_group=rank_group,
            group=group,
        ) 
            
    @torch.inference_mode()
    def encode(
        self, 
        input_ids: torch.LongTensor, 
        cpu_input_ids: torch.LongTensor, 
        buffer_ids: torch.LongTensor
    ):
        self.clear_kv()
        logits = None
        seq_len = input_ids.shape[1]
        chunk_size = 128
        num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceil division
        
        # Track prefetch future for next chunk
        prefetch_future: Optional[Future] = None
        next_buf = 1 - self.model._cur_buf_idx
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            next_start_idx = (i + 1) * chunk_size
            next_end_idx = min(next_start_idx + chunk_size, seq_len)
            
            # Immediately submit CPU gather for chunk i+1 (non-blocking)
            if i < num_chunks - 1:
                next_chunk_input_ids = cpu_input_ids[:, next_start_idx:next_end_idx]
                prefetch_future = self.model.submit_cpu_gather(next_chunk_input_ids, next_buf)
            
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            dec_len = end_idx - start_idx
            
            self.pre_encode(dec_len=dec_len)
            
            # compute stream waits for current buffer to be ready
            self.model.wait_current_ready_nonblocking()
            
            # Launch GPU compute for chunk i
            logits = self.prefill(
                model=self.model,
                x=chunk_input_ids,
                buffer_ids=buffer_ids,
                input_pos=self.cachelens,
                kv_append_indptr=self.qo_indptr * dec_len,
                kv_page_indices=self.paged_kv_indices,
                kv_page_indptr=self.paged_kv_indptr,
                kv_page_lastlen=self.paged_kv_last_page_len,
            )
            self.cachelens += dec_len
            
            if i < num_chunks - 1:
                assert prefetch_future is not None
                # Block on CPU gather while GPU is computing
                next_buf, next_U, next_T = prefetch_future.result()
                # Enqueue H2D from CPU gather result to GPU buffer
                self.model.enqueue_h2d_from_stage(next_buf, next_U, next_T)
                self.model.swap_to(next_buf)
                buffer_ids = self.model.current_buffer_ids()
                next_buf = 1 - next_buf
                
        return self.sample(logits)
            