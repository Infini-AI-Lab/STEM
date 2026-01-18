from transformers import LlamaConfig
import torch
import flashinfer

class AttnServer:
    
    def __init__(self,
        config: LlamaConfig,
        batch_size: int = 1,
        max_length: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16) -> None:
        
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.get("head_dim", config.hidden_size // config.num_attention_heads)
        
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        
        self.max_num_pages = self.batch_size
        self.page_size = self.max_length
        self.kv_page_indices = torch.arange(self.max_num_pages).int().to(self.device)
        self.kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                                self.workspace_buffer, "HND", use_tensor_cores=True
                            )
        
        self.kv_cache = [
            torch.empty(
                self.max_num_pages, 
                2,
                self.num_key_value_heads,
                self.page_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device=self.device
            )
            for _ in range(self.num_layers)
        ]
        
    def fill(
        self,
        layer_idx: int,
        request_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len: int,
    ):
        self.kv_cache[layer_idx][request_id][0].copy_(key_cache.transpose(0, 1))
        self.kv_cache[layer_idx][request_id][1].copy_(value_cache.transpose(0, 1))
        self.kv_last_page_len[request_id] = seq_len
        
    def plan(self):
        self.kv_last_page_len += 1
        self.decode_wrapper.plan(
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            data_type=torch.bfloat16,
        )
        
    def decode(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int):
        
        key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
        
        flashinfer.append_paged_kv_cache(
            key_states,
            value_states,
            self.kv_page_indptr,
            self.kv_cache[layer_idx],
            self.kv_page_indices,
            self.kv_page_indptr,
            self.kv_last_page_len,
            kv_layout="HND"
        )
        
        q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
        hidden_states = self.dense_decode_wrapper.run(
        q, 
        self.flashinfer_kv_cache[layer_idx]
        )
        hidden_states = hidden_states.reshape(self.batch_size, 1, self.hidden_size)
        
        return hidden_states
        
        