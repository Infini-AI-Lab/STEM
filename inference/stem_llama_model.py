import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class PLELlamaMLP(nn.Module):
    """Modified Llama MLP with PLE (Per-Layer Embedding) support."""
    
    def __init__(self, config: LlamaConfig, use_ple: bool = False, ple_buffer_size: int = 4096):
        super().__init__()
        self.config = config
        self.use_ple = use_ple
        
        # Standard MLP components
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        # PLE components (only if use_ple is True)
        if use_ple:
            self.ple_buffer = nn.Embedding(ple_buffer_size, config.intermediate_size)
        else:
            pass
        
    def forward(self, x: torch.Tensor, buffer_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_ple and buffer_ids is not None:
            gate = F.silu(self.gate_proj(x))
            ple = self.ple_buffer(buffer_ids)
            y = self.down_proj(gate * ple)
        else:
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            y = self.down_proj(gate * up)
        return y
    

class PLELlamaDecoderLayerBlock(nn.Module):
    """Modified Llama decoder layer with PLE support."""
    
    def __init__(self, config: LlamaConfig, start_layer_idx: int, end_layer_idx: int, ple_buffer_size: int = 2048):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_local_layers = end_layer_idx - start_layer_idx
        
        self_attn_blocks = []
        mlp_blocks = []
        inp_ln_blocks = []
        post_attn_ln_blocks = []
        for layer_idx in range(start_layer_idx, end_layer_idx):    
            # Use standard attention implementation
            try:
                from transformers.models.llama.modeling_llama import LlamaAttention
                self_attn_blocks.append(LlamaAttention(config, layer_idx))
            except (ImportError, AttributeError):
                # Fallback: try to get from config
                if hasattr(config, '_attn_implementation'):
                    self_attn_blocks.append(config._attn_implementation(config, layer_idx))
                else:
                    raise ImportError("Could not import LlamaAttention")
                
            mlp_blocks.append(PLELlamaMLP(config, use_ple=(layer_idx > 0), ple_buffer_size=ple_buffer_size))
            
            inp_ln_blocks.append(nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps))
            post_attn_ln_blocks.append(nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps))

        self.self_attn_blocks = nn.ModuleList(self_attn_blocks)
        self.mlp_blocks = nn.ModuleList(mlp_blocks)
        self.inp_ln_blocks = nn.ModuleList(inp_ln_blocks)
        self.post_attn_ln_blocks = nn.ModuleList(post_attn_ln_blocks)

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        buffer_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = hidden_states

        for local_i in range(self.num_local_layers):
            # --- Attention block: x = x + Attn(LN(x)) ---
            residual = x
            xn = self.inp_ln_blocks[local_i](x)
            layer_past = past_key_values[local_i] if past_key_values is not None else None
            attn_out, self_attn_weights, present_key_value = self.self_attn_blocks[local_i](
                hidden_states=xn,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            x = residual + attn_out
            
            # --- MLP block: x = x + MLP(LN(x)) ---
            residual = x
            xn = self.post_attn_ln_blocks[local_i](x)
            mlp_out = self.mlp_blocks[local_i](xn, buffer_ids=buffer_ids)
            x = residual + mlp_out
        
        return x
    

class PLELlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig, ple_buffer_size: int = 4096):
        # Initialize parent
        super().__init__(config)
        
        blocks = []
        for layer_idx in range(0, config.num_hidden_layers, config.num_local_layers):
            end_layer_idx = min(layer_idx + config.num_local_layers, config.num_hidden_layers)
            blocks.append(PLELlamaDecoderLayerBlock(config, layer_idx, end_layer_idx, ple_buffer_size=ple_buffer_size))
        self.blocks = nn.ModuleList(blocks)
        
        self.num_ple_layers = config.num_hidden_layers - 1  # First layer doesn't use PLE
        self.ple_buffer_size = ple_buffer_size
        