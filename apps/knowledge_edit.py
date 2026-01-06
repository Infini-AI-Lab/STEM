import argparse
import gc
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import List, Optional, Union, Callable, Dict

import torch

from apps.main.stem import LMTransformer as StemTransformer
from apps.main.stem import StemLMTransformer, StemLMTransformerArgs
from apps.main.transformer import create_causal_mask

from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import BlockMask
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, AutoConfig, AutoTokenizer

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.generation.utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)





class StemTransformerKnowledgeEdit(StemTransformer):
    def forward(
        self, 
        token_values: torch.Tensor,
        ple_token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
        stem_embeddings_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)
        
        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)
        
        