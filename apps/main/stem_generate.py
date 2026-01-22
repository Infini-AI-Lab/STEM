from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Union

import torch
from torch import nn
from tqdm import tqdm

from omegaconf import OmegaConf
from torch.nn import functional as F

from apps.main.stem import StemLMTransformer, StemLMTransformerArgs
from lingua.args import dataclass_from_dict
from lingua.tokenizer import Tokenizer, build_tokenizer
from lingua.checkpoint import CONSOLIDATE_NAME
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict as dcp_get_model_state_dict

from transformers.utils import logging

logger = logging.get_logger(__name__)



def load_consolidated_model_and_tokenizer(
    consolidated_path,
    model_cls=StemLMTransformer,
    model_args_cls=StemLMTransformerArgs,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)
    
    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = model_cls(model_args)
    
    
    backbone_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)
    model.lm_transformer.load_state_dict(backbone_dict["model"])
    
    stem_dict = torch.load(ckpt_path / CONSOLIDATE_STEM_NAME, weights_only=True)
    for module_name, module in model.named_modules():
        if isinstance(module, ParallelEmbedding):
            for p_name, p in module.named_parameters(recurse=False):
                fq_name = f"{module_name}.{p_name}" if module_name else p_name
                if fq_name in stem_dict:
                    p.copy_(stem_dict[fq_name].to(p.device))
    
    # Move model to GPU and set dtype
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    
    return model, tokenizer, config
    