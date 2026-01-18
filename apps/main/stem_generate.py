# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional

import torch
from torch import nn
from tqdm import tqdm

from omegaconf import OmegaConf
from torch.nn import functional as F
import xformers

from apps.main.transformer import LMTransformer, LMTransformerArgs
from apps.main.stem import StemLMTransformer, StemLMTransformerArgs
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.stem_checkpoint import CONSOLIDATE_STEM_NAME, consolidate_stem_shards
from lingua.stem_dist_utils import ParallelEmbedding
from lingua.tokenizer import Tokenizer, build_tokenizer
from lingua.transformer import (
    Attention,
    causal_mask,
    generate_doc_mask_mod,
    lengths_to_local_ids,
    lengths_to_start_ids,
)
from torch.nn.attention.flex_attention import create_block_mask

from apps.main.generate import (
    sample_top_p, 
    sample_top_k, 
    sample_tokens, 
    pack_prompts, 
    batch_prompts, 
    KVCache,
    PackedCausalTransformerGeneratorArgs,
    PackedCausalTransformerGenerator,
)

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
    model.load_state_dict(backbone_dict["model"], strict=False)
    
    if not (ckpt_path / CONSOLIDATE_STEM_NAME).exists():
        consolidate_stem_shards(os.path.dirname(ckpt_path))
    
    stem_dict = torch.load(ckpt_path / CONSOLIDATE_STEM_NAME, weights_only=True)
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                weight_key = f"{module_name}.weight" if module_name else "weight"
                if weight_key in stem_dict:
                    module.weight.copy_(stem_dict[weight_key].to(module.weight.device))
    
    # Move model to GPU and set dtype
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    
    return model, tokenizer, config


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs, cfg, strict=False
    )
    print(cfg)

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(cfg.ckpt)

    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Allow multiple prompts
    prompts = []
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)

    # Start generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    # Calculate tokens per second
    total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    tokens_per_second = total_tokens / (end_time - start_time)

    # Display the results
    for i, gen in enumerate(generation):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated Text: {gen}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()