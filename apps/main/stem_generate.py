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

from apps.main.stem import StemLMTransformer, StemLMTransformerArgs, STEM_MODEL_REGISTRY
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME, consolidate_checkpoints
from lingua.stem_checkpoint import (
    CONSOLIDATE_STEM_NAME,
    consolidate_stem_shards,
    load_stem_shards_resharded,
)
from lingua.stem_dist_utils import ParallelEmbedding, is_stem_initialized
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
    model_cls=None,
    model_args_cls=None,
    tokenizer_name: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    model_overrides: Optional[dict] = None,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    # Resolve model class from config's model_type when not explicitly provided
    if model_cls is None or model_args_cls is None:
        model_type = getattr(config, "model_type", "llama")
        if model_type not in STEM_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{model_type}' in checkpoint config. "
                f"Available: {list(STEM_MODEL_REGISTRY.keys())}"
            )
        reg_cls, reg_args_cls = STEM_MODEL_REGISTRY[model_type][:2]
        model_cls = model_cls or reg_cls
        model_args_cls = model_args_cls or reg_args_cls

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_cfg = OmegaConf.merge(config.model, OmegaConf.create(model_overrides)) if model_overrides else config.model
    model_args = dataclass_from_dict(model_args_cls, model_cfg, strict=False)
    tok_name = config.data.tokenizer.name
    tok_path = config.data.tokenizer.path
    if tokenizer_path:
        tok_path = tokenizer_path
    if tokenizer_name:
        tok_name = tokenizer_name
    tokenizer = build_tokenizer(tok_name, tok_path)
    model = model_cls(model_args)
    
    backbone_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)
    if "model" in backbone_dict:
        backbone_dict = backbone_dict["model"]
    if next(iter(backbone_dict.keys())).startswith("model"):
        backbone_dict = {k.replace("model.", "lm_transformer."): v for k, v in backbone_dict.items()}
    # Relax strict loading for keys that are either (a) loaded from a separate
    # consolidation (stem_embeddings, loaded below) or (b) structurally
    # reconstructed at model construction from params.json (DAG `alpha` scalar,
    # which some DCP save paths drop for 1-element params; `alpha_init` from
    # params.json fully determines its value).
    missing_keys, unexpected_keys = model.load_state_dict(backbone_dict, strict=False)

    stem_layer_indices = list(model.lm_transformer.stem_layers)
    expected_stem_emb_missing = {
        f"stem_embeddings.{i}.weight" for i in range(len(stem_layer_indices))
    }
    expected_alpha_missing = {
        f"lm_transformer.layers.{i}.feed_forward.alpha" for i in stem_layer_indices
    }
    missing_set = set(missing_keys)
    unaccounted_missing = missing_set - expected_stem_emb_missing - expected_alpha_missing
    assert not unaccounted_missing, f"Missing keys: {sorted(unaccounted_missing)}"
    # Every stem_embedding key must be missing here (loaded from stem shards below).
    assert expected_stem_emb_missing.issubset(missing_set), (
        f"Unexpected stem_embeddings layout; got missing={sorted(missing_set)}"
    )
    alpha_absent = expected_alpha_missing & missing_set
    if alpha_absent:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "DAG `alpha` parameters absent from backbone checkpoint; using "
            "`alpha_init` from params.json (module default init). Affected: "
            f"{sorted(alpha_absent)}"
        )
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    
    if is_stem_initialized():
        # Distributed: load sharded stem weights for this STEM MP rank
        # Use the parent dir (pre-consolidation checkpoint dir) which contains stem_shards/
        ckpt_parent = Path(os.path.dirname(ckpt_path))
        load_stem_shards_resharded(model, ckpt_parent)
    else:
        # Non-distributed: load consolidated (full) stem weights.
        #
        # We call ``consolidate_stem_shards`` unconditionally (rather than
        # gating it on the existence of ``consolidated_stem.pth``) so that a
        # stale, column-scrambled consolidation produced by a pre-fix
        # version of this function is detected via the missing sorted-order
        # marker and rebuilt in rank-sorted order.  When the marker is
        # already present, the call short-circuits without re-reading the
        # shards, so the hot path is effectively free.
        consolidate_stem_shards(os.path.dirname(ckpt_path))

        stem_dict = torch.load(ckpt_path / CONSOLIDATE_STEM_NAME, weights_only=True)
        with torch.no_grad():
            for module_name, module in model.named_modules():
                if isinstance(module, (nn.Embedding, ParallelEmbedding)):
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

    consolidate_path = consolidate_checkpoints(cfg.ckpt)
    consolidate_path = str(consolidate_path)
    
    model, tokenizer, _ = load_consolidated_model_and_tokenizer(consolidate_path)

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