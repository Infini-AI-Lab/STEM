# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Packed text generation for Longcat n-gram LMs. Same behavior as
# :mod:`apps.main.generate` but prefilling passes ``ngram_segment_ids`` so
# n-gram embeddings do not mix across packed prompts. Run:
#   python -m apps.main.longcat_generate ckpt=... model_type=olmo3
#
# ``model_type`` defaults from ``params.json`` if omitted (must be one of
# ``longcat`` / ``qwen3`` / ``olmo3``).

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union

import torch
from omegaconf import OmegaConf
from torch import nn

from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
)
from apps.main.longcat import (
    LongcatLMTransformer,
    LongcatLMTransformerArgs,
    LongcatOLMo3LMTransformer,
    LongcatOLMo3LMTransformerArgs,
    LongcatQwen3LMTransformer,
    LongcatQwen3LMTransformerArgs,
)
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.longcat_checkpoint import (
    NGRAM_SHARD_SUBDIR,
    load_ngram_shards_resharded_for_inference,
)
from lingua.stem_dist_utils import ParallelEmbedding, VocabParallelEmbedding
from lingua.tokenizer import build_tokenizer
from lingua.longcat_ngram import segment_ids_from_packed_lengths

# Matches ``LONGCAT_MODEL_REGISTRY`` / ``longcat_ngram_train`` (backbone name -> Longcat-wrapped class).
LongcatModelRegistry = Dict[str, Tuple[Type[nn.Module], Type]]


def _get_longcat_registry() -> LongcatModelRegistry:
    return {
        "llama": (LongcatLMTransformer, LongcatLMTransformerArgs),
        "qwen3": (LongcatQwen3LMTransformer, LongcatQwen3LMTransformerArgs),
        "olmo3": (LongcatOLMo3LMTransformer, LongcatOLMo3LMTransformerArgs),
    }


def _vocab_parallel_state_keys(model: nn.Module) -> set[str]:
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (ParallelEmbedding, VocabParallelEmbedding)):
            for param_name, _ in module.named_parameters(recurse=False):
                keys.add(f"{module_name}.{param_name}" if module_name else param_name)
    return keys


class LongcatPackedCausalTransformerGenerator(PackedCausalTransformerGenerator):
    """Packs prompts like :class:`PackedCausalTransformerGenerator` but prefill provides ``ngram_segment_ids`` for Longcat n-gram boundaries."""

    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor):
        self.setup_prefilling(lengths=lengths)
        ngram_kwargs: dict = {}
        ngram_kwargs["ngram_segment_ids"] = segment_ids_from_packed_lengths(
            lengths, device=tokens.device, dtype=torch.long
        ).unsqueeze(0)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            mask=self.prefill_mask,
            attn_impl="flex_attention",
            **ngram_kwargs,
        )
        self.setup_generation(lengths=lengths)
        return prefill_out


def load_longcat_consolidated_model_and_tokenizer(
    consolidated_path: Union[str, Path],
    model_type: Optional[str] = None,
    model_cls: Optional[Type[nn.Module]] = None,
    model_args_cls: Optional[Type] = None,
):
    """
    Load a Longcat n-gram checkpoint.

    Longcat training saves the FSDP-managed backbone under ``consolidated/`` and
    saves vocabulary-parallel n-gram tables separately under
    ``../ngram_shards/``.  The backbone load is therefore strict except for
    those n-gram table keys, which are loaded from the side shards afterward.
    If ``model_type`` is None, it is read from the checkpoint config (as in
    training), defaulting to ``"llama"`` if missing.
    """
    ckpt_path = Path(consolidated_path)
    config = OmegaConf.load(ckpt_path / "params.json")
    reg = _get_longcat_registry()

    if model_cls is None or model_args_cls is None:
        resolved_type = model_type
        if resolved_type is None:
            resolved_type = str(getattr(config, "model_type", "llama") or "llama")
        if resolved_type not in reg:
            raise ValueError(
                f"Unknown Longcat model_type '{resolved_type}'. Available: {list(reg.keys())}"
            )
        model_cls = model_cls or reg[resolved_type][0]
        model_args_cls = model_args_cls or reg[resolved_type][1]

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)  # type: ignore[misc, arg-type]
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = model_cls(model_args)  # type: ignore[operator, misc]
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True, map_location="cpu")
    if "model" in st_dict:
        st_dict = st_dict["model"]

    first_key = next(iter(st_dict.keys()))
    if first_key.startswith("model."):
        st_dict = {k.replace("model.", "", 1): v for k, v in st_dict.items()}

    model_state_keys = model.state_dict().keys()
    for output_key, embedding_key in (
        (
            "lm_transformer.output.tied_module.weight",
            "lm_transformer.tok_embeddings.weight",
        ),
        ("output.tied_module.weight", "tok_embeddings.weight"),
    ):
        if (
            output_key in model_state_keys
            and output_key not in st_dict
            and embedding_key in st_dict
        ):
            st_dict[output_key] = st_dict[embedding_key]

    ngram_keys = _vocab_parallel_state_keys(model)
    missing_keys, unexpected_keys = model.load_state_dict(st_dict, strict=False)
    missing = set(missing_keys)
    unexpected = set(unexpected_keys)
    if missing - ngram_keys:
        raise RuntimeError(
            "Missing non-ngram keys while loading Longcat backbone: "
            f"{sorted(missing - ngram_keys)}"
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys while loading Longcat backbone: {sorted(unexpected)}"
        )

    step_ckpt_path = ckpt_path.parent
    if not load_ngram_shards_resharded_for_inference(
        model, step_ckpt_path, map_location="cpu"
    ):
        raise FileNotFoundError(
            f"Longcat n-gram shards not found under {step_ckpt_path / NGRAM_SHARD_SUBDIR}"
        )
        
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, tokenizer, config


def main() -> None:
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs, cfg, strict=False
    )
    print(cfg)

    model, tokenizer, _ = load_longcat_consolidated_model_and_tokenizer(
        cfg.ckpt,
        model_type=getattr(cfg, "model_type", None),
    )

    generator = LongcatPackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    prompts = []
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)

    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    tokens_per_second = total_tokens / (end_time - start_time)

    for i, gen in enumerate(generation):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated Text: {gen}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
