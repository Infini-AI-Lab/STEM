#!/usr/bin/env python3
"""
Run lm-eval directly with a HuggingFace model (no lingua wrapper).
"""

import argparse
import json
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Direct HF lm-eval runner")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="HF model directory or model id",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional local dataset path passed to task config",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="arc_challenge",
        help="lm-eval task name (default: arc_challenge)",
    )
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    parser.add_argument("--bos-token", type=str, default=None)
    parser.add_argument("--eos-token", type=str, default=None)
    parser.add_argument("--unk-token", type=str, default=None)
    parser.add_argument(
        "--register-sp-special-tokens",
        action="store_true",
        help="Register <unk>, <s>, </s> if tokenizer does not define them",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/eval_mob350m/hf_simple_results.json",
        help="Where to save full lm-eval results JSON",
    )
    return parser.parse_args()


def maybe_register_special_tokens(tok):
    special_tokens = {}
    if tok.unk_token is None:
        special_tokens["unk_token"] = "<unk>"
    if tok.bos_token is None:
        special_tokens["bos_token"] = "<s>"
    if tok.eos_token is None:
        special_tokens["eos_token"] = "</s>"
    if special_tokens:
        tok.add_special_tokens(special_tokens)
    return special_tokens


def main():
    args = parse_args()

    model_config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    # transformers>=5 may normalize to {"rope_type": ...} while some custom
    # model code still expects {"type": ...}.
    if isinstance(getattr(model_config, "rope_scaling", None), dict):
        rope_scaling = model_config.rope_scaling
        if rope_scaling.get("rope_type") == "default" and "factor" not in rope_scaling:
            model_config.rope_scaling = None
            rope_scaling = None
        if rope_scaling is None:
            pass
        elif "factor" not in rope_scaling:
            rope_scaling["factor"] = 1.0
        if rope_scaling is not None and "type" not in rope_scaling and "rope_type" in rope_scaling:
            rope_scaling["type"] = rope_scaling["rope_type"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=args.use_fast_tokenizer,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        unk_token=args.unk_token,
    )

    added = {}
    if args.register_sp_special_tokens:
        added = maybe_register_special_tokens(tokenizer)

    model = HFLM(
        pretrained=args.model_path,
        tokenizer=tokenizer,
        config=model_config,
        trust_remote_code=True,
        device=args.device,
        batch_size=args.batch_size,
    )

    task = {"task": args.task}
    if args.dataset_path:
        task["dataset_path"] = args.dataset_path

    results = simple_evaluate(
        model=model,
        tasks=[task],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f)

    metric = results["results"][args.task]
    print(f"task={args.task}")
    print(f"acc={metric.get('acc,none')}")
    print(f"acc_norm={metric.get('acc_norm,none')}")
    if added:
        print(f"registered_special_tokens={added}")


if __name__ == "__main__":
    main()
