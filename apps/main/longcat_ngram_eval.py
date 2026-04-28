# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# lm-eval for Longcat n-gram checkpoints. Same harness layout as
# :mod:`apps.main.eval`, plus ``stem_parallel_size`` for vocabulary-parallel
# n-gram tables (STEM-style process groups; see :mod:`apps.main.stem_eval`).
#
# ``model_type`` is the Longcat backbone only: ``llama``, ``qwen3``, or ``olmo3``.
#
# CLI: ``torchrun ... -m apps.main.longcat_eval config=... model_type=olmo3 ckpt_dir=...``

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import wandb
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from omegaconf import OmegaConf

from apps.main import stem_eval as stem_eval_mod
from apps.main.eval import EvalArgs, _truncate_at_stop
from apps.main.eval_utils import (
    apply_humaneval_runtime_patches,
    apply_mbpp_runtime_patches,
    harness_has_humaneval_task,
    harness_has_mbpp_task,
)
from apps.main.longcat_ngram_generate import (
    LongcatPackedCausalTransformerGenerator,
    load_longcat_consolidated_model_and_tokenizer,
    _get_longcat_registry,
)
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import (
    DistributedArgs,
    get_global_rank,
    setup_torch_distributed,
)
from lingua.lm_eval_dp import simple_evaluate as dp_simple_evaluate
from lingua.stem_dist_utils import (
    get_stem_data_parallel_group,
    get_stem_data_parallel_rank,
    get_stem_data_parallel_world_size,
    initialize_stem_process_group,
    is_stem_initialized,
)
from lingua.tokenizer import build_tokenizer

logger = logging.getLogger(__name__)

EVAL_FOLDER_NAME = stem_eval_mod.EVAL_FOLDER_NAME


@dataclass
class LongcatEvalArgs(EvalArgs):
    """Eval config for Longcat; adds vocab / n-gram parallel width."""

    stem_parallel_size: int = 1


_LONGCAT_EVAL_FIELD_NAMES = {f.name for f in fields(LongcatEvalArgs)}


def _longcat_eval_yaml_to_structured(file_cfg: OmegaConf) -> OmegaConf:
    """Drop keys not on :class:`LongcatEvalArgs` so shared YAMLs can carry extras."""
    plain = OmegaConf.to_container(file_cfg, resolve=True)
    if not isinstance(plain, dict):
        return OmegaConf.create({})
    filtered = {k: v for k, v in plain.items() if k in _LONGCAT_EVAL_FIELD_NAMES}
    return OmegaConf.create(filtered)


class _LongcatEvalHarnessLM(LM):
    """Same idea as :class:`apps.main.stem_eval.EvalHarnessLM`: lm_eval sees rank 0 / world 1."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = stem_eval_mod.StemMockAccelerator()
        self._rank = 0
        self._world_size = 1
        self.device = generator.device

    def generate_until(self, requests):
        prompts, gen_args = zip(*[req.args for req in requests])
        from collections import OrderedDict

        groups = OrderedDict()
        for i, (prompt, ga) in enumerate(zip(prompts, gen_args)):
            key = tuple(
                sorted(
                    ((k, tuple(v) if isinstance(v, list) else v) for k, v in ga.items()),
                    key=lambda x: x[0],
                )
            )
            if key not in groups:
                groups[key] = (ga, [])
            groups[key][1].append((i, prompt))

        results = [None] * len(requests)
        for key, (ga, indexed_prompts) in groups.items():
            temperature = ga.get("temperature", 0.0)
            top_p = ga.get("top_p", None)
            top_k = ga.get("top_k", None)
            until = ga.get("until", [])
            if isinstance(until, str):
                until = [until]

            self.generator.temperature = temperature
            self.generator.top_p = top_p
            self.generator.top_k = top_k
            self.generator.until = until

            group_prompts = [p for _, p in indexed_prompts]
            generations, _, _ = self.generator.generate(group_prompts)
            for (orig_idx, _), g in zip(indexed_prompts, generations):
                g = _truncate_at_stop(g, until)
                results[orig_idx] = g

        return results

    def loglikelihood(self, requests):
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate(inputs)
        results = []
        for p, ll, gr in zip(prompts, lls, greedy):
            p_len = len(
                self.generator.tokenizer.encode(p, add_bos=False, add_eos=False)
            )
            results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))

        self.generator.max_gen_len = max_gen_len
        return results

    def loglikelihood_rolling(self, requests):
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = []
        for ll in lls:
            results.append((ll.sum().item(),))
        self.generator.max_gen_len = max_gen_len

        return results


def launch_longcat_eval(cfg: LongcatEvalArgs) -> None:
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())

    if harness_has_mbpp_task(cfg.harness):
        apply_mbpp_runtime_patches(logger=logger)
    if harness_has_humaneval_task(cfg.harness):
        apply_humaneval_runtime_patches(logger=logger)

    if (
        torch.distributed.is_initialized()
        and int(cfg.stem_parallel_size) > 1
        and not is_stem_initialized()
    ):
        initialize_stem_process_group(int(cfg.stem_parallel_size))
        logger.info(
            "Initialized stem process groups for Longcat eval "
            f"(stem_parallel_size={cfg.stem_parallel_size})"
        )

    _has_dp_peers = (
        is_stem_initialized() and get_stem_data_parallel_world_size() > 1
    )
    if is_stem_initialized():
        dp_rank = get_stem_data_parallel_rank()
        dp_ws = get_stem_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_ws = 1

    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / "params.json").exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    consolidate_path = str(consolidate_path)
    torch.distributed.barrier()

    backbone = cfg.model_type
    reg = _get_longcat_registry()
    if backbone not in reg:
        raise ValueError(
            f"Unknown Longcat backbone model_type '{backbone}'. "
            f"Available: {list(reg.keys())}"
        )
    logger.info(f"Loading Longcat model (backbone={backbone})")
    model, tokenizer, train_cfg = load_longcat_consolidated_model_and_tokenizer(
        consolidate_path,
        model_type=backbone,
    )
    if cfg.tokenizer_path is not None:
        tokenizer_name = cfg.tokenizer_name or train_cfg.data.tokenizer.name
        logger.info(
            "Overriding tokenizer from CLI "
            f"(name={tokenizer_name}, path={cfg.tokenizer_path})"
        )
        tokenizer = build_tokenizer(tokenizer_name, cfg.tokenizer_path)
    logger.info("Model loaded")
    model.eval()
    generator = LongcatPackedCausalTransformerGenerator(
        cfg.generator, model, tokenizer
    )

    wrap = _LongcatEvalHarnessLM(generator)
    harness_kwargs = asdict(cfg.harness)
    if dp_ws > 1:
        logger.info(
            "Running lm_eval with data-parallel sharding: "
            f"dp_rank={dp_rank}, dp_ws={dp_ws}"
        )
        results = dp_simple_evaluate(
            wrap,
            **harness_kwargs,
            data_parallel_rank=dp_rank,
            data_parallel_world_size=dp_ws,
        )
    else:
        results = dp_simple_evaluate(wrap, **harness_kwargs)

    if _has_dp_peers and results is not None:
        safe_gather_keys = [
            "results",
            "versions",
            "n-shot",
            "higher_is_better",
            "n-samples",
            "git_hash",
            "date",
            "pretty_env_info",
            "transformers_version",
            "lm_eval_version",
        ]
        results_to_gather = {
            k: v for k, v in results.items() if k in safe_gather_keys
        }
        results_to_gather = json.loads(
            json.dumps(results_to_gather, default=lambda o: None)
        )
        dp_group = get_stem_data_parallel_group()
        gathered = [None] * dp_ws
        torch.distributed.all_gather_object(
            gathered, results_to_gather, group=dp_group
        )
        results = stem_eval_mod._merge_shard_results(gathered)
        logger.info(f"Merged harness results from {dp_ws} DP shards")

    val_results: Optional[dict] = None
    if cfg.validation:
        val_results = stem_eval_mod.eval_on_val(
            generator, cfg.validation, train_cfg
        )

    rank = get_global_rank() if torch.distributed.is_initialized() else 0
    if rank == 0 and results is not None:
        with open(Path(cfg.dump_dir) / "results.json", "w") as f:
            safe_keys = [
                "results",
                "versions",
                "n-shot",
                "higher_is_better",
                "n-samples",
                "git_hash",
                "date",
                "pretty_env_info",
                "transformers_version",
                "lm_eval_version",
            ]
            serializable_results = {
                k: v for k, v in results.items() if k in safe_keys
            }
            f.write(json.dumps(serializable_results))
        logger.info(f"All evaluation results: {results['results']}")
        if val_results is not None:
            with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
                f.write(json.dumps(val_results))
            logger.info(f"All validation results: {val_results}")
    if cfg.metric_log_dir and rank == 0 and results is not None:
        metric_log_path = Path(cfg.metric_log_dir) / "metrics.eval.jsonl"
        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {"created_at": datetime.utcnow().isoformat()}
        if cfg.global_step is not None:
            timestamp["global_step"] = cfg.global_step
        print(
            json.dumps(timestamp | results["results"]),
            file=open(metric_log_path, mode="a"),
            flush=True,
        )
        val_log_path = Path(cfg.metric_log_dir) / "metrics.validation.jsonl"
        if val_results is not None:
            print(
                json.dumps(timestamp | val_results),
                file=open(val_log_path, mode="a"),
                flush=True,
            )

    if rank == 0:
        _wandb_initialized_here = False
        if wandb.run is None and cfg.wandb is not None:
            wandb_kwargs = (
                cfg.wandb if isinstance(cfg.wandb, dict) else asdict(cfg.wandb)
            )
            wandb.init(**wandb_kwargs)
            _wandb_initialized_here = True

        if wandb.run is not None:
            wandb_metrics = {}
            if results is not None and "results" in results:
                for task_name, task_metrics in results["results"].items():
                    if isinstance(task_metrics, dict):
                        for metric_name, value in task_metrics.items():
                            if isinstance(value, (int, float)):
                                wandb_metrics[f"evals/{task_name}/{metric_name}"] = value
            if val_results is not None:
                for src_name, src_metrics in val_results.items():
                    if isinstance(src_metrics, dict):
                        for metric_name, value in src_metrics.items():
                            if isinstance(value, (int, float)):
                                wandb_metrics[
                                    f"validation/{src_name}/{metric_name}"
                                ] = value
            if wandb_metrics:
                wandb.log(wandb_metrics, step=cfg.global_step)
                logger.info(
                    f"Logged {len(wandb_metrics)} eval metrics to wandb at step {cfg.global_step}"
                )

            if _wandb_initialized_here:
                wandb.finish()

    del generator
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def main() -> None:
    cli_args = OmegaConf.from_cli()
    file_cfg = _longcat_eval_yaml_to_structured(OmegaConf.load(cli_args.config))
    del cli_args.config

    default_cfg = OmegaConf.structured(LongcatEvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_longcat_eval(cfg)


if __name__ == "__main__":
    main()
