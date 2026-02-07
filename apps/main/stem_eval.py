# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import Any, List, Optional, Tuple, Union
from lm_eval import simple_evaluate
from omegaconf import OmegaConf
import torch
import wandb
from apps.main.stem_generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.main.stem import StemLMTransformer, StemLMTransformerArgs
from apps.main.eval import LMHarnessArgs, ValidationArgs, all_dicts_same
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.data import init_choice_state, setup_sources
from lingua.distributed import (
    DistributedArgs,
    dist_mean_dict,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)
from lingua.stem_dist_utils import (
    initialize_stem_process_group,
    is_stem_initialized,
    get_stem_data_parallel_group,
    get_stem_data_parallel_rank,
    get_stem_data_parallel_world_size,
    get_stem_model_parallel_group,
)

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()

@dataclass
class StemEvalArgs:
    name: str = "stem_evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    stem_parallel_size: int = 1  # STEM model parallel size (>1 enables distributed ParallelEmbedding)
    generator: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    harness: Optional[LMHarnessArgs] = field(default_factory=LMHarnessArgs)
    validation: Optional[ValidationArgs] = field(default_factory=ValidationArgs)

    wandb: Optional[Any] = None

    global_step: Optional[int] = None  # for in-training evaluation


class StemMockAccelerator:
    def gather(self, tensor):
        if is_stem_initialized():
            # Only one MP group is running eval (world_size=1 for lm_eval).
            # gather is not expected to be called, but handle safely.
            return tensor.unsqueeze(0)
        elif torch.distributed.is_initialized():
            out = [torch.zeros_like(tensor) for _ in range(get_world_size())]
            torch.distributed.all_gather(out, tensor)
            return torch.stack(out)
        else:
            return tensor.unsqueeze(0)

    def wait_for_everyone(self):
        if is_stem_initialized():
            # Barrier only within the active MP group to avoid hanging
            # (dp_rank > 0 ranks are not participating in eval).
            torch.distributed.barrier(group=get_stem_model_parallel_group())
        elif torch.distributed.is_initialized():
            torch.distributed.barrier()


# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = StemMockAccelerator()
        if is_stem_initialized():
            self._rank = 0
            self._world_size = 1
        elif torch.distributed.is_initialized():
            self._rank = get_global_rank()
            self._world_size = get_world_size()
        else:
            self._rank = 0
            self._world_size = 1
        self.device = generator.device

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        temperature = gen_args.get("temperature", 0.0)
        top_p = gen_args.get("top_p", None)
        top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.top_k = top_k
        self.generator.until = until
        generations, _, _ = self.generator.generate(prompts)
        filtered_gen = []
        for g in generations:
            for e in until:
                g = g.replace(e, "")
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate(inputs)
        results = []
        for p, ll, gr in zip(prompts, lls, greedy):
            p_len = len(self.generator.tokenizer.encode(p, add_bos=False, add_eos=False))
            results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))

        self.generator.max_gen_len = max_gen_len
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = []
        for ll in lls:
            results.append((ll.sum().item(),))
        self.generator.max_gen_len = max_gen_len

        return results
    

def eval_on_val(generator, val_args: ValidationArgs, train_cfg):
    srcs = {}
    for src in val_args.sources:
        path = os.path.join(val_args.root_dir, src)
        srcs[path] = 1.0
    if hasattr(train_cfg, 'data') and hasattr(train_cfg.data, 'sources'):
        for src in train_cfg.data.sources:
            path = os.path.join(train_cfg.data.root_dir, src)
            srcs[path] = 1.0

    # Determine effective DP rank/degree for validation data splitting.
    if is_stem_initialized():
        dp_rank = 0
        dp_degree = 1
    elif torch.distributed.is_initialized():
        dp_rank = get_global_rank()
        dp_degree = get_world_size()
    else:
        dp_rank = 0
        dp_degree = 1
    
    multi_state = init_choice_state("", srcs, 0, dp_rank, dp_degree, "*.val.jsonl")
    path_to_iter = setup_sources(multi_state)

    max_gen_len = generator.max_gen_len
    # We temporarily lower max gen len
    generator.max_gen_len = 1

    all_val_metrics = {}
    for src in path_to_iter:
        jsonl_iterator = path_to_iter[src]
        texts = []
        logger.info(f"Running validation on {src}...")
        for step, (content, state) in enumerate(jsonl_iterator):
            if state['current_iter'] > 0 or (val_args.max_steps is not None and step >= val_args.max_steps):
                break
            content_key = "text" if ("text" in content) else "content"
            texts.append(content[content_key])
        
        _, loglikelihood, _ = generator.generate(texts)

        metrics = defaultdict(list)
        for i, ll in enumerate(loglikelihood):
            tmp = ll.sum().item()
            metrics['nll'].append(tmp)
            metrics['nll_per_token'].append(tmp / len(ll))
            metrics['nll_per_char'].append(tmp / len(texts[i]))

            metrics['avg_seqlen'].append(len(ll))
        
        for m in metrics:
            metrics[m] = sum(metrics[m]) / len(metrics[m])

        if dp_degree > 1 and torch.distributed.is_initialized():
            metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done. Metrics: {metrics}")

        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(f"Duplicate source name {name}, path {src} in validation sources, renaming to {name}_1")
            name = f"{name}_1"
        all_val_metrics[name] = metrics

    generator.max_gen_len = max_gen_len

    return all_val_metrics


def launch_stem_eval(cfg: StemEvalArgs):
    # Setup distributed if needed (but don't require it)
    if torch.distributed.is_initialized():
        pass  # Already initialized
    elif torch.cuda.device_count() > 1:
        setup_torch_distributed(DistributedArgs())
    
    # Initialize STEM process groups for distributed ParallelEmbedding
    # Skip if already initialized (e.g. when called from stem_train.py during training)
    if torch.distributed.is_initialized() and cfg.stem_parallel_size > 1 and not is_stem_initialized():
        initialize_stem_process_group(cfg.stem_parallel_size)
        logger.info(f"Initialized STEM process groups with parallel size: {cfg.stem_parallel_size}")
    
    _has_dp_peers = (
        is_stem_initialized() and get_stem_data_parallel_world_size() > 1
    )
    _eval_participates = True
    if _has_dp_peers and get_stem_data_parallel_rank() != 0:
        _eval_participates = False

    # Check if checkpoint is already consolidated
    ckpt_path = Path(cfg.ckpt_dir)
    if (
        ckpt_path.exists()
        and (ckpt_path / "params.json").exists()
        and next(ckpt_path.glob("*.pth"), None) is not None
    ):
        consolidate_path = ckpt_path
    else:
        consolidate_path = ckpt_path / CONSOLIDATE_FOLDER
        if not consolidate_path.exists():
            rank = get_global_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    consolidate_path = str(consolidate_path)
    
    # All ranks wait for checkpoint consolidation
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Non-participating ranks: wait for eval-participating ranks to finish
    if not _eval_participates:
        logger.info("dp_rank != 0 — skipping eval, waiting at barrier")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return
    
    logger.info("Loading STEM model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=StemLMTransformer,
        model_args_cls=StemLMTransformerArgs,
    )
    logger.info("STEM model loaded")
    model.eval()
    generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)

    wrap = EvalHarnessLM(generator)
    results = simple_evaluate(wrap, **asdict(cfg.harness))
    
    val_results = None
    if cfg.validation:
        val_results = eval_on_val(generator, cfg.validation, train_cfg)
    
    rank = get_global_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        with open(Path(cfg.dump_dir) / "results.json", "w") as f:
            # Filter out non-serializable keys (configs contains function objects)
            safe_keys = ['results', 'versions', 'n-shot', 'higher_is_better', 
                        'n-samples', 'git_hash', 'date', 'pretty_env_info',
                        'transformers_version', 'lm_eval_version']
            serializable_results = {k: v for k, v in results.items() if k in safe_keys}
            f.write(json.dumps(serializable_results))
        logger.info(f"All evaluation results: {results['results']}")
        if val_results is not None:
            with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
                f.write(json.dumps(val_results))
            logger.info(f"All validation results: {val_results}")
    
    if cfg.metric_log_dir and rank == 0:
        metric_log_path = Path(cfg.metric_log_dir) / "metrics.eval.jsonl"

        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
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
            
        # Log eval results to wandb (rank 0 only)
    if rank == 0:
        _wandb_initialized_here = False
        # Async eval: cfg.wandb is set but no active run yet → init one
        if wandb.run is None and cfg.wandb is not None:
            wandb_kwargs = cfg.wandb if isinstance(cfg.wandb, dict) else asdict(cfg.wandb)
            wandb.init(**wandb_kwargs)
            _wandb_initialized_here = True

        # Sync eval: wandb.run is already active from MetricLogger in the training loop
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
                                wandb_metrics[f"validation/{src_name}/{metric_name}"] = value
            if wandb_metrics:
                wandb.log(wandb_metrics, step=cfg.global_step)
                logger.info(f"Logged {len(wandb_metrics)} eval metrics to wandb at step {cfg.global_step}")

            # Only finish the run if we initialized it here (async eval).
            # In sync eval, the training loop's MetricLogger owns the run.
            if _wandb_initialized_here:
                wandb.finish()
    
    del generator

    # Sync with non-participating ranks (they are waiting at a matching barrier)
    if _has_dp_peers and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: StemLMTransformerArgs

    @dataclass
    class StemLMTransformerArgs:
        dim: int

    Then you can pass model.dim=32 to change values in StemLMTransformerArgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate StemEvalArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call stem_eval.py with stem_eval.py model.dim=64

    Then the final StemEvalArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in StemEvalArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(StemEvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_stem_eval(cfg)


if __name__ == "__main__":
    main()

