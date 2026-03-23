# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
import json
import logging
import os
import wandb
from pathlib import Path
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import Any, List, Optional, Tuple, Union
from lm_eval import simple_evaluate
from omegaconf import OmegaConf
import torch
from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.main.transformer import LMTransformer, LMTransformerArgs
from apps.main.qwen3 import Qwen3LMTransformer, Qwen3LMTransformerArgs
from apps.main.olmo3 import OLMo3LMTransformer, OLMo3LMTransformerArgs
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

EVAL_FOLDER_NAME = "{:010d}"

# Registry mapping model_type strings to (model_cls, model_args_cls)
MODEL_REGISTRY = {
    "llama": (LMTransformer, LMTransformerArgs),
    "qwen3": (Qwen3LMTransformer, Qwen3LMTransformerArgs),
    "olmo3": (OLMo3LMTransformer, OLMo3LMTransformerArgs),
}

logger = logging.getLogger()


def _truncate_at_stop(text: str, stops: List[str]) -> str:
    """Truncate generation at first stop sequence occurrence."""
    stop_positions = [text.find(stop) for stop in stops if stop]
    stop_positions = [pos for pos in stop_positions if pos >= 0]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


@dataclass
class LMHarnessArgs:
    tasks: Optional[List[Any]] = None
    num_fewshot: Optional[int] = None
    device: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: Optional[Union[int, float]] = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: Optional[str] = None
    verbosity: str = "INFO"
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234
    batch_size: Union[int, str] = 8
    confirm_run_unsafe_code: bool = False

@dataclass
class ValidationArgs:
    max_steps: Optional[int] = None # If None the whole validation file is used -> /!\ This number of steps is gpu dependent (100 max steps on 8 gpus = 800 steps on 1 gpu)
    use_val_from_train_src: bool = True # Use the validation set from training sources
    root_dir: str = ""
    sources: List[str] = field(default_factory=list) # Other sources to eval on


@dataclass
class MonkeyPatchArgs:
    enable_timeout_patch: bool = False
    timeout: float = 10.0
    num_workers: int = 1


@dataclass
class EvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    model_type: str = "llama"  # "llama", "qwen3", or "olmo3"
    generator: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    harness: Optional[LMHarnessArgs] = field(default_factory=LMHarnessArgs)
    validation: Optional[ValidationArgs] = field(default_factory=ValidationArgs)
    monkeypatch: MonkeyPatchArgs = field(default_factory=MonkeyPatchArgs)

    wandb: Optional[Any] = None

    global_step: Optional[int] = None  # for in-training evaluation


def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)


class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()


# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])

        # Group requests by gen_args so we can handle different configs
        from collections import OrderedDict
        groups = OrderedDict()  # frozen gen_args -> list of (original_index, prompt)
        for i, (prompt, ga) in enumerate(zip(prompts, gen_args)):
            key = tuple(sorted(
                ((k, tuple(v) if isinstance(v, list) else v) for k, v in ga.items()),
                key=lambda x: x[0],
            ))
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
    for src in train_cfg.data.sources:
        path = os.path.join(train_cfg.data.root_dir, src)
        srcs[path] = 1.0

    multi_state = init_choice_state("", srcs, 0, get_global_rank(), get_world_size(), "*.val.jsonl")
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
        metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done. Metrics: {metrics}")

        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(f"Duplicate source name {name}, path {src} in validation sources, renaming to {name}_1")
            name = f"{name}_1"
        all_val_metrics[name] = metrics

    generator.max_gen_len = max_gen_len

    return all_val_metrics


def apply_lm_eval_monkeypatches(cfg: MonkeyPatchArgs) -> None:
    if not cfg.enable_timeout_patch:
        return

    try:
        from lm_eval.tasks.mbpp import utils as mbpp_utils
    except Exception as e:
        logger.warning(
            "Monkeypatch requested but import failed: %s", e
        )
        return

    original_pass_at_1 = getattr(mbpp_utils, "pass_at_1")

    # Avoid double-patching if this process re-enters launch_eval.
    if not getattr(original_pass_at_1, "_stem_mbpp_timeout_patched", False):

        @wraps(original_pass_at_1)
        def patched_pass_at_1(references, predictions):
            if isinstance(references, str):
                references = [references]
            if predictions and isinstance(predictions[0], str):
                predictions = [[p] for p in predictions]
            return pass_at_k.compute(
                references=references,
                predictions=predictions,
                k=[1],
                timeout=cfg.timeout,
                num_workers=cfg.num_workers,
            )[0]["pass@1"]

        patched_pass_at_1._stem_mbpp_timeout_patched = True
        mbpp_utils.pass_at_1 = patched_pass_at_1
        logger.info(
            "Applied MBPP pass@1 monkeypatch: timeout=%s num_workers=%s",
            cfg.timeout,
            cfg.num_workers,
        )

def launch_eval(cfg: EvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
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

    # Resolve model class from model_type
    if cfg.model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{cfg.model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls, model_args_cls = MODEL_REGISTRY[cfg.model_type]
    logger.info(f"Loading model (type={cfg.model_type})")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=model_cls,
        model_args_cls=model_args_cls,
    )
    logger.info("Model loaded")
    model.eval()
    generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)

    apply_lm_eval_monkeypatches(cfg.monkeypatch)

    wrap = EvalHarnessLM(generator)
    results = simple_evaluate(wrap, **asdict(cfg.harness))
    val_results =  None
    if cfg.validation:
        val_results = eval_on_val(generator, cfg.validation, train_cfg)
    if get_global_rank() == 0:
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
    if cfg.metric_log_dir and get_global_rank() == 0:
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
    if get_global_rank() == 0:
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


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate EvalArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_eval(cfg)


if __name__ == "__main__":
    main()
