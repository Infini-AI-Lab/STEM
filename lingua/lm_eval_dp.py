"""Data-parallel wrapper for lm_eval's evaluate/simple_evaluate.

Adds ``data_parallel_rank`` / ``data_parallel_world_size`` parameters that
control which slice of evaluation data a process works on, **decoupled** from
lm_eval's internal distributed ops (gather_object, barrier, padding sync).

With these parameters the caller sets ``lm._rank = 0, lm._world_size = 1`` so
that lm_eval never triggers any ``torch.distributed`` collectives, while the
data-parallel rank/world_size controls doc sharding via ``build_all_requests``
and ``doc_iterator``.  Each STEM model-parallel group can therefore run
evaluation fully independently on its own shard.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import lm_eval.api.metrics
import lm_eval.api.model
import lm_eval.api.registry
import lm_eval.api.task
from lm_eval.caching.cache import delete_cache
from lm_eval.defaults import DEFAULT_OTHER_SEED, DEFAULT_RANDOM_SEED
from lm_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import (
    handle_non_serializable,
    hash_dict_images,
    hash_string,
    positional_deprecated,
    set_torch_seed,
    setup_logging,
    simple_parse_args_string,
    wrap_text,
)


if TYPE_CHECKING:
    from lm_eval.api.model import LM
    from lm_eval.api.task import Task
    from lm_eval.loggers import EvaluationTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patched ``evaluate`` — the ONLY changes vs. upstream are marked with
# ``# DP-PATCH`` comments.  Diff is ~10 lines.
# ---------------------------------------------------------------------------

def evaluate(
    lm: "LM",
    task_dict,
    limit: int | None = None,
    samples: dict | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: int | None = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: str | None = None,
    apply_chat_template: bool | str = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    confirm_run_unsafe_code: bool = False,
    # DP-PATCH: extra parameters for data-parallel sharding
    data_parallel_rank: int | None = None,
    data_parallel_world_size: int | None = None,
):
    """``lm_eval.evaluator.evaluate`` with data-parallel sharding support.

    When *data_parallel_rank* / *data_parallel_world_size* are given they
    control which documents are assigned to this process (via
    ``build_all_requests`` and ``doc_iterator``).  ``lm.rank`` /
    ``lm.world_size`` are still used for distributed-op gating (padding sync,
    gather_object, barrier) — the caller should set those to ``0`` / ``1`` so
    that no cross-process communication is attempted.
    """

    if limit is not None and samples is not None:
        raise ValueError(
            "Either 'limit' or 'samples' must be None, but both are not None."
        )
    if samples is not None:
        logger.info(f"Evaluating examples for tasks {list(samples.keys())}")
    if apply_chat_template:
        logger.warning(
            "Chat template formatting change affects loglikelihood and multiple-choice tasks. See docs/chat-template-readme.md for details."
        )

    # DP-PATCH: resolve sharding rank/world_size -------------------------
    shard_rank = data_parallel_rank if data_parallel_rank is not None else lm.rank
    shard_world_size = (
        data_parallel_world_size
        if data_parallel_world_size is not None
        else lm.world_size
    )
    # --------------------------------------------------------------------

    requests = defaultdict(list)
    padding_requests = defaultdict(int)

    eval_tasks = get_task_list(task_dict)
    if not log_samples and not all(
        "bypass" not in getattr(task_output.task, "_metric_fn_list", {})
        for task_output in eval_tasks
    ):
        raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    incompatible_tasks = []
    for task_output in eval_tasks:
        task: "Task" = task_output.task
        if getattr(task, "MULTIMODAL", False) and not getattr(lm, "MULTIMODAL", False):
            incompatible_tasks.append(task_output.task_name)
        elif getattr(task, "UNSAFE_CODE", False) and not confirm_run_unsafe_code:
            raise ValueError(
                f"Attempted to run task: {task_output.task_name} which is marked as unsafe. "
                "Set confirm_run_unsafe_code=True to run this task."
            )
    if len(incompatible_tasks) > 0 and not getattr(lm, "MULTIMODAL", False):
        raise ValueError(
            f"Attempted to run tasks: {incompatible_tasks} which require multimodal input, "
            "but the selected model type does not currently implement this. "
            "Multimodal support is currently restricted to the "
            "['hf-multimodal', 'vllm-vlm'] model type."
        )

    limit_arg = limit
    limits = []
    for task_output in eval_tasks:
        task = task_output.task

        limit = get_sample_size(task, limit_arg)
        limits.append(limit)
        task.build_all_requests(
            limit=limit,
            samples=samples.get(task_output.task_name, None)
            if samples is not None
            else samples,
            rank=shard_rank,                    # DP-PATCH (was lm.rank)
            world_size=shard_world_size,         # DP-PATCH (was lm.world_size)
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=bool(apply_chat_template),
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template", None)
            if apply_chat_template
            else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "")
            if apply_chat_template
            else "",
        )
        logger.debug(
            f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
        )
        if write_out:
            print_writeout(task)
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        # lm.world_size-gated padding sync — skipped when lm.world_size == 1
        if lm.world_size > 1:
            import torch

            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )
            reqtype = (
                "loglikelihood"
                if task.OUTPUT_TYPE == "multiple_choice"
                else task.OUTPUT_TYPE
            )
            numpad = max(gathered_item) - gathered_item[lm.rank]
            padding_requests[reqtype] += numpad

    # --- Run LM on inputs ---
    for reqtype, reqs in requests.items():
        logger.info(f"Running {reqtype} requests")
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        resps = getattr(lm, reqtype)(cloned_reqs)

        for x, req in zip(resps, cloned_reqs, strict=True):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    # --- Post-process outputs ---
    RANK = lm.rank
    WORLD_SIZE = lm.world_size

    for task_output, limit in zip(eval_tasks, limits, strict=True):
        task = task_output.task
        task.apply_filters()

        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)

        for filter_key in task.instances[0].filtered_resps:
            indices = (
                samples.get(task_output.task_name, None)
                if samples is not None
                else None
            )
            doc_iterator = task.doc_iterator(
                rank=shard_rank,                 # DP-PATCH (was RANK)
                limit=limit,
                world_size=shard_world_size,      # DP-PATCH (was WORLD_SIZE)
                samples=indices,
            )
            for doc_id, doc in doc_iterator:
                doc_id_true = indices[doc_id] if indices else doc_id
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id_true,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                        "filter": filter_key,
                        "metrics": list(metrics.keys()),
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": hash_string(requests[0].arguments[0]),
                        "target_hash": hash_string(str(target)),
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)

    # lm.world_size-gated gather — skipped when lm.world_size == 1
    if WORLD_SIZE > 1:
        import torch

        for task_output in eval_tasks:
            if log_samples:
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.logged_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )
                if RANK == 0:
                    task_output.logged_samples = list(
                        itertools.chain.from_iterable(full_samples)
                    )

            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(
                        itertools.chain.from_iterable(metric_list)
                    )

    # DP-PATCH: always aggregate (RANK == 0 when lm.world_size == 1)
    if RANK == 0:
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, task_dict
            )

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)

        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if len(task_list) != 0:
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better:
                            _higher_is_better[m] = h
                        if (
                            m in _higher_is_better
                            and _higher_is_better[m] is not None
                            and _higher_is_better[m] != h
                        ):
                            logger.warning(
                                f"Higher_is_better values for metric {m} in group "
                                f"{group} are not consistent. Defaulting to None."
                            )
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **(
                {"groups": dict(group_agg.items())}
                if (bool(group_agg) & show_group_table)
                else {}
            ),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output, limit in zip(eval_tasks, limits, strict=True)
            },
        }
        if log_samples:
            samples = (
                hash_dict_images(samples)
                if os.environ.get("LMEVAL_HASHMM", "1") != "0"
                and (hasattr(lm, "MULTIMODAL"))
                else samples
            )
            results_dict["samples"] = dict(samples)

        return results_dict

    else:
        return None


# ---------------------------------------------------------------------------
# Patched ``simple_evaluate`` — identical to upstream except it accepts and
# forwards the two DP parameters.
# ---------------------------------------------------------------------------

def simple_evaluate(
    model,
    model_args=None,
    tasks=None,
    num_fewshot=None,
    batch_size=None,
    max_batch_size=None,
    device=None,
    use_cache=None,
    cache_requests=False,
    rewrite_requests_cache=False,
    delete_requests_cache=False,
    limit=None,
    samples=None,
    bootstrap_iters=100000,
    check_integrity=False,
    write_out=False,
    log_samples=True,
    evaluation_tracker=None,
    system_instruction=None,
    apply_chat_template=False,
    fewshot_as_multiturn=True,
    gen_kwargs=None,
    task_manager=None,
    verbosity=None,
    predict_only=False,
    random_seed=DEFAULT_RANDOM_SEED,
    numpy_random_seed=DEFAULT_OTHER_SEED,
    torch_random_seed=DEFAULT_OTHER_SEED,
    fewshot_random_seed=DEFAULT_OTHER_SEED,
    confirm_run_unsafe_code=False,
    metadata=None,
    # DP-PATCH: extra parameters
    data_parallel_rank: int | None = None,
    data_parallel_world_size: int | None = None,
):
    """``lm_eval.evaluator.simple_evaluate`` with data-parallel support.

    When *data_parallel_rank* / *data_parallel_world_size* are set, evaluation
    data is sharded across processes while ``lm.rank`` / ``lm.world_size``
    remain ``0`` / ``1`` so that no cross-process torch.distributed
    collectives are triggered by lm_eval.
    """
    if verbosity is not None:
        setup_logging(verbosity=verbosity)
    start_date = time.time()

    if limit is not None and samples is not None:
        raise ValueError(
            "Either 'limit' or 'samples' must be None, but both are not None."
        )

    _NEEDS_CHAT_TEMPLATE = ("inst", "chat")
    if (
        (
            isinstance(model_args, str)
            and any(kw in model_args.lower() for kw in _NEEDS_CHAT_TEMPLATE)
        )
        or (
            isinstance(model_args, dict)
            and any(
                any(kw in str(v).lower() for kw in _NEEDS_CHAT_TEMPLATE)
                for v in model_args.values()
            )
        )
    ) and not apply_chat_template:
        logger.warning(
            wrap_text(
                f"""pretrained={model_args.get("pretrained") if isinstance(model_args, dict) else model_args} appears to be an
                instruct or chat variant but chat template is not applied.
                Recommend setting `apply_chat_template` (optionally `fewshot_as_multiturn`).""",
            )
        )

    if delete_requests_cache:
        logger.info("Deleting requests cache...")
        delete_cache()

    seed_message = []
    if random_seed is not None:
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)
    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)
    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        set_torch_seed(torch_random_seed)
    if fewshot_random_seed is not None:
        seed_message.append(f"Setting fewshot manual seed to {fewshot_random_seed}")
    if seed_message:
        logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found. Please verify the task names."
        )

    if gen_kwargs:
        if isinstance(gen_kwargs, str):
            gen_kwargs = simple_parse_args_string(gen_kwargs)
        logger.warning(
            f"generation_kwargs: {gen_kwargs} specified through cli, these settings "
            "will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if not gen_kwargs:
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            logger.warning("model_args not specified. Using defaults.")
            model_args = ""
        if isinstance(model_args, dict):
            logger.info(
                f"Initializing {model} model, with arguments: {model_args}"
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
        else:
            logger.info(
                wrap_text(
                    f"Initializing {model} model, with arguments: "
                    f"{simple_parse_args_string(model_args)}"
                )
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
    else:
        if not isinstance(model, lm_eval.api.model.LM):
            raise TypeError(
                f"The value of `model` passed to simple_evaluate() was of type "
                f"{type(model)}, but is required to be a subclass of "
                f"lm_eval.api.model.LM."
            )
        logger.info("Using pre-initialized model")
        lm = model

    if use_cache is not None:
        logger.info(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache + "_rank" + str(lm.rank) + ".db",
        )

    if task_manager is None:
        metadata = (
            simple_parse_args_string(model_args)
            if isinstance(model_args, str)
            else model_args
            if isinstance(model_args, dict)
            else {}
        ) | (metadata or {})
        task_manager = TaskManager(metadata=metadata)

    task_dict = get_task_dict(tasks, task_manager)

    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }
            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )
                    logger.info(
                        f"{task_obj.config.task}: Using gen_kwargs: "
                        f"{task_obj.config.generation_kwargs}"
                    )
                if predict_only:
                    logger.info(
                        f"Processing {task_name} in output-only mode. "
                        "Metrics will not be calculated!"
                    )
                    task_obj.override_metric(metric_name="bypass")
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its "
                            "config. Manual configuration will be ignored."
                        )
                    else:
                        logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from "
                            f"{default_num_fewshot} to {num_fewshot}"
                        )
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                adjusted_task_dict[task_name] = task_obj
        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model if isinstance(model, str) else "CUSTOM",
            model_args=model_args or "",
            system_instruction=system_instruction,
            chat_template=getattr(lm, "apply_chat_template", None)
            if apply_chat_template
            else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    # DP-PATCH: forward dp parameters to our patched evaluate
    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        samples=samples,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        verbosity=verbosity,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        data_parallel_rank=data_parallel_rank,
        data_parallel_world_size=data_parallel_world_size,
    )
    if verbosity is not None:
        setup_logging(verbosity=verbosity)

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        if hasattr(lm, "get_model_info"):
            results["config"].update(lm.get_model_info())
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (
                    list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []
                ),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = start_date
        add_env_info(results)
        add_tokenizer_info(results, lm)
        return results
    else:
        return None
