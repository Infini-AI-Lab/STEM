from __future__ import annotations

from functools import wraps
import importlib
import logging
import multiprocessing as mp
import os
import uuid
from typing import Any, Optional

import evaluate as hf_evaluate

# lm_eval's humaneval/utils.py and mbpp/utils.py call evaluate.load("code_eval") at import time
# with no experiment_id, so every rank shares one Arrow cache path (NFS races, ENOENT, stale handle).
# Patch once when this module loads (stem_eval / eval import eval_utils before running lm_eval).
_stem_original_evaluate_load = hf_evaluate.load


def harness_has_mbpp_task(harness: Optional[Any]) -> bool:
    if harness is None or getattr(harness, "tasks", None) is None:
        return False
    for task in harness.tasks:
        if isinstance(task, str):
            if task == "mbpp":
                return True
            continue
        if isinstance(task, dict) and task.get("task") == "mbpp":
            return True
    return False


def _is_humaneval_task_name(name: str) -> bool:
    return name == "humaneval" or name.startswith("humaneval_")


def harness_has_humaneval_task(harness: Optional[Any]) -> bool:
    if harness is None or getattr(harness, "tasks", None) is None:
        return False
    for task in harness.tasks:
        if isinstance(task, str):
            if _is_humaneval_task_name(task):
                return True
            continue
        if isinstance(task, dict):
            t = task.get("task")
            if isinstance(t, str) and _is_humaneval_task_name(t):
                return True
    return False


def _unique_experiment_id(prefix: str = "mbpp-code-eval") -> str:
    rank = os.environ.get("RANK", "0")
    return f"{prefix}-{rank}-{os.getpid()}-{uuid.uuid4().hex}"


def _stem_patched_evaluate_load(path: str, *args: Any, **kwargs: Any) -> Any:
    if path == "code_eval" and "experiment_id" not in kwargs:
        kwargs = {**kwargs, "experiment_id": _unique_experiment_id("code-eval-auto")}
    return _stem_original_evaluate_load(path, *args, **kwargs)


hf_evaluate.load = _stem_patched_evaluate_load  # type: ignore[method-assign]


def _patch_metric_check_correctness_to_fork(metric_obj: Any, logger: logging.Logger) -> None:
    metric_module_name = metric_obj.__class__.__module__
    try:
        metric_module = importlib.import_module(metric_module_name)
    except Exception as e:
        logger.warning("Unable to import code_eval module %s: %s", metric_module_name, e)
        return

    original_check_correctness = getattr(metric_module, "check_correctness", None)
    if original_check_correctness is None:
        logger.warning("code_eval module has no check_correctness; skipping fork patch")
        return
    if getattr(original_check_correctness, "_stem_force_fork", False):
        return

    try:
        execute_module = importlib.import_module(original_check_correctness.__module__)
        unsafe_execute = getattr(execute_module, "unsafe_execute")
    except Exception as e:
        logger.warning("Unable to import code_eval execute module: %s", e)
        return

    @wraps(original_check_correctness)
    def fork_check_correctness(check_program, timeout, task_id, completion_id):
        ctx = mp.get_context("fork")
        manager = ctx.Manager()
        result = manager.list()
        p = ctx.Process(target=unsafe_execute, args=(check_program, result, timeout))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()
        if not result:
            result.append("timed out")
        return dict(
            task_id=task_id,
            passed=result[0] == "passed",
            result=result[0],
            completion_id=completion_id,
        )

    fork_check_correctness._stem_force_fork = True
    metric_module.check_correctness = fork_check_correctness
    logger.info("Applied MBPP code_eval fork-context patch")


def apply_mbpp_runtime_patches(logger: logging.Logger) -> None:
    """
    Runtime MBPP patching without modifying lm-eval task files.
    - Uses a per-process metric instance id to avoid lock contention.
    - Forces code_eval workers to run in fork context for lower overhead.
    - Replaces mbpp pass_at_1 to use configurable timeout/num_workers.
    """
    try:
        from lm_eval.tasks.mbpp import utils as mbpp_utils
    except Exception as e:
        logger.warning("Unable to import MBPP utils for runtime patching: %s", e)
        return

    try:
        metric = hf_evaluate.load("code_eval", experiment_id=_unique_experiment_id())
    except Exception as e:
        logger.warning("Unable to load code_eval metric for MBPP runtime patch: %s", e)
        return

    _patch_metric_check_correctness_to_fork(metric, logger)
    mbpp_utils.pass_at_k = metric

    original_pass_at_1 = getattr(mbpp_utils, "pass_at_1", None)
    if original_pass_at_1 is None:
        logger.warning("MBPP utils has no pass_at_1; skipping pass_at_1 patch")
        return
    if getattr(original_pass_at_1, "_stem_runtime_patched", False):
        return

    @wraps(original_pass_at_1)
    def patched_pass_at_1(references, predictions):
        if isinstance(references, str):
            references = [references]
        if predictions and isinstance(predictions[0], str):
            predictions = [[p] for p in predictions]
        timeout = float(os.environ.get("MBPP_CODE_EVAL_TIMEOUT", "3.0"))
        workers = int(os.environ.get("MBPP_CODE_EVAL_NUM_WORKERS", "8"))
        return metric.compute(
            references=references,
            predictions=predictions,
            k=[1],
            timeout=timeout,
            num_workers=workers,
        )[0]["pass@1"]

    patched_pass_at_1._stem_runtime_patched = True
    mbpp_utils.pass_at_1 = patched_pass_at_1
    logger.info("Applied MBPP runtime pass_at_1 patch")


def apply_humaneval_runtime_patches(logger: logging.Logger) -> None:
    """
    Runtime HumanEval patching (lm_eval.tasks.humaneval), mirroring MBPP/code_eval setup.
    See https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/humaneval
    - Per-process code_eval experiment_id to reduce lock contention.
    - Fork-based check_correctness for sandboxed execution.
    - pass_at_k uses HUMANEVAL_CODE_EVAL_TIMEOUT / HUMANEVAL_CODE_EVAL_NUM_WORKERS.
    """
    try:
        from lm_eval.tasks.humaneval import utils as humaneval_utils
    except Exception as e:
        logger.warning("Unable to import HumanEval utils for runtime patching: %s", e)
        return

    try:
        metric = hf_evaluate.load("code_eval", experiment_id=_unique_experiment_id("humaneval-code-eval"))
    except Exception as e:
        logger.warning("Unable to load code_eval metric for HumanEval runtime patch: %s", e)
        return

    _patch_metric_check_correctness_to_fork(metric, logger)
    humaneval_utils.compute_ = metric

    original_pass_at_k = getattr(humaneval_utils, "pass_at_k", None)
    if original_pass_at_k is None:
        logger.warning("HumanEval utils has no pass_at_k; skipping pass_at_k patch")
        return
    if getattr(original_pass_at_k, "_stem_runtime_patched", False):
        return

    @wraps(original_pass_at_k)
    def patched_pass_at_k(references, predictions, k=None):
        assert k is not None
        if isinstance(k, int):
            k = [k]
        timeout = float(os.environ.get("HUMANEVAL_CODE_EVAL_TIMEOUT", "3.0"))
        workers = int(os.environ.get("HUMANEVAL_CODE_EVAL_NUM_WORKERS", "8"))
        res = metric.compute(
            references=references,
            predictions=predictions,
            k=k,
            timeout=timeout,
            num_workers=workers,
        )
        return res[0]

    patched_pass_at_k._stem_runtime_patched = True
    humaneval_utils.pass_at_k = patched_pass_at_k
    logger.info("Applied HumanEval runtime pass_at_k patch")
