#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lm-eval-stem"))
from lm_eval import simple_evaluate
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
start = time.perf_counter()
results = simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.2-1B,trust_remote_code=True",
    tasks=[{"task": "mbpp", "dataset_name": "full"}],
    batch_size="8",
    num_fewshot=3,
    log_samples=False,
    bootstrap_iters=0,
    limit=64,
    confirm_run_unsafe_code=True,
)
elapsed = time.perf_counter() - start
print({"elapsed_sec": round(elapsed, 3)})
print(results["results"].get("mbpp", {}))