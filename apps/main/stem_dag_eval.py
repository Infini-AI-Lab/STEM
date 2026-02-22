# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Evaluation script for DAG-STEM models.

Registers the DAG model types into ``STEM_MODEL_REGISTRY`` so that
``launch_stem_eval`` can resolve ``llama_dag`` / ``qwen3_dag`` /
``olmo3_dag`` from the eval config.

Usage::

    python -m apps.main.stem_dag_eval config=apps/main/configs/stem_dag_eval.yaml
"""

# Register DAG model types before importing the eval entrypoint.
from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_dag import DAG_STEM_MODEL_REGISTRY

STEM_MODEL_REGISTRY.update(DAG_STEM_MODEL_REGISTRY)

from apps.main.stem_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
