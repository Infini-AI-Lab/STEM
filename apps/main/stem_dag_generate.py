# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Generation script for DAG-STEM models.

Registers the DAG model types into ``STEM_MODEL_REGISTRY`` so that
``load_consolidated_model_and_tokenizer`` can resolve ``llama_dag`` /
``qwen3_dag`` / ``olmo3_dag`` from ``params.json``.

Usage::

    python -m apps.main.stem_dag_generate ckpt=<path>
"""

# Register DAG model types before importing the generation entrypoint.
from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_dag import DAG_STEM_MODEL_REGISTRY

STEM_MODEL_REGISTRY.update(DAG_STEM_MODEL_REGISTRY)

from apps.main.stem_generate import main  # noqa: E402

if __name__ == "__main__":
    main()
