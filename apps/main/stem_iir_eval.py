# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Evaluation script for IIR-STEM models.

Registers the IIR model types into ``STEM_MODEL_REGISTRY`` so that
``launch_stem_eval`` can resolve ``llama_iir`` / ``qwen3_iir`` /
``olmo3_iir`` from the eval config.

Usage::

    python -m apps.main.stem_iir_eval config=apps/main/configs/stem_eval.yaml
"""

# Register IIR model types before importing the eval entrypoint.
from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_iir import IIR_STEM_MODEL_REGISTRY

STEM_MODEL_REGISTRY.update(IIR_STEM_MODEL_REGISTRY)

from apps.main.stem_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
