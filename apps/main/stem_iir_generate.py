# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Generation script for IIR-STEM models.

Registers the IIR model types into ``STEM_MODEL_REGISTRY`` so that
``load_consolidated_model_and_tokenizer`` can resolve ``llama_iir`` /
``qwen3_iir`` / ``olmo3_iir`` from ``params.json``.

Usage::

    python -m apps.main.stem_iir_generate ckpt=<path>
"""

# Register IIR model types before importing the generation entrypoint.
from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_iir import IIR_STEM_MODEL_REGISTRY

STEM_MODEL_REGISTRY.update(IIR_STEM_MODEL_REGISTRY)

from apps.main.stem_generate import main  # noqa: E402

if __name__ == "__main__":
    main()
