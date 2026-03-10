# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Generation script for Selective-State-Space-STEM models.

Registers the Selective-State-Space model types into ``STEM_MODEL_REGISTRY`` so
that ``load_consolidated_model_and_tokenizer`` can resolve
``llama_sss`` / ``llama_selective_iir`` / ``qwen3_sss`` / ``qwen3_selective_iir`` /
``olmo3_sss`` / ``olmo3_selective_iir`` from ``params.json``.

Usage::

    python -m apps.main.stem_sss_generate ckpt=<path>
"""

# Register Selective-State-Space model types before importing the generation entrypoint.
from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_sss import SELECTIVE_IIR_STEM_MODEL_REGISTRY

STEM_MODEL_REGISTRY.update(SELECTIVE_IIR_STEM_MODEL_REGISTRY)

from apps.main.stem_generate import main  # noqa: E402

if __name__ == "__main__":
    main()