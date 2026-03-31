# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Generation script for sliding-window STEM models.

Registers window model types into ``STEM_MODEL_REGISTRY`` so that
``load_consolidated_model_and_tokenizer`` can resolve
``llama_window`` / ``qwen3_window`` / ``olmo3_window`` from ``params.json``.
"""

from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_window import WINDOW_STEM_MODEL_REGISTRY

STEM_MODEL_REGISTRY.update(WINDOW_STEM_MODEL_REGISTRY)

from apps.main.stem_generate import main  # noqa: E402

if __name__ == "__main__":
    main()
