# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Training script for DAG-STEM models.

Uses the same training loop as ``stem_train.py`` but registers the
DAG model types (``llama_dag``, ``qwen3_dag``, ``olmo3_dag``) and
uses ``STEMDagLMTransformerArgs`` so that the ``alpha_init`` hyper-parameter
is parsed from the config.

Usage::

    torchrun --nproc-per-node=4 -m apps.main.stem_dag_train \\
        config=apps/main/configs/stem_dag_llama3_1B.yaml

Optional: set ``freeze_stem_up_proj: true`` in the config (or
``freeze_stem_up_proj=true`` on the CLI) to keep DAG-STEM layer w3
(up-projection) fixed while training stem embeddings and the rest of
the backbone (subject to ``train_stage``).
"""

from dataclasses import dataclass, field

from omegaconf import OmegaConf

# Register DAG model types into the shared STEM_MODEL_REGISTRY *before*
# importing the training function (which resolves model_type from the registry).
from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_dag import DAG_STEM_MODEL_REGISTRY, STEMDagLMTransformerArgs

STEM_MODEL_REGISTRY.update(DAG_STEM_MODEL_REGISTRY)

from apps.main.stem_train import StemTrainArgs, train  # noqa: E402


@dataclass
class DagStemTrainArgs(StemTrainArgs):
    """Training args for DAG-STEM.  Inherits stem_lr / stem_weight_decay
    from ``StemTrainArgs`` and overrides the model field so that OmegaConf
    can parse ``alpha_init`` from the config."""
    model: STEMDagLMTransformerArgs = field(default_factory=STEMDagLMTransformerArgs)
    # Freeze w3 (up-projection) in DAG-STEM stem layers after checkpoint load.
    freeze_stem_up_proj: bool = False


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(DagStemTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
