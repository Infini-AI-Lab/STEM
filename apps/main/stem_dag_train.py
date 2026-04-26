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
"""

from dataclasses import dataclass, field

from omegaconf import ListConfig, OmegaConf

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


def _load_config_arg(config_arg):
    """Load one config path or a list of config paths, merged left to right."""
    if isinstance(config_arg, ListConfig):
        paths = list(config_arg)
    elif isinstance(config_arg, (list, tuple)):
        paths = list(config_arg)
    else:
        paths = [config_arg]
    if not paths:
        raise ValueError("config must contain at least one path")
    return OmegaConf.merge(*[OmegaConf.load(str(path)) for path in paths])


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = _load_config_arg(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(DagStemTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
