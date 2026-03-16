from dataclasses import dataclass, field

from omegaconf import OmegaConf

from apps.main.stem import STEM_MODEL_REGISTRY
from apps.main.stem_train import StemTrainArgs, train as base_stem_train
from apps.main.stem_compressed import (
    CompressedStemLMTransformerArgs,
    COMPRESSED_STEM_MODEL_REGISTRY,
    clear_default_lookup_table_for_all,
    set_default_lookup_table_for_all,
)
from lingua.tokenizer import CompressedTokenizer, build_tokenizer


@dataclass
class CompressedStemTrainArgs(StemTrainArgs):
    model: CompressedStemLMTransformerArgs = field(
        default_factory=CompressedStemLMTransformerArgs
    )


def train(args: CompressedStemTrainArgs):
    if not args.model_type.endswith("_compressed"):
        raise ValueError(
            "Compressed STEM training requires model_type ending with "
            "'_compressed' (e.g. llama_compressed)"
        )

    tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
    compressed_tokenizer = CompressedTokenizer(tokenizer)

    if args.model.stem_vocab_size is None:
        args.model.stem_vocab_size = len(compressed_tokenizer)

    set_default_lookup_table_for_all(compressed_tokenizer.lookup_table)

    original_registry = dict(STEM_MODEL_REGISTRY)
    STEM_MODEL_REGISTRY.update(COMPRESSED_STEM_MODEL_REGISTRY)
    try:
        base_stem_train(args)
    finally:
        STEM_MODEL_REGISTRY.clear()
        STEM_MODEL_REGISTRY.update(original_registry)
        clear_default_lookup_table_for_all()


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(CompressedStemTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
