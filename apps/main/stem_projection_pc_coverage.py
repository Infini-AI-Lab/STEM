"""PC warmup with optional pack-time skip of all-common sequences.

Enable in YAML, for example:

  data:
    pack_skip_common_threshold: 1000
    pack_skip_max_skips: 10000

``pack_skip_vocab_size`` is filled from the tokenizer in validate_train_args.
"""

from apps.main.stem_projection_pc_warmup_nohooks import main

if __name__ == "__main__":
    main()
