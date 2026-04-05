#!/usr/bin/env python3
"""load_dataset(local_path, ...); len(data) for each folder under test_eval.

Layout matches setup/download_eval_datasets.sh (hellaswag/, super_glue/, …).

Usage:
  python setup/print_test_eval_hub_lengths.py
  python setup/print_test_eval_hub_lengths.py --root /other/path/test_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

DEFAULT_ROOT = "/data-fsx/beidchen-sandbox/data/eval_data"

# (label, subdir under root, config or None, split)
SINGLE = [
    ("hellaswag", "hellaswag", None, "train"),
    ("boolq", "super_glue", "boolq", "validation"),
    ("piqa", "piqa", None, "validation"),
    ("winogrande", "winogrande", "winogrande_xl", "validation"),
    ("openbookqa", "openbookqa", "main", "test"),
    ("arc_easy", "ai2_arc", "ARC-Easy", "train"),
    ("arc_challenge", "ai2_arc", "ARC-Challenge", "train"),
    ("race", "race", "high", "test"),
    ("gsm8k", "gsm8k", "main", "test"),
    ("mbpp", "mbpp", None, "test"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(DEFAULT_ROOT),
        help=f"test_eval root (default: {DEFAULT_ROOT})",
    )
    args = p.parse_args()
    root = args.root.expanduser().resolve()

    for label, sub, name, split in SINGLE:
        path = str(root / sub)
        if name is None:
            data = load_dataset(path, split=split)
        else:
            data = load_dataset(path, name, split=split)
        print(
            f"{label}: len={len(data):,}  load_dataset({path!r}"
            + (f", {name!r}" if name else "")
            + f", split={split!r})"
        )


if __name__ == "__main__":
    main()
