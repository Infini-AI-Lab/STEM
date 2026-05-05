"""Lightweight tests for knowledge-editing prompt selection.

These tests intentionally avoid loading a model or checkpoint. They validate
that every zero-shot prompt type builds exactly one query block and that the
numbered CLI aliases normalize to the expected canonical prompt types.
"""

from __future__ import annotations

import sys
import types
import unittest


try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault(
        "torch",
        types.SimpleNamespace(no_grad=lambda: (lambda fn: fn)),
    )

from apps.main.knowledge_editing.experiment import (  # noqa: E402
    build_prompt_by_type,
    get_prompt_default_entities,
    normalize_prompt_type,
)


ZERO_SHOT_CASES = {
    "country-capital-zero-shot": ("Country:", 1),
    "math-text-zero-shot": ("Problem:", 1),
    "math-unary-op-zero-shot": ("Operation:", 1),
    "math-binary-op-zero-shot": ("Operation:", 1),
    "math-derivative-zero-shot": ("Function:", 1),
    "math-prime-composite-zero-shot": ("Number:", 1),
    "math-area-zero-shot": ("Shape:", 1),
    "coding-sort-reverse-zero-shot": ("Action:", 1),
    "coding-builtin-call-zero-shot": ("Operation:", 1),
    "coding-array-constructor-zero-shot": ("Array type:", 1),
    "coding-pandas-method-zero-shot": ("Request:", 1),
    "coding-sql-aggregate-zero-shot": ("Aggregate:", 1),
}


ZERO_SHOT_ALIASES = {
    "math-prompt-1-zero-shot": "math-unary-op-zero-shot",
    "math-prompt-2-zero-shot": "math-binary-op-zero-shot",
    "math-prompt-3-zero-shot": "math-derivative-zero-shot",
    "math-prompt-4-zero-shot": "math-prime-composite-zero-shot",
    "math-prompt-5-zero-shot": "math-area-zero-shot",
    "coding-prompt-1-zero-shot": "coding-sort-reverse-zero-shot",
    "coding-prompt-2-zero-shot": "coding-builtin-call-zero-shot",
    "coding-prompt-3-zero-shot": "coding-array-constructor-zero-shot",
    "coding-prompt-4-zero-shot": "coding-pandas-method-zero-shot",
    "coding-prompt-5-zero-shot": "coding-sql-aggregate-zero-shot",
}


class ZeroShotPromptRegistryTests(unittest.TestCase):
    def test_zero_shot_prompts_have_single_query_block(self) -> None:
        for prompt_type, (marker, expected_count) in ZERO_SHOT_CASES.items():
            with self.subTest(prompt_type=prompt_type):
                source, _ = get_prompt_default_entities(prompt_type)
                prompt = build_prompt_by_type(prompt_type, source)
                self.assertEqual(prompt.count(marker), expected_count)
                self.assertIn(source, prompt)

    def test_numbered_zero_shot_aliases_normalize(self) -> None:
        for alias, canonical in ZERO_SHOT_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(normalize_prompt_type(alias), canonical)

    def test_numbered_zero_shot_aliases_build(self) -> None:
        for alias in ZERO_SHOT_ALIASES:
            with self.subTest(alias=alias):
                source, _ = get_prompt_default_entities(alias)
                prompt = build_prompt_by_type(alias, source)
                self.assertIn(source, prompt)
                self.assertGreater(len(prompt), len(source))


if __name__ == "__main__":
    unittest.main()
