"""Lightweight tests for knowledge-editing plot preparation.

These tests avoid importing the full model stack and do not require
matplotlib. They validate the deterministic pieces that protect the production
plotter from malformed inputs and unstable axes.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest


try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault(
        "torch",
        types.SimpleNamespace(
            Tensor=object,
            no_grad=lambda: (lambda fn: fn),
        ),
    )
else:
    if not hasattr(_torch, "Tensor"):
        setattr(_torch, "Tensor", object)

from apps.main.knowledge_editing.experiment import (  # noqa: E402
    TopKResult,
    TopKToken,
    _last_prompt_block,
    _nice_probability_ylim,
    _plot_token_label,
    plot_topk_probabilities,
)


def _result(case: str, probability: float = 0.5) -> TopKResult:
    return TopKResult(
        case=case,
        prompt=f"Shot: example\n\nCountry: {case}\nCapital:",
        prompt_token_ids=[1, 2, 3],
        top_k=1,
        tokens=[
            TopKToken(
                token_id=42,
                token_str=" Madrid",
                logit=1.0,
                probability=probability,
            )
        ],
    )


def _topk_results() -> dict[str, TopKResult]:
    return {
        "original": TopKResult(
            case="original",
            prompt="Country: Spain\nCapital:",
            prompt_token_ids=[1, 2, 3],
            top_k=4,
            tokens=[
                TopKToken(10, " Madrid", 3.0, 0.72),
                TopKToken(11, " the", 0.4, 0.03),
                TopKToken(12, " Barcelona", 0.2, 0.02),
                TopKToken(13, " a", 0.1, 0.015),
            ],
        ),
        "target": TopKResult(
            case="target",
            prompt="Country: Germany\nCapital:",
            prompt_token_ids=[1, 4, 3],
            top_k=4,
            tokens=[
                TopKToken(20, " Berlin", 2.5, 0.60),
                TopKToken(21, " called", 0.6, 0.04),
                TopKToken(22, " the", 0.5, 0.035),
                TopKToken(23, " located", 0.4, 0.025),
            ],
        ),
        "intervened": TopKResult(
            case="intervened",
            prompt="Country: Spain\nCapital:",
            prompt_token_ids=[1, 2, 3],
            top_k=4,
            tokens=[
                TopKToken(20, " Berlin", 2.6, 0.62),
                TopKToken(21, " called", 0.6, 0.04),
                TopKToken(22, " the", 0.5, 0.035),
                TopKToken(23, " located", 0.4, 0.025),
            ],
        ),
    }


class PlotPreparationTests(unittest.TestCase):
    def test_probability_axis_uses_formal_tick_ceiling(self) -> None:
        self.assertEqual(_nice_probability_ylim(0.0), 0.2)
        self.assertEqual(_nice_probability_ylim(0.61), 0.8)
        self.assertEqual(_nice_probability_ylim(0.72), 0.8)
        self.assertEqual(_nice_probability_ylim(0.95), 1.0)

    def test_probability_axis_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            _nice_probability_ylim(float("nan"))
        with self.assertRaises(ValueError):
            _nice_probability_ylim(-0.1)

    def test_tick_labels_trim_tokenizer_spacing(self) -> None:
        self.assertEqual(_plot_token_label(" the", 1), "the")
        self.assertEqual(_plot_token_label("", 1), "<empty:1>")

    def test_caption_uses_final_query_block(self) -> None:
        prompt = "Example: one\nAnswer: two\n\nCountry: Spain\nCapital:"
        self.assertEqual(_last_prompt_block(prompt), "Country: Spain\nCapital:")

    def test_plotter_validates_required_cases_before_matplotlib_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)
            with self.assertRaisesRegex(ValueError, "Missing top-k results"):
                plot_topk_probabilities(
                    {"original": _result("original")},
                    output_png=out / "figure.png",
                    output_pdf=out / "figure.pdf",
                    source_entity="Spain",
                    target_entity="Germany",
                )

    def test_plotter_rejects_invalid_probabilities_before_matplotlib_import(self) -> None:
        topk_results = {
            "original": _result("original", probability=float("nan")),
            "target": _result("target"),
            "intervened": _result("intervened"),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)
            with self.assertRaisesRegex(ValueError, "Invalid probability"):
                plot_topk_probabilities(
                    topk_results,
                    output_png=out / "figure.png",
                    output_pdf=out / "figure.pdf",
                    source_entity="Spain",
                    target_entity="Germany",
                )

    def test_plotter_writes_png_and_pdf_when_matplotlib_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)
            png_path = out / "knowledge_edit_topk_probs.png"
            pdf_path = out / "knowledge_edit_topk_probs.pdf"
            plot_topk_probabilities(
                _topk_results(),
                output_png=png_path,
                output_pdf=pdf_path,
                source_entity="Spain",
                target_entity="Germany",
            )
            self.assertGreater(png_path.stat().st_size, 10_000)
            self.assertGreater(pdf_path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
