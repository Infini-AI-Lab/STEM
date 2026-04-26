"""CLI entry point for the STEM diagnostics dashboard.

Reads artifacts from a previous diagnostics run and produces:

* ``path_interference_dashboard.json``
* ``debuggability_report.json``
* ``diagnostics_summary.md``

Usage::

    python -m apps.main.diagnostic_dashboard --diagnostics-dir path/to/diagnostics

Or with explicit output directory and run ID::

    python -m apps.main.diagnostic_dashboard \\
        --diagnostics-dir path/to/diagnostics \\
        --output-dir path/to/out \\
        --run-id my_run \\
        --total-layers 32

No model loading is required.  The script is safe to run on any machine
that has the diagnostics artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m apps.main.diagnostic_dashboard",
        description=(
            "Generate a diagnostics dashboard from STEM eval artifacts. "
            "Reads JSONL / JSON artifact files and writes three output files: "
            "path_interference_dashboard.json, debuggability_report.json, "
            "and diagnostics_summary.md."
        ),
    )
    p.add_argument(
        "--diagnostics-dir",
        required=True,
        metavar="DIR",
        help="Directory containing the input artifact files "
             "(layer_path_metrics.jsonl, interventions_task_aligned.jsonl, etc.).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to write output files into (default: same as --diagnostics-dir).",
    )
    p.add_argument(
        "--run-id",
        default="unknown",
        metavar="ID",
        help="Run identifier propagated into DebuggabilityRecord rows (default: 'unknown').",
    )
    p.add_argument(
        "--total-layers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Total number of transformer layers (used for the B1 breadth rule). "
            "If omitted, estimated from layer_path_metrics.jsonl data."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        default=False,
        help="Print the global verdict and task-level classifications to stdout after writing files.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    diag_dir = Path(args.diagnostics_dir)
    if not diag_dir.exists():
        print(
            f"ERROR: --diagnostics-dir does not exist: {diag_dir}",
            file=sys.stderr,
        )
        return 2

    from lingua.diagnostic_dashboard import run_diagnostic_dashboard

    result = run_diagnostic_dashboard(
        diagnostics_dir=diag_dir,
        run_id=args.run_id,
        output_dir=args.output_dir,
        total_layers=args.total_layers,
    )

    out_dir = Path(args.output_dir) if args.output_dir else diag_dir
    pi_path = out_dir / "path_interference_dashboard.json"
    dr_path = out_dir / "debuggability_report.json"
    md_path = result.get("markdown_path", str(out_dir / "diagnostics_summary.md"))

    print("Done. Output files written:")
    print(f"  {pi_path}")
    print(f"  {dr_path}")
    print(f"  {md_path}")

    if args.print_summary:
        dr = result.get("debuggability_report") or {}
        print("\n=== Debuggability Report ===")
        print(f"Global classification: {dr.get('global_classification', 'unknown')}")
        print(f"Summary counts: {json.dumps(dr.get('summary_counts') or {})}")
        print("Per-task verdicts:")
        for rec in dr.get("per_task") or []:
            print(
                f"  {rec.get('task','?'):40s}  {rec.get('classification','?'):25s}  "
                f"confidence={rec.get('confidence', 0.0):.2f}"
            )
        avail = result.get("artifacts_available") or {}
        present = [k for k, v in avail.items() if v]
        missing = [k for k, v in avail.items() if not v]
        if present:
            print(f"\nArtifacts present: {', '.join(present)}")
        if missing:
            print(f"Artifacts missing: {', '.join(missing)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
