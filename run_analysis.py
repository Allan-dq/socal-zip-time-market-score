#!/usr/bin/env python3
"""Run the complete SoCal ZIP-by-time market-score assignment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "socal-market-score-matplotlib")
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from socal_market_score.workflow import run_workflow


DEFAULT_INPUT = (
    ROOT
    / "data"
    / "raw"
    / "SoCal - Real Estate Market Metrics - ZIP×Time Score Series.xlsx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, validate, and export the SoCal Market Score model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--report", type=Path, default=ROOT / "REPORT.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = run_workflow(
        input_path=args.input,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        report_path=args.report,
    )
    metrics = context["oof_metrics"]
    print(
        f"Selected {context['selected_spec'].name}; "
        f"OOF MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
        f"R2={metrics['r2']:.3f}."
    )


if __name__ == "__main__":
    main()
