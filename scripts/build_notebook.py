#!/usr/bin/env python3
"""Build and execute the analysis notebook."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "socal_market_score_model.ipynb"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "socal-market-score-matplotlib")
)


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip())


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown(
            """
# SoCal ZIP-by-Time Market Score Model

This notebook is the executable companion to the technical report. It covers the latest-2025 calibration and the extended analyses: all-period scoring, a time-series example, feature interpretation, temporal validation, and geographic distribution-shift analysis.

The score is contemporaneous and descriptive. It is not a forecast.
"""
        ),
        code(
            """
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "socal-market-score-matplotlib"))

import pandas as pd
from IPython.display import Image, display

from socal_market_score.workflow import run_workflow

INPUT_PATH = ROOT / "data" / "raw" / "SoCal - Real Estate Market Metrics - ZIP×Time Score Series.xlsx"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
REPORT_PATH = ROOT / "REPORT.md"
"""
        ),
        markdown(
            """
## 1. Rebuild the complete solution

All preprocessing is learned inside model pipelines. Candidate hyperparameters are tuned inside the inner cross-validation loop, and the outer loop provides honest model-comparison predictions.
"""
        ),
        code(
            """
context = run_workflow(
    input_path=INPUT_PATH,
    output_dir=OUTPUT_DIR,
    figure_dir=FIGURE_DIR,
    report_path=REPORT_PATH,
)

pd.Series(context["oof_metrics"], name="Repeated nested-CV metric")
"""
        ),
        markdown("## 2. Data eligibility and coverage"),
        code('pd.Series(context["audit"], name="Value").to_frame()'),
        markdown(
            """
The calibration rows are the intersection of (1) numeric reference scores and (2) ZIPs with at least one 2025 record. Each ZIP's latest row is the maximum `PERIOD_END` within calendar year 2025. ZIPs without 2025 records are excluded from latest scoring, as required.
"""
        ),
        code('context["exclusions"].head(15)'),
        markdown(
            """
## 3. Research-backed model search

The candidate set covers Ridge, Elastic Net, Huber regression, PLS, RBF SVR, random forest, extra trees, and histogram gradient boosting, plus a median baseline.

- Ridge: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
- Elastic Net: https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00503.x
- Random forests: https://doi.org/10.1023/A:1010933404324
- Gradient boosting: https://doi.org/10.1214/aos/1013203451
- Nested CV bias: https://pubmed.ncbi.nlm.nih.gov/16504092/
- Data leakage: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage

The one-standard-error rule selects the simplest model statistically close to the best mean outer-fold MAE.
"""
        ),
        markdown("### Feature-representation screen"),
        code('context["feature_representation_screen"]'),
        markdown("### Repeated nested-CV model comparison"),
        code(
            """
comparison_columns = [
    "candidate", "outer_mae_mean", "outer_mae_se", "oof_mae",
    "oof_rmse", "oof_r2", "oof_spearman", "within_one_se", "selected",
]
context["comparison"][comparison_columns]
"""
        ),
        code('display(Image(filename=str(FIGURE_DIR / "model_comparison.png")))'),
        markdown(
            """
## 4. Reference-score validation

`y_pred` in the evaluation CSV is the mean repeated nested-CV out-of-fold prediction for each labeled ZIP, not an in-sample fitted value.
"""
        ),
        code(
            """
evaluation = context["evaluation"].copy()
evaluation[["Zip Code", "City", "County", "PERIOD_END", "y_true", "y_pred", "residual"]].head(15)
"""
        ),
        code('display(Image(filename=str(FIGURE_DIR / "reference_validation.png")))'),
        markdown("## 5. Feature importance and coefficient direction"),
        code('context["feature_importance"].head(12)'),
        code('display(Image(filename=str(FIGURE_DIR / "feature_importance.png")))'),
        markdown("## 6. Latest-period outputs and sanity checks"),
        code(
            """
latest_scores = pd.read_csv(OUTPUT_DIR / "scores_latest_all_zips.csv")
display(latest_scores.nlargest(10, "MarketScore_hat"))
display(latest_scores.nsmallest(10, "MarketScore_hat"))
"""
        ),
        code('context["sanity_checks"]'),
        code('display(Image(filename=str(FIGURE_DIR / "top_bottom_latest.png")))'),
        code('display(Image(filename=str(FIGURE_DIR / "sanity_checks.png")))'),
        markdown("## 7. All-period score series and stability"),
        code('pd.Series(context["stability_summary"], name="Value").to_frame()'),
        code('display(Image(filename=str(FIGURE_DIR / "score_trend_example.png")))'),
        code('display(Image(filename=str(FIGURE_DIR / "temporal_stability.png")))'),
        markdown(
            """
## 8. Distribution-shift analysis

Feature-level standardized differences and KS statistics are complemented by adversarial validation. A high adversarial AUC means the selected reference ZIPs can be distinguished from the other latest-2025 ZIPs, which warns that random CV may not fully describe deployment risk.
"""
        ),
        code('pd.Series(context["shift_summary"], name="Value").to_frame()'),
        code('context["feature_shift"].head(12)'),
        code('display(Image(filename=str(FIGURE_DIR / "distribution_shift.png")))'),
        markdown("## 9. Output audit"),
        code(
            """
required_files = [
    OUTPUT_DIR / "scores_latest_select_zips_eval.csv",
    OUTPUT_DIR / "scores_latest_all_zips.csv",
    OUTPUT_DIR / "scores_all_periods.csv",
    OUTPUT_DIR / "market_score_model.joblib",
    REPORT_PATH,
]
pd.DataFrame({
    "file": [str(path.relative_to(ROOT)) for path in required_files],
    "exists": [path.exists() for path in required_files],
    "bytes": [path.stat().st_size if path.exists() else 0 for path in required_files],
})
"""
        ),
        markdown(
            """
## 10. Conclusion

The selected regularized model balances predictive accuracy, stability, and interpretability in a small, geographically concentrated label set. The technical report contains the full measured results, limitations, and production recommendations. In particular, the detected geographic covariate shift should be addressed with broader reference labeling before the score is used for high-stakes decisions.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    return notebook


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbformat.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(
        notebook,
        timeout=1_800,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Executed notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
