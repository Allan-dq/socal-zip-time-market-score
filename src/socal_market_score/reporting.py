"""Figures and the English technical report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RESEARCH_SOURCES


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#243447",
            "axes.edgecolor": "#AAB7C4",
            "text.color": "#243447",
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    frame = comparison.sort_values("outer_mae_mean", ascending=True).copy()
    colors = np.where(frame["selected"], "#D1495B", "#3B82A0")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(frame["candidate"], frame["outer_mae_mean"], color=colors, alpha=0.9)
    ax.errorbar(
        frame["outer_mae_mean"],
        frame["candidate"],
        xerr=frame["outer_mae_se"],
        fmt="none",
        ecolor="#243447",
        capsize=3,
        linewidth=1,
    )
    ax.set_title("Repeated nested cross-validation model comparison")
    ax.set_xlabel("Outer-fold mean absolute error (lower is better)")
    ax.set_ylabel("")
    _save(fig, path)


def plot_reference_validation(evaluation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.scatterplot(
        data=evaluation,
        x="y_true",
        y="y_pred",
        hue="Reference City",
        palette="tab10",
        s=62,
        alpha=0.85,
        ax=ax,
    )
    minimum = min(evaluation["y_true"].min(), evaluation["y_pred"].min()) - 3
    maximum = max(evaluation["y_true"].max(), evaluation["y_pred"].max()) + 3
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="#5F6B73")
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.set_title("Reference scores vs repeated nested-CV predictions")
    ax.set_xlabel("Reference Market Score")
    ax.set_ylabel("Out-of-fold predicted score")
    ax.legend(title="Reference city", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, path)


def plot_top_bottom(latest_scores: pd.DataFrame, path: Path) -> None:
    ordered = latest_scores.sort_values("MarketScore_hat")
    frame = pd.concat([ordered.head(10), ordered.tail(10)]).drop_duplicates("Zip Code")
    frame = frame.sort_values("MarketScore_hat")
    frame["label"] = frame["Zip Code"].astype(str) + " - " + frame["City"].astype(str)
    colors = ["#D1495B" if index < 10 else "#2A9D8F" for index in range(len(frame))]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(frame["label"], frame["MarketScore_hat"], color=colors)
    ax.set_xlim(0, 100)
    ax.set_title("Latest-2025 top and bottom ZIP market scores")
    ax.set_xlabel("Market Score")
    ax.set_ylabel("")
    for index, value in enumerate(frame["MarketScore_hat"]):
        ax.text(min(value + 1, 96), index, f"{value:.1f}", va="center", fontsize=9)
    _save(fig, path)


def plot_example_trend(
    all_scores: pd.DataFrame,
    example_zip: int,
    reference_score: float | None,
    path: Path,
) -> None:
    frame = all_scores[all_scores["Zip Code"].astype(int).eq(example_zip)].sort_values(
        "PERIOD_END"
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        frame["PERIOD_END"],
        frame["MarketScore_hat"],
        color="#26708E",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="MarketScore_hat",
    )
    if reference_score is not None and np.isfinite(reference_score):
        ax.axhline(
            reference_score,
            color="#D1495B",
            linestyle="--",
            linewidth=1.5,
            label=f"2025 reference score ({reference_score:.0f})",
        )
    city = frame["City"].iloc[-1] if not frame.empty else ""
    ax.set_ylim(0, 100)
    ax.set_title(f"Market Score over time: ZIP {example_zip} ({city})")
    ax.set_xlabel("Period end")
    ax.set_ylabel("Market Score")
    ax.legend(loc="best")
    _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    frame = importance.head(12).sort_values(
        "permutation_mae_increase_mean", ascending=True
    )
    if "standardized_coefficient_mean" in frame:
        colors = np.where(
            frame["standardized_coefficient_mean"].fillna(0).ge(0),
            "#2A9D8F",
            "#D1495B",
        )
    else:
        colors = "#3B82A0"
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(
        frame["feature"],
        frame["permutation_mae_increase_mean"],
        color=colors,
    )
    ax.set_title("Validation permutation importance")
    ax.set_xlabel("Increase in validation MAE after permutation")
    ax.set_ylabel("")
    _save(fig, path)


def plot_shift(feature_shift: pd.DataFrame, path: Path) -> None:
    frame = feature_shift.head(12).sort_values(
        "absolute_standardized_mean_difference", ascending=True
    )
    colors = np.where(frame["standardized_mean_difference"].ge(0), "#E9C46A", "#3B82A0")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(
        frame["feature"],
        frame["standardized_mean_difference"],
        color=colors,
    )
    ax.axvline(0, color="#243447", linewidth=0.8)
    ax.set_title("Calibration-to-deployment feature shift")
    ax.set_xlabel("Standardized mean difference (other ZIPs minus calibration ZIPs)")
    ax.set_ylabel("")
    _save(fig, path)


def plot_sanity_checks(sanity: pd.DataFrame, path: Path) -> None:
    frame = sanity.sort_values("spearman_correlation")
    colors = np.where(frame["direction_aligned"], "#2A9D8F", "#D1495B")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(frame["signal"], frame["spearman_correlation"], color=colors)
    ax.axvline(0, color="#243447", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title("Latest-score market-signal sanity checks")
    ax.set_xlabel("Spearman correlation with MarketScore_hat")
    ax.set_ylabel("")
    _save(fig, path)


def plot_stability(all_scores: pd.DataFrame, path: Path) -> None:
    ordered = all_scores.sort_values(["Zip Code", "PERIOD_END"]).copy()
    changes = ordered.groupby("Zip Code", observed=True)["MarketScore_hat"].diff().abs()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.histplot(changes.dropna(), bins=35, color="#3B82A0", ax=ax)
    ax.axvline(10, color="#E9C46A", linestyle="--", label="10-point change")
    ax.axvline(15, color="#D1495B", linestyle="--", label="15-point change")
    ax.set_title("Distribution of consecutive absolute score changes")
    ax.set_xlabel("Absolute score change")
    ax.set_ylabel("Observation pairs")
    ax.legend()
    _save(fig, path)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{value:.3f}")
        else:
            selected[column] = selected[column].map(
                lambda value: str(value).replace("|", "\\|")
            )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(path: Path, context: dict[str, Any]) -> None:
    """Write the completed English assignment report with measured results."""
    audit = context["audit"]
    selected = context["selected_spec"]
    oof = context["oof_metrics"]
    city = context["city_metrics"]
    stability = context["stability_summary"]
    shift = context["shift_summary"]
    comparison = context["comparison"].copy()
    comparison["selected"] = comparison["selected"].map({True: "Yes", False: ""})
    comparison_table = _markdown_table(
        comparison,
        [
            "candidate",
            "outer_mae_mean",
            "outer_mae_se",
            "oof_mae",
            "oof_rmse",
            "oof_r2",
            "selected",
        ],
    )
    importance_table = _markdown_table(
        context["feature_importance"].head(10),
        [
            "feature",
            "permutation_mae_increase_mean",
            "standardized_coefficient_mean",
            "positive_coefficient_share",
        ],
    )
    sanity_table = _markdown_table(
        context["sanity_checks"],
        [
            "signal",
            "expected_direction",
            "spearman_correlation",
            "direction_aligned",
        ],
    )
    shift_table = _markdown_table(
        context["feature_shift"].head(8),
        [
            "feature",
            "standardized_mean_difference",
            "ks_statistic",
            "ks_p_value",
            "wasserstein_distance",
        ],
    )
    city_table = _markdown_table(
        context["city_folds"].sort_values("mae"),
        ["held_out_city", "rows", "mae", "rmse", "r2", "spearman"],
    )
    representation_table = _markdown_table(
        context["feature_representation_screen"],
        ["representation", "feature_count", "mae_mean", "rmse_mean", "r2_mean"],
    )

    report = f"""# SoCal ZIP-by-Time Market Score Model

## Executive summary

This solution builds a single, bounded, explainable market score for every available Southern California ZIP-period record. It completes both the required latest-2025 calibration and every optional deliverable: full-period scoring, a time-series example, feature interpretation, geographic shift diagnostics, and output validation.

The usable calibration sample contains **{audit['calibration_rows']} ZIPs**. That is small relative to the correlated real-estate feature set, so model choice was based on repeated nested cross-validation rather than training fit. The selected specification is **{selected.name}**, chosen with the one-standard-error rule. Its repeated nested-CV out-of-fold MAE is **{oof['mae']:.2f} points** (bootstrap 95% interval: **{context['mae_ci'][0]:.2f} to {context['mae_ci'][1]:.2f}**), RMSE is **{oof['rmse']:.2f}**, R-squared is **{oof['r2']:.3f}**, and Spearman correlation is **{oof['spearman']:.3f}**.

## 1. Data scope and eligibility

- The SoCal sheet contains {audit['source_rows']:,} rows, {audit['unique_zips']:,} ZIPs, and {audit['unique_period_end_dates']} distinct period-end dates from {audit['period_start']} through {audit['period_end']}.
- The reference sheet contains {audit['reference_rows']} rows, of which {audit['numeric_reference_scores']} have numeric scores.
- Following the assignment definition, each ZIP's latest record is the maximum `PERIOD_END` within calendar year 2025. This yields {audit['latest_2025_rows']} latest-scoring rows.
- After excluding non-numeric reference scores and reference ZIPs with no 2025 record, {audit['calibration_rows']} rows remain for supervised calibration.
- {audit['zips_without_2025']} source ZIPs have no 2025 record and are excluded from latest-period output, but their existing historical rows remain in the all-period file.
- Duplicate `(Zip Code, PERIOD_END)` rows: {audit['duplicate_zip_period_rows']}. Invalid period durations: {audit['invalid_period_duration_rows']}.

No external market data was used.

## 2. Feature engineering and leakage control

The model uses the 12 contemporaneous level features: sale and list price, sale and list price per square foot, homes sold, pending sales, new listings, inventory, median days on market, average sale-to-list ratio, share sold above list, and share off market within two weeks. Short-horizon MoM and YoY changes were evaluated but were less reliable in cross-validation and increased temporal volatility.

Each feature is converted to a cross-sectional percentile rank within its exact `PERIOD_END`. This removes market-wide nominal drift, makes 2012 and 2025 values comparable, reduces outlier sensitivity, and keeps the score interpretation relative to the contemporaneous SoCal market. A trailing-365-day median representation was also evaluated explicitly.

Median imputation and standardization are fitted inside each cross-validation training fold through a scikit-learn `Pipeline`; held-out rows never influence fitted preprocessing. Final predictions are clipped to `[0, 100]`.

The fixed-Ridge representation screen below isolates the effect of adding volatile change features and trailing smoothing before model-family tuning:

{representation_table}

## 3. Model research and selection

The search covered a median baseline, Ridge, Elastic Net, Huber regression, partial least squares, RBF support-vector regression, random forest, extremely randomized trees, and histogram gradient boosting. The design reflects the following evidence:

- [Ridge]({RESEARCH_SOURCES['ridge']}) shrinks coefficients and is robust to correlated predictors.
- [Elastic Net]({RESEARCH_SOURCES['elastic_net']}) combines L1 and L2 penalties and encourages grouped selection among correlated features.
- [Huber regression]({RESEARCH_SOURCES['huber']}) reduces the influence of large residuals in small samples.
- [PLS regression]({RESEARCH_SOURCES['pls']}) is a supervised latent-component method for correlated predictors.
- [Random forests]({RESEARCH_SOURCES['random_forest']}) and [gradient boosting]({RESEARCH_SOURCES['gradient_boosting']}) provide nonlinear alternatives.
- [RBF SVR]({RESEARCH_SOURCES['svr']}) provides a regularized nonlinear kernel benchmark suited to small samples.
- Nested CV is used because evaluating the same CV run used for tuning can be optimistically biased; the [Varma and Simon study]({RESEARCH_SOURCES['nested_cv']}) shows that tuning must be repeated inside each outer fold.
- The preprocessing design follows scikit-learn's [data-leakage guidance]({RESEARCH_SOURCES['leakage']}).

Five repeats of five-fold outer CV were used, with four-fold inner tuning by MAE. The one-standard-error rule selects the least complex model statistically close to the best mean outer-fold MAE.

{comparison_table}

Final full-sample tuning parameters: `{context['best_params']}`.

## 4. Reference-score validation

The required evaluation file contains repeated nested-CV predictions, not in-sample fitted values. This is the most defensible estimate available without a separate labeled test set.

- OOF MAE: {oof['mae']:.3f}
- OOF RMSE: {oof['rmse']:.3f}
- OOF R-squared: {oof['r2']:.3f}
- OOF Spearman correlation: {oof['spearman']:.3f}
- Fitted full-sample MAE (descriptive only): {context['training_metrics']['mae']:.3f}

### Geographic stress test

Holding out each reference city in turn is deliberately harder than random CV. Aggregate city-held-out MAE is {city['mae']:.3f}, RMSE is {city['rmse']:.3f}, and R-squared is {city['r2']:.3f}. These results quantify the risk of transporting labels from a small set of representative cities.

{city_table}

## 5. Explainability

Permutation importance is measured on outer validation folds, so it reflects the increase in held-out MAE after shuffling a feature. Standardized coefficients show the direction and approximate score contribution of a one-standard-deviation percentile-rank movement. Correlated features can share importance, so individual values should not be interpreted causally. See scikit-learn's [permutation-importance guidance]({RESEARCH_SOURCES['permutation_importance']}).

{importance_table}

## 6. Latest-period sanity checks

The following Spearman correlations test whether scores co-move with recognizable market signals. They are diagnostics, not constraints; inventory is especially ambiguous because it can proxy both supply pressure and market size.

{sanity_table}

## 7. Full-period scoring and temporal stability

All {audit['source_rows']:,} existing ZIP-period rows are scored; missing time points are not created. The example trend uses ZIP {context['example_zip']}.

- Median absolute consecutive change: {stability['median_absolute_consecutive_change']:.2f} points
- 95th percentile absolute consecutive change: {stability['p95_absolute_consecutive_change']:.2f} points
- Share of consecutive changes above 15 points: {stability['share_consecutive_change_over_15']:.2%}
- Median absolute approximately year-over-year change: {stability['median_absolute_year_over_year_change']:.2f} points
- 95th percentile absolute approximately year-over-year change: {stability['p95_absolute_year_over_year_change']:.2f} points
- Share of approximately year-over-year changes above 15 points: {stability['share_year_over_year_change_over_15']:.2%}

The percentile-rank representation is the main protection against nominal price drift. The score remains descriptive and contemporaneous; it is not a forecast.

## 8. Distribution-shift bonus

The selected reference ZIPs are compared with the other latest-2025 scoring ZIPs using per-feature standardized mean differences, two-sample Kolmogorov-Smirnov statistics, Wasserstein distance, and adversarial validation. The [KS test]({RESEARCH_SOURCES['ks_test']}) compares empirical distributions. The adversarial classifier's repeated-CV AUC is **{shift['adversarial_auc_mean']:.3f} ± {shift['adversarial_auc_std']:.3f}**; 0.5 would indicate no detectable separation. {shift['features_abs_smd_over_0_5']} selected features have an absolute standardized mean difference above 0.5, and {shift['target_ood_share']:.1%} of non-calibration latest ZIPs exceed the calibration sample's 95th-percentile nearest-neighbor distance threshold.

{shift_table}

This is evidence of possible covariate shift, not proof that the score relationship changes. Under covariate shift, ordinary CV can misrepresent deployment risk; see [Sugiyama et al.]({RESEARCH_SOURCES['covariate_shift']}). Recommended production actions are to collect labels from underrepresented counties, monitor the same diagnostics by refresh, flag distant ZIPs for review, and evaluate importance weighting or regional recalibration only after obtaining a defensible target-domain validation set.

## 9. Deliverables

- `outputs/scores_latest_select_zips_eval.csv`: valid reference ZIPs with honest OOF predictions.
- `outputs/scores_latest_all_zips.csv`: all ZIPs with a 2025 record, using each ZIP's latest 2025 row.
- `outputs/scores_all_periods.csv`: every existing source ZIP-period row.
- `outputs/model_comparison.csv`, `outputs/feature_importance.csv`, `outputs/sanity_checks.csv`, `outputs/feature_shift_diagnostics.csv`, and supporting validation diagnostics.
- `figures/`: model comparison, validation, top/bottom ZIPs, example trend, feature importance, sanity checks, stability, and distribution-shift plots.
- `outputs/market_score_model.joblib`: fitted preprocessing pipeline, selected features, and metadata.

## 10. Assumptions and limitations

1. Only {audit['calibration_rows']} usable labels are available, concentrated in six cities. CV uncertainty and geographic transport risk remain material.
2. The reference score is treated as the supervised target; its construction and label uncertainty are unknown.
3. Percentile ranks are batch-relative. Scoring a new period requires a contemporaneous cohort of ZIP feature rows.
4. The model is associative, not causal. Coefficients and permutation importance do not establish that changing a market metric will change the true underlying market quality.
5. Clipping guarantees the required bounds but can compress extreme predictions.
6. No future market values or external datasets are used. This solution must not be presented as a price or market forecast.

## 11. Reproducibility

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_analysis.py
.venv/bin/python scripts/build_notebook.py
.venv/bin/python -m unittest discover -s tests
```

Randomness is fixed at seed 42. Package versions are pinned in `requirements.txt`.
"""
    path.write_text(report, encoding="utf-8")
