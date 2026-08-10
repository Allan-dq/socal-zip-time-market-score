"""End-to-end orchestration for the SoCal market-score assignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import RANDOM_STATE, SCORE_MAX, SCORE_MIN
from .data import (
    build_calibration_set,
    build_reference_exclusions,
    data_audit,
    engineer_features,
    latest_2025_rows,
    load_workbook,
)
from .modeling import (
    bootstrap_mae_interval,
    candidate_specs,
    distribution_shift_diagnostics,
    feature_representation_screen,
    fit_final_model,
    group_stress_test,
    regression_metrics,
    run_nested_model_comparison,
    sanity_checks,
    select_one_standard_error,
    temporal_stability,
    validation_feature_importance,
)
from .reporting import (
    configure_plotting,
    plot_example_trend,
    plot_feature_importance,
    plot_model_comparison,
    plot_reference_validation,
    plot_sanity_checks,
    plot_shift,
    plot_stability,
    plot_top_bottom,
    write_report,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _score_frame(
    frame: pd.DataFrame,
    model: Any,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    scored = frame.copy()
    raw_prediction = np.asarray(model.predict(scored[list(feature_columns)])).ravel()
    scored["MarketScore_hat"] = np.clip(raw_prediction, SCORE_MIN, SCORE_MAX)
    return scored


def _example_zip(
    all_scored: pd.DataFrame, calibration: pd.DataFrame
) -> tuple[int, float | None]:
    eligible_zips = set(calibration["Zip Code"].astype(int))
    counts = (
        all_scored[all_scored["Zip Code"].astype(int).isin(eligible_zips)]
        .groupby("Zip Code", observed=True)
        .size()
        .sort_values(ascending=False)
    )
    zip_code = int(counts.index[0])
    score = calibration.loc[
        calibration["Zip Code"].astype(int).eq(zip_code), "y_true"
    ]
    return zip_code, float(score.iloc[0]) if not score.empty else None


def run_workflow(
    input_path: Path,
    output_dir: Path,
    figure_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Run model selection, fit, scoring, diagnostics, and reporting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    print("Loading and validating workbook...", flush=True)
    socal_raw, reference = load_workbook(input_path)
    socal = engineer_features(socal_raw)
    latest = latest_2025_rows(socal)
    calibration = build_calibration_set(latest, reference)
    audit = data_audit(socal_raw, reference, latest, calibration)
    exclusions = build_reference_exclusions(socal_raw, latest, reference)

    representation_screen = feature_representation_screen(calibration)

    print("Running repeated nested model comparison...", flush=True)
    specs = candidate_specs()
    comparison, nested_folds, prediction_map = run_nested_model_comparison(
        calibration, specs
    )
    selected_spec, comparison = select_one_standard_error(comparison, specs)
    selected_prediction = prediction_map[selected_spec.name]
    y_true = calibration["y_true"].to_numpy(dtype=float)
    oof_metrics = regression_metrics(y_true, selected_prediction)
    mae_ci = bootstrap_mae_interval(y_true, selected_prediction)

    print(f"Selected model: {selected_spec.name}", flush=True)
    final_model, best_params, final_cv_results = fit_final_model(
        calibration, selected_spec
    )
    fitted_prediction = np.clip(
        np.asarray(
            final_model.predict(calibration[list(selected_spec.feature_columns)])
        ).ravel(),
        SCORE_MIN,
        SCORE_MAX,
    )
    training_metrics = regression_metrics(y_true, fitted_prediction)

    print("Running geographic stress test and interpretation diagnostics...", flush=True)
    city_folds, city_metrics, city_detail = group_stress_test(
        calibration, selected_spec
    )
    feature_importance = validation_feature_importance(
        calibration, selected_spec
    )

    latest_scored = _score_frame(
        latest, final_model, selected_spec.feature_columns
    )
    all_scored = _score_frame(socal, final_model, selected_spec.feature_columns)
    sanity = sanity_checks(latest_scored)
    stability_summary, stability_by_zip = temporal_stability(all_scored)
    shift_summary, feature_shift, shift_flags = distribution_shift_diagnostics(
        calibration, latest, selected_spec.feature_columns
    )

    evaluation = calibration[
        [
            "Zip Code",
            "City",
            "County",
            "PERIOD_END",
            "y_true",
            "Reference City",
            "Reference County",
        ]
    ].copy()
    evaluation["y_pred"] = selected_prediction
    evaluation["residual"] = evaluation["y_true"] - evaluation["y_pred"]

    latest_output = latest_scored[
        ["Zip Code", "City", "County", "PERIOD_END", "MarketScore_hat"]
    ].copy()
    latest_output["PERIOD"] = latest_output.pop("PERIOD_END").dt.strftime("%Y-%m-%d")
    latest_output = latest_output[
        ["Zip Code", "City", "County", "PERIOD", "MarketScore_hat"]
    ]
    latest_output["MarketScore_hat"] = latest_output["MarketScore_hat"].round(2)

    all_output = all_scored[
        ["Zip Code", "City", "County", "PERIOD_END", "MarketScore_hat"]
    ].copy()
    all_output["PERIOD"] = all_output.pop("PERIOD_END").dt.strftime("%Y-%m-%d")
    all_output = all_output[
        ["Zip Code", "City", "County", "PERIOD", "MarketScore_hat"]
    ]
    all_output["MarketScore_hat"] = all_output["MarketScore_hat"].round(2)

    evaluation_output = evaluation[
        ["Zip Code", "City", "County", "PERIOD_END", "y_true", "y_pred"]
    ].copy()
    evaluation_output["PERIOD"] = evaluation_output.pop("PERIOD_END").dt.strftime(
        "%Y-%m-%d"
    )
    evaluation_output = evaluation_output[
        ["Zip Code", "City", "County", "PERIOD", "y_true", "y_pred"]
    ]
    evaluation_output["y_pred"] = evaluation_output["y_pred"].round(4)

    print("Writing outputs and figures...", flush=True)
    evaluation_output.to_csv(
        output_dir / "scores_latest_select_zips_eval.csv", index=False
    )
    latest_output.to_csv(output_dir / "scores_latest_all_zips.csv", index=False)
    all_output.to_csv(output_dir / "scores_all_periods.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    nested_folds.to_csv(output_dir / "nested_cv_folds.csv", index=False)
    city_folds.to_csv(output_dir / "city_holdout_metrics.csv", index=False)
    city_detail.to_csv(output_dir / "city_holdout_predictions.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    sanity.to_csv(output_dir / "sanity_checks.csv", index=False)
    stability_by_zip.to_csv(output_dir / "temporal_stability_by_zip.csv", index=False)
    feature_shift.to_csv(output_dir / "feature_shift_diagnostics.csv", index=False)
    shift_flags.to_csv(output_dir / "latest_distribution_shift_flags.csv", index=False)
    exclusions.to_csv(output_dir / "reference_exclusions.csv", index=False)
    representation_screen.to_csv(
        output_dir / "feature_representation_screen.csv", index=False
    )
    final_cv_results[
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "params",
        ]
    ].to_csv(output_dir / "final_model_tuning.csv", index=False)

    metrics_payload = {
        "selected_model": selected_spec.name,
        "best_params": best_params,
        "oof_metrics": oof_metrics,
        "oof_mae_bootstrap_95_interval": list(mae_ci),
        "training_metrics_descriptive_only": training_metrics,
        "city_holdout_metrics": city_metrics,
        "temporal_stability": stability_summary,
        "distribution_shift": shift_summary,
    }
    _write_json(output_dir / "data_audit.json", audit)
    _write_json(output_dir / "metrics.json", metrics_payload)

    model_bundle = {
        "pipeline": final_model,
        "selected_model": selected_spec.name,
        "feature_columns": list(selected_spec.feature_columns),
        "best_params": best_params,
        "score_bounds": [SCORE_MIN, SCORE_MAX],
        "random_state": RANDOM_STATE,
        "feature_engineering": (
            "Cross-sectional percentile ranks within PERIOD_END; optional trailing "
            "features use a backward-looking 365-day median."
        ),
    }
    joblib.dump(model_bundle, output_dir / "market_score_model.joblib")

    example_zip, example_reference_score = _example_zip(all_scored, calibration)
    plot_model_comparison(comparison, figure_dir / "model_comparison.png")
    plot_reference_validation(evaluation, figure_dir / "reference_validation.png")
    plot_top_bottom(latest_scored, figure_dir / "top_bottom_latest.png")
    plot_example_trend(
        all_scored,
        example_zip,
        example_reference_score,
        figure_dir / "score_trend_example.png",
    )
    plot_feature_importance(
        feature_importance, figure_dir / "feature_importance.png"
    )
    plot_shift(feature_shift, figure_dir / "distribution_shift.png")
    plot_sanity_checks(sanity, figure_dir / "sanity_checks.png")
    plot_stability(all_scored, figure_dir / "temporal_stability.png")

    context = {
        "audit": audit,
        "selected_spec": selected_spec,
        "best_params": best_params,
        "oof_metrics": oof_metrics,
        "mae_ci": mae_ci,
        "training_metrics": training_metrics,
        "city_metrics": city_metrics,
        "city_folds": city_folds,
        "stability_summary": stability_summary,
        "shift_summary": shift_summary,
        "comparison": comparison,
        "feature_importance": feature_importance,
        "feature_representation_screen": representation_screen,
        "sanity_checks": sanity,
        "feature_shift": feature_shift,
        "example_zip": example_zip,
        "evaluation": evaluation,
        "latest_scores": latest_output,
        "all_scores": all_output,
        "exclusions": exclusions,
    }
    write_report(report_path, context)
    print("Workflow complete.", flush=True)
    return context
