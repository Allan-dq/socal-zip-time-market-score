"""Model comparison, validation, interpretability, and shift diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneGroupOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    cross_val_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import (
    CURRENT_ALL_FEATURES,
    CURRENT_LEVEL_FEATURES,
    CURRENT_LEVEL_YOY_FEATURES,
    RANDOM_STATE,
    SCORE_MAX,
    SCORE_MIN,
    TRAILING_ALL_FEATURES,
    TRAILING_LEVEL_FEATURES,
    TRAILING_LEVEL_YOY_FEATURES,
)


@dataclass(frozen=True)
class CandidateSpec:
    """A model family, feature representation, and tuning grid."""

    name: str
    feature_columns: tuple[str, ...]
    model: Any
    param_grid: dict[str, list[Any]]
    complexity_rank: int

    def make_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", clone(self.model)),
            ]
        )


def candidate_specs(random_state: int = RANDOM_STATE) -> list[CandidateSpec]:
    """Return the predeclared model search space used by nested CV."""
    current = tuple(CURRENT_LEVEL_FEATURES)
    trailing = tuple(TRAILING_LEVEL_FEATURES)
    ridge_grid = {"model__alpha": list(np.logspace(-2, 3, 11))}
    huber_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0],
        "model__epsilon": [1.2, 1.35, 1.7],
    }
    return [
        CandidateSpec(
            "Dummy median | current levels",
            current,
            DummyRegressor(strategy="median"),
            {},
            0,
        ),
        CandidateSpec("Ridge | current levels", current, Ridge(), ridge_grid, 1),
        CandidateSpec("Ridge | trailing-year levels", trailing, Ridge(), ridge_grid, 2),
        CandidateSpec(
            "Elastic net | current levels",
            current,
            ElasticNet(max_iter=50_000, random_state=random_state),
            {
                "model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
            3,
        ),
        CandidateSpec(
            "Huber | current levels",
            current,
            HuberRegressor(max_iter=5_000),
            huber_grid,
            4,
        ),
        CandidateSpec(
            "Huber | trailing-year levels",
            trailing,
            HuberRegressor(max_iter=5_000),
            huber_grid,
            5,
        ),
        CandidateSpec(
            "PLS | current levels",
            current,
            PLSRegression(scale=False, max_iter=2_000),
            {"model__n_components": [2, 3, 4, 5, 6, 8]},
            6,
        ),
        CandidateSpec(
            "RBF SVR | current levels",
            current,
            SVR(),
            {
                "model__C": [1.0, 10.0, 100.0],
                "model__epsilon": [1.0, 5.0],
                "model__gamma": ["scale", 0.1],
            },
            10,
        ),
        CandidateSpec(
            "Random forest | current levels",
            current,
            RandomForestRegressor(
                n_estimators=150, random_state=random_state, n_jobs=1
            ),
            {
                "model__max_depth": [3, None],
                "model__min_samples_leaf": [3, 8],
                "model__max_features": [0.7],
            },
            8,
        ),
        CandidateSpec(
            "Extra trees | current levels",
            current,
            ExtraTreesRegressor(
                n_estimators=150, random_state=random_state, n_jobs=1
            ),
            {
                "model__max_depth": [3, None],
                "model__min_samples_leaf": [3, 8],
                "model__max_features": [0.7],
            },
            7,
        ),
        CandidateSpec(
            "Histogram gradient boosting | current levels",
            current,
            HistGradientBoostingRegressor(
                max_iter=200, random_state=random_state, early_stopping=False
            ),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [5, 10],
                "model__min_samples_leaf": [10, 20],
                "model__l2_regularization": [10.0],
            },
            9,
        ),
    ]


def feature_representation_screen(
    calibration: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare stable levels with MoM/YoY and trailing-year alternatives."""
    representations = {
        "Current levels": CURRENT_LEVEL_FEATURES,
        "Current levels + YoY": CURRENT_LEVEL_YOY_FEATURES,
        "Current levels + MoM + YoY": CURRENT_ALL_FEATURES,
        "Trailing-year levels": TRAILING_LEVEL_FEATURES,
        "Trailing-year levels + YoY": TRAILING_LEVEL_YOY_FEATURES,
        "Trailing-year levels + MoM + YoY": TRAILING_ALL_FEATURES,
    }
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    rows = []
    for name, columns in representations.items():
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
        scores = cross_validate(
            pipeline,
            calibration[columns],
            calibration["y_true"].to_numpy(dtype=float),
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "representation": name,
                "feature_count": len(columns),
                "mae_mean": float(-scores["test_mae"].mean()),
                "mae_std": float(scores["test_mae"].std(ddof=1)),
                "rmse_mean": float(-scores["test_rmse"].mean()),
                "r2_mean": float(scores["test_r2"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)


def _clip(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(predictions, dtype=float).ravel(), SCORE_MIN, SCORE_MAX)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the primary score-alignment metrics."""
    correlation = spearmanr(y_true, y_pred, nan_policy="omit").statistic
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
    }


def _json_params(params: dict[str, Any]) -> str:
    clean = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in params.items()
    }
    return json.dumps(clean, sort_keys=True)


def evaluate_candidate_nested(
    calibration: pd.DataFrame,
    spec: CandidateSpec,
    outer_splits: int = 5,
    outer_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Evaluate one candidate with repeated nested cross-validation."""
    X = calibration[list(spec.feature_columns)]
    y = calibration["y_true"].to_numpy(dtype=float)
    outer_cv = RepeatedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=random_state,
    )
    prediction_sum = np.zeros(len(calibration), dtype=float)
    prediction_count = np.zeros(len(calibration), dtype=int)
    fold_rows: list[dict[str, Any]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y), start=1):
            inner_cv = KFold(
                n_splits=4,
                shuffle=True,
                random_state=random_state + fold,
            )
            search = GridSearchCV(
                spec.make_pipeline(),
                spec.param_grid,
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(X.iloc[train_index], y[train_index])
            prediction = _clip(search.predict(X.iloc[test_index]))
            prediction_sum[test_index] += prediction
            prediction_count[test_index] += 1
            fold_metric = regression_metrics(y[test_index], prediction)
            fold_rows.append(
                {
                    "candidate": spec.name,
                    "fold": fold,
                    "repeat": (fold - 1) // outer_splits + 1,
                    "fold_within_repeat": (fold - 1) % outer_splits + 1,
                    "train_rows": len(train_index),
                    "test_rows": len(test_index),
                    "mae": fold_metric["mae"],
                    "rmse": fold_metric["rmse"],
                    "r2": fold_metric["r2"],
                    "spearman": fold_metric["spearman"],
                    "best_params": _json_params(search.best_params_),
                }
            )

    if (prediction_count == 0).any():
        raise RuntimeError(f"Candidate {spec.name} did not predict every calibration row.")
    oof_prediction = prediction_sum / prediction_count
    fold_frame = pd.DataFrame(fold_rows)
    aggregate = regression_metrics(y, oof_prediction)
    summary = {
        "candidate": spec.name,
        "feature_count": len(spec.feature_columns),
        "complexity_rank": spec.complexity_rank,
        "outer_folds": int(len(fold_frame)),
        "outer_mae_mean": float(fold_frame["mae"].mean()),
        "outer_mae_std": float(fold_frame["mae"].std(ddof=1)),
        "outer_mae_se": float(
            fold_frame["mae"].std(ddof=1) / np.sqrt(len(fold_frame))
        ),
        "outer_rmse_mean": float(fold_frame["rmse"].mean()),
        "oof_mae": aggregate["mae"],
        "oof_rmse": aggregate["rmse"],
        "oof_r2": aggregate["r2"],
        "oof_spearman": aggregate["spearman"],
    }
    return summary, fold_frame, oof_prediction


def run_nested_model_comparison(
    calibration: pd.DataFrame,
    specs: list[CandidateSpec] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate the full model search space and return honest OOF predictions."""
    specs = specs or candidate_specs()
    summaries = []
    fold_frames = []
    predictions: dict[str, np.ndarray] = {}
    for spec in specs:
        summary, folds, oof_prediction = evaluate_candidate_nested(calibration, spec)
        summaries.append(summary)
        fold_frames.append(folds)
        predictions[spec.name] = oof_prediction
        print(
            f"Nested CV: {spec.name}: MAE={summary['oof_mae']:.3f}, "
            f"RMSE={summary['oof_rmse']:.3f}",
            flush=True,
        )
    comparison = pd.DataFrame(summaries).sort_values(
        ["outer_mae_mean", "complexity_rank"]
    )
    return comparison.reset_index(drop=True), pd.concat(fold_frames, ignore_index=True), predictions


def select_one_standard_error(
    comparison: pd.DataFrame, specs: list[CandidateSpec]
) -> tuple[CandidateSpec, pd.DataFrame]:
    """Select the simplest candidate within one standard error of the best."""
    ranked = comparison.copy()
    best = ranked.loc[ranked["outer_mae_mean"].idxmin()]
    threshold = float(best["outer_mae_mean"] + best["outer_mae_se"])
    ranked["within_one_se"] = ranked["outer_mae_mean"].le(threshold)
    eligible = ranked[ranked["within_one_se"]].sort_values(
        ["complexity_rank", "outer_mae_mean"]
    )
    selected_name = str(eligible.iloc[0]["candidate"])
    ranked["selected"] = ranked["candidate"].eq(selected_name)
    spec_lookup = {spec.name: spec for spec in specs}
    return spec_lookup[selected_name], ranked.sort_values("outer_mae_mean").reset_index(drop=True)


def bootstrap_mae_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 5_000,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Return a non-parametric 95% interval for OOF MAE."""
    rng = np.random.default_rng(random_state)
    n_rows = len(y_true)
    values = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n_rows, size=n_rows)
        values[index] = mean_absolute_error(y_true[sample], y_pred[sample])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def fit_final_model(
    calibration: pd.DataFrame,
    spec: CandidateSpec,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Tune the selected family on all calibration rows and refit it."""
    X = calibration[list(spec.feature_columns)]
    y = calibration["y_true"].to_numpy(dtype=float)
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=random_state)
    search = GridSearchCV(
        spec.make_pipeline(),
        spec.param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        search.fit(X, y)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    best_params = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in search.best_params_.items()
    }
    return search.best_estimator_, best_params, cv_results


def group_stress_test(
    calibration: pd.DataFrame,
    spec: CandidateSpec,
    group_column: str = "Reference City",
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Stress-test geographic transportability by holding out one city at a time."""
    X = calibration[list(spec.feature_columns)]
    y = calibration["y_true"].to_numpy(dtype=float)
    groups = calibration[group_column].astype(str).to_numpy()
    prediction = np.full(len(calibration), np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    splitter = LeaveOneGroupOut()
    for fold, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + fold)
        search = GridSearchCV(
            spec.make_pipeline(),
            spec.param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            search.fit(X.iloc[train_index], y[train_index])
        held_out = str(groups[test_index][0])
        fold_prediction = _clip(search.predict(X.iloc[test_index]))
        prediction[test_index] = fold_prediction
        metrics = regression_metrics(y[test_index], fold_prediction)
        rows.append(
            {
                "held_out_city": held_out,
                "rows": len(test_index),
                **metrics,
                "best_params": _json_params(search.best_params_),
            }
        )
    if np.isnan(prediction).any():
        raise RuntimeError("City-held-out validation did not predict every row.")
    detail = calibration[
        ["Zip Code", "City", "County", "PERIOD_END", "y_true", group_column]
    ].copy()
    detail["y_pred_city_holdout"] = prediction
    return pd.DataFrame(rows), regression_metrics(y, prediction), detail


def validation_feature_importance(
    calibration: pd.DataFrame,
    spec: CandidateSpec,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Aggregate outer-fold permutation importance and standardized coefficients."""
    X = calibration[list(spec.feature_columns)]
    y = calibration["y_true"].to_numpy(dtype=float)
    outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    permutation_rows = []
    coefficient_rows = []
    for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y), start=1):
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=random_state + fold)
        search = GridSearchCV(
            spec.make_pipeline(),
            spec.param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            search.fit(X.iloc[train_index], y[train_index])
        importance = permutation_importance(
            search.best_estimator_,
            X.iloc[test_index],
            y[test_index],
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=random_state + fold,
            n_jobs=1,
        )
        for feature, mean_value, std_value in zip(
            spec.feature_columns,
            importance.importances_mean,
            importance.importances_std,
            strict=True,
        ):
            permutation_rows.append(
                {
                    "feature": feature,
                    "fold": fold,
                    "permutation_mae_increase": mean_value,
                    "permutation_repeat_std": std_value,
                }
            )
        model = search.best_estimator_.named_steps["model"]
        if hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_).ravel()
            if len(coefficients) == len(spec.feature_columns):
                for feature, coefficient in zip(
                    spec.feature_columns, coefficients, strict=True
                ):
                    coefficient_rows.append(
                        {
                            "feature": feature,
                            "fold": fold,
                            "standardized_coefficient": coefficient,
                        }
                    )

    permutation = pd.DataFrame(permutation_rows)
    summary = (
        permutation.groupby("feature", as_index=False)
        .agg(
            permutation_mae_increase_mean=("permutation_mae_increase", "mean"),
            permutation_mae_increase_std=("permutation_mae_increase", "std"),
        )
    )
    if coefficient_rows:
        coefficients = pd.DataFrame(coefficient_rows)
        coefficient_summary = (
            coefficients.groupby("feature", as_index=False)
            .agg(
                standardized_coefficient_mean=("standardized_coefficient", "mean"),
                standardized_coefficient_std=("standardized_coefficient", "std"),
                positive_coefficient_share=(
                    "standardized_coefficient",
                    lambda values: float((values > 0).mean()),
                ),
            )
        )
        summary = summary.merge(coefficient_summary, on="feature", how="left")
    summary["feature"] = summary["feature"].str.replace(
        r"^(RANK__|TRAILING__)", "", regex=True
    )
    return summary.sort_values(
        "permutation_mae_increase_mean", ascending=False
    ).reset_index(drop=True)


def sanity_checks(latest_scored: pd.DataFrame) -> pd.DataFrame:
    """Check score alignment with intuitive contemporaneous market signals."""
    price_drop_proxy = (-latest_scored["MEDIAN_LIST_PRICE_MOM"]).clip(lower=0)
    signals = [
        ("Sold above list share", latest_scored["SOLD_ABOVE_LIST"], 1),
        ("Average sale-to-list ratio", latest_scored["AVG_SALE_TO_LIST"], 1),
        ("Median days on market", latest_scored["MEDIAN_DOM"], -1),
        ("Inventory", latest_scored["INVENTORY"], -1),
        ("List-price drop proxy", price_drop_proxy, -1),
    ]
    rows = []
    for name, values, expected_sign in signals:
        valid = values.notna() & latest_scored["MarketScore_hat"].notna()
        correlation = spearmanr(
            values[valid], latest_scored.loc[valid, "MarketScore_hat"]
        ).statistic
        rows.append(
            {
                "signal": name,
                "rows": int(valid.sum()),
                "expected_direction": "positive" if expected_sign > 0 else "negative",
                "spearman_correlation": float(correlation),
                "direction_aligned": bool(np.sign(correlation) == expected_sign),
            }
        )
    return pd.DataFrame(rows)


def temporal_stability(
    all_scored: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Summarize consecutive and approximately year-over-year score changes."""
    ordered = all_scored.sort_values(["Zip Code", "PERIOD_END"]).copy()
    ordered["absolute_consecutive_change"] = ordered.groupby(
        "Zip Code", observed=True
    )["MarketScore_hat"].diff().abs()
    per_zip = (
        ordered.groupby("Zip Code", as_index=False, observed=True)
        .agg(
            observations=("MarketScore_hat", "size"),
            median_absolute_consecutive_change=(
                "absolute_consecutive_change",
                "median",
            ),
            p95_absolute_consecutive_change=(
                "absolute_consecutive_change",
                lambda values: values.quantile(0.95),
            ),
            max_absolute_consecutive_change=(
                "absolute_consecutive_change",
                "max",
            ),
        )
    )
    consecutive = ordered["absolute_consecutive_change"].dropna().to_numpy()

    year_changes: list[float] = []
    for _, group in ordered.groupby("Zip Code", observed=True):
        group = group.sort_values("PERIOD_END")
        dates = group["PERIOD_END"].tolist()
        scores = group["MarketScore_hat"].to_numpy(dtype=float)
        for index in range(1, len(group)):
            target_date = dates[index] - pd.DateOffset(years=1)
            prior_indices = np.arange(index)
            day_distance = np.array(
                [abs((dates[j] - target_date).days) for j in prior_indices]
            )
            nearest = int(day_distance.argmin())
            if day_distance[nearest] <= 62:
                year_changes.append(abs(scores[index] - scores[nearest]))

    year_array = np.asarray(year_changes, dtype=float)
    summary = {
        "consecutive_comparisons": int(len(consecutive)),
        "median_absolute_consecutive_change": float(np.median(consecutive)),
        "p95_absolute_consecutive_change": float(np.quantile(consecutive, 0.95)),
        "share_consecutive_change_over_10": float((consecutive > 10).mean()),
        "share_consecutive_change_over_15": float((consecutive > 15).mean()),
        "year_over_year_comparisons": int(len(year_array)),
        "median_absolute_year_over_year_change": float(np.median(year_array)),
        "p95_absolute_year_over_year_change": float(np.quantile(year_array, 0.95)),
        "share_year_over_year_change_over_15": float((year_array > 15).mean()),
    }
    return summary, per_zip


def distribution_shift_diagnostics(
    calibration: pd.DataFrame,
    latest: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Compare calibration ZIPs with other latest-2025 scoring ZIPs."""
    calibration_zips = set(calibration["Zip Code"].astype(int))
    target = latest[~latest["Zip Code"].astype(int).isin(calibration_zips)].copy()
    rows = []
    for feature in feature_columns:
        train_values = calibration[feature].dropna().to_numpy(dtype=float)
        target_values = target[feature].dropna().to_numpy(dtype=float)
        pooled_sd = np.sqrt(
            (np.var(train_values, ddof=1) + np.var(target_values, ddof=1)) / 2
        )
        standardized_difference = (
            (np.mean(target_values) - np.mean(train_values)) / pooled_sd
            if pooled_sd > 0
            else 0.0
        )
        ks_result = ks_2samp(train_values, target_values, alternative="two-sided")
        rows.append(
            {
                "feature": feature.replace("RANK__", "").replace("TRAILING__", ""),
                "calibration_mean": float(np.mean(train_values)),
                "target_mean": float(np.mean(target_values)),
                "standardized_mean_difference": float(standardized_difference),
                "absolute_standardized_mean_difference": float(
                    abs(standardized_difference)
                ),
                "ks_statistic": float(ks_result.statistic),
                "ks_p_value": float(ks_result.pvalue),
                "wasserstein_distance": float(
                    wasserstein_distance(train_values, target_values)
                ),
            }
        )
    feature_shift = pd.DataFrame(rows).sort_values(
        "absolute_standardized_mean_difference", ascending=False
    )

    combined = pd.concat(
        [
            calibration[list(feature_columns)].assign(domain=0),
            target[list(feature_columns)].assign(domain=1),
        ],
        ignore_index=True,
    )
    adversarial_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=5_000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    shift_cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=RANDOM_STATE
    )
    auc_scores = cross_val_score(
        adversarial_model,
        combined[list(feature_columns)],
        combined["domain"],
        scoring="roc_auc",
        cv=shift_cv,
        n_jobs=1,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(
        imputer.fit_transform(calibration[list(feature_columns)])
    )
    target_matrix = scaler.transform(imputer.transform(target[list(feature_columns)]))
    neighbors = NearestNeighbors(n_neighbors=2, n_jobs=1).fit(train_matrix)
    calibration_distances = neighbors.kneighbors(train_matrix)[0][:, 1]
    threshold = float(np.quantile(calibration_distances, 0.95))
    target_distances = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(
        train_matrix
    ).kneighbors(target_matrix)[0][:, 0]
    flags = target[["Zip Code", "City", "County", "PERIOD_END"]].copy()
    flags["nearest_calibration_distance"] = target_distances
    flags["ood_flag"] = target_distances > threshold
    flags = flags.sort_values("nearest_calibration_distance", ascending=False)

    summary = {
        "target_rows": int(len(target)),
        "adversarial_auc_mean": float(np.mean(auc_scores)),
        "adversarial_auc_std": float(np.std(auc_scores, ddof=1)),
        "calibration_distance_p95_threshold": threshold,
        "target_ood_share": float(flags["ood_flag"].mean()),
        "features_abs_smd_over_0_5": int(
            feature_shift["absolute_standardized_mean_difference"].gt(0.5).sum()
        ),
    }
    return summary, feature_shift.reset_index(drop=True), flags.reset_index(drop=True)
