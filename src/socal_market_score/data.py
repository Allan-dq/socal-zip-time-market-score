"""Data loading, validation, feature engineering, and coverage audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    RAW_FEATURES,
    REQUIRED_REFERENCE_COLUMNS,
    REQUIRED_SOCAL_COLUMNS,
    TRAILING_WINDOW,
)


def load_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the two required workbook sheets."""
    socal = pd.read_excel(path, sheet_name="SoCal", engine="openpyxl")
    reference = pd.read_excel(
        path, sheet_name="Market Score Reference", engine="openpyxl"
    )

    missing_socal = REQUIRED_SOCAL_COLUMNS - set(socal.columns)
    missing_reference = REQUIRED_REFERENCE_COLUMNS - set(reference.columns)
    if missing_socal:
        raise ValueError(f"SoCal sheet is missing columns: {sorted(missing_socal)}")
    if missing_reference:
        raise ValueError(
            f"Market Score Reference sheet is missing columns: {sorted(missing_reference)}"
        )

    for column in ["PERIOD_BEGIN", "PERIOD_END"]:
        socal[column] = pd.to_datetime(socal[column], errors="coerce")
    if socal[["PERIOD_BEGIN", "PERIOD_END"]].isna().any().any():
        raise ValueError("At least one source period could not be parsed as a date.")

    socal["Zip Code"] = pd.to_numeric(socal["Zip Code"], errors="coerce").astype(
        "Int64"
    )
    reference["Zip Code"] = pd.to_numeric(
        reference["Zip Code"], errors="coerce"
    ).astype("Int64")
    if socal["Zip Code"].isna().any() or reference["Zip Code"].isna().any():
        raise ValueError("At least one ZIP code could not be parsed.")

    for feature in RAW_FEATURES:
        socal[feature] = pd.to_numeric(socal[feature], errors="coerce")
    reference["Market Score raw"] = reference["Market Score"]
    reference["Market Score"] = pd.to_numeric(
        reference["Market Score"], errors="coerce"
    )

    if socal.duplicated(["Zip Code", "PERIOD_END"]).any():
        raise ValueError("Duplicate (Zip Code, PERIOD_END) rows were found.")
    if not socal["PERIOD_DURATION"].eq(90).all():
        raise ValueError("PERIOD_DURATION contains values other than 90 days.")

    return socal, reference


def engineer_features(socal: pd.DataFrame) -> pd.DataFrame:
    """Create time-comparable percentile features and trailing-year medians.

    Percentile ranks are computed cross-sectionally within each PERIOD_END. This
    makes historical and current observations comparable despite market-wide
    nominal level changes. The trailing features use only current and prior
    observations for the same ZIP.
    """
    engineered = socal.sort_values(["Zip Code", "PERIOD_END"]).reset_index(drop=True)
    rank_columns = [f"RANK__{feature}" for feature in RAW_FEATURES]
    engineered[rank_columns] = engineered.groupby(
        "PERIOD_END", observed=True
    )[RAW_FEATURES].rank(pct=True, method="average")

    trailing_columns = [f"TRAILING__{feature}" for feature in RAW_FEATURES]
    engineered[trailing_columns] = np.nan
    for _, group in engineered.groupby("Zip Code", observed=True, sort=False):
        ranked = group.set_index("PERIOD_END")[rank_columns]
        trailing = ranked.rolling(TRAILING_WINDOW, min_periods=1).median().to_numpy()
        engineered.loc[group.index, trailing_columns] = trailing

    return engineered


def latest_2025_rows(socal: pd.DataFrame) -> pd.DataFrame:
    """Return each ZIP's latest available record within calendar year 2025."""
    rows_2025 = socal[socal["PERIOD_END"].dt.year.eq(2025)].copy()
    return (
        rows_2025.sort_values(["Zip Code", "PERIOD_END"])
        .drop_duplicates("Zip Code", keep="last")
        .reset_index(drop=True)
    )


def build_calibration_set(
    latest: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Join valid numeric reference scores to latest-2025 feature rows."""
    reference_fields = reference[
        ["Zip Code", "City", "County", "Market Score"]
    ].rename(
        columns={
            "City": "Reference City",
            "County": "Reference County",
            "Market Score": "y_true",
        }
    )
    calibration = latest.merge(reference_fields, on="Zip Code", how="inner")
    calibration = calibration[calibration["y_true"].notna()].copy()
    calibration["y_true"] = calibration["y_true"].astype(float)
    return calibration.reset_index(drop=True)


def build_reference_exclusions(
    socal: pd.DataFrame, latest: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Document why reference rows were not used for calibration."""
    latest_zips = set(latest["Zip Code"].astype(int))
    rows: list[dict[str, Any]] = []
    for _, record in reference.iterrows():
        zip_code = int(record["Zip Code"])
        city = record["City"]
        county = record["County"]
        score = record["Market Score"]
        raw_score = record["Market Score raw"]
        reasons = []
        if pd.isna(score):
            reasons.append("Reference score is not numeric")
        if zip_code not in latest_zips:
            reasons.append("No 2025 market record")
        if reasons:
            rows.append(
                {
                    "Zip Code": zip_code,
                    "City": city,
                    "County": county,
                    "Market Score raw": raw_score,
                    "Exclusion reason": "; ".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def data_audit(
    socal: pd.DataFrame,
    reference: pd.DataFrame,
    latest: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict[str, Any]:
    """Return compact counts used in the report and tests."""
    zips_with_2025 = set(latest["Zip Code"].astype(int))
    all_zips = set(socal["Zip Code"].astype(int))
    return {
        "source_rows": int(len(socal)),
        "source_columns": int(socal.shape[1]),
        "unique_zips": int(socal["Zip Code"].nunique()),
        "period_start": socal["PERIOD_END"].min().date().isoformat(),
        "period_end": socal["PERIOD_END"].max().date().isoformat(),
        "unique_period_end_dates": int(socal["PERIOD_END"].nunique()),
        "reference_rows": int(len(reference)),
        "numeric_reference_scores": int(reference["Market Score"].notna().sum()),
        "latest_2025_rows": int(len(latest)),
        "calibration_rows": int(len(calibration)),
        "zips_without_2025": int(len(all_zips - zips_with_2025)),
        "duplicate_zip_period_rows": int(
            socal.duplicated(["Zip Code", "PERIOD_END"]).sum()
        ),
        "invalid_period_duration_rows": int((socal["PERIOD_DURATION"] != 90).sum()),
        "feature_missing_rate_all": float(socal[RAW_FEATURES].isna().mean().mean()),
        "feature_missing_rate_calibration": float(
            calibration[RAW_FEATURES].isna().mean().mean()
        ),
    }
