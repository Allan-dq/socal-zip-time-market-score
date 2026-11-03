"""Data-contract tests for generated outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "socal-market-score-matplotlib")
)
sys.path.insert(0, str(ROOT / "src"))

from socal_market_score.data import engineer_features, latest_2025_rows

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "SoCal - Real Estate Market Metrics - ZIP×Time Score Series.xlsx"
)
OUTPUTS = ROOT / "outputs"


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = pd.read_excel(INPUT, sheet_name="SoCal", engine="openpyxl")
        cls.source["PERIOD_END"] = pd.to_datetime(cls.source["PERIOD_END"])
        cls.evaluation = pd.read_csv(OUTPUTS / "scores_latest_select_zips_eval.csv")
        cls.latest = pd.read_csv(OUTPUTS / "scores_latest_all_zips.csv")
        cls.all_periods = pd.read_csv(OUTPUTS / "scores_all_periods.csv")
        cls.metrics = json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8"))

    def test_required_schemas(self) -> None:
        self.assertEqual(
            list(self.evaluation.columns),
            ["Zip Code", "City", "County", "PERIOD", "y_true", "y_pred"],
        )
        self.assertEqual(
            list(self.latest.columns),
            ["Zip Code", "City", "County", "PERIOD", "MarketScore_hat"],
        )
        self.assertEqual(
            list(self.all_periods.columns),
            ["Zip Code", "City", "County", "PERIOD", "MarketScore_hat"],
        )

    def test_all_existing_rows_are_scored(self) -> None:
        self.assertEqual(len(self.all_periods), len(self.source))
        self.assertFalse(self.all_periods["MarketScore_hat"].isna().any())

    def test_latest_2025_rule(self) -> None:
        source_2025 = self.source[self.source["PERIOD_END"].dt.year.eq(2025)]
        expected = source_2025.groupby("Zip Code")["PERIOD_END"].max()
        actual = self.latest.set_index("Zip Code")["PERIOD"].map(pd.Timestamp)
        self.assertEqual(len(actual), len(expected))
        pd.testing.assert_series_equal(
            actual.sort_index(), expected.sort_index(), check_names=False
        )

    def test_score_bounds(self) -> None:
        for frame, column in [
            (self.evaluation, "y_pred"),
            (self.latest, "MarketScore_hat"),
            (self.all_periods, "MarketScore_hat"),
        ]:
            self.assertTrue(frame[column].between(0, 100).all())

    def test_evaluation_is_out_of_fold_and_complete(self) -> None:
        self.assertFalse(self.evaluation[["y_true", "y_pred"]].isna().any().any())
        self.assertTrue(self.evaluation["Zip Code"].is_unique)
        self.assertGreater(self.metrics["oof_metrics"]["r2"], 0)
        self.assertNotIn("Dummy median", self.metrics["selected_model"])

    def test_model_bundle_reproduces_latest_scores(self) -> None:
        bundle = joblib.load(OUTPUTS / "market_score_model.joblib")
        engineered = engineer_features(self.source)
        latest = latest_2025_rows(engineered)
        prediction = np.clip(
            bundle["pipeline"].predict(latest[bundle["feature_columns"]]), 0, 100
        )
        reproduced = pd.Series(
            np.round(prediction, 2), index=latest["Zip Code"].astype(int)
        ).sort_index()
        delivered = self.latest.set_index("Zip Code")["MarketScore_hat"].sort_index()
        np.testing.assert_allclose(reproduced.to_numpy(), delivered.to_numpy())


if __name__ == "__main__":
    unittest.main()
