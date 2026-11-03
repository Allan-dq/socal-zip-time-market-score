# SoCal ZIP-by-Time Market Score Model

## Executive summary

This project builds a single, bounded, explainable market score for every available Southern California ZIP-period record. It covers the latest-2025 calibration and the extended analyses: full-period scoring, a time-series example, feature interpretation, geographic shift diagnostics, and output validation.

The usable calibration sample contains **83 ZIPs**. That is small relative to the correlated real-estate feature set, so model choice was based on repeated nested cross-validation rather than training fit. The selected specification is **Ridge | current levels**, chosen with the one-standard-error rule. Its repeated nested-CV out-of-fold MAE is **7.17 points** (bootstrap 95% interval: **5.81 to 8.63**), RMSE is **9.75**, R-squared is **0.649**, and Spearman correlation is **0.826**.

## 1. Data scope and eligibility

- The SoCal sheet contains 12,445 rows, 764 ZIPs, and 166 distinct period-end dates from 2012-03-31 through 2025-12-31.
- The reference sheet contains 143 rows, of which 120 have numeric scores.
- By definition, each ZIP's latest record is the maximum `PERIOD_END` within calendar year 2025. This yields 524 latest-scoring rows.
- After excluding non-numeric reference scores and reference ZIPs with no 2025 record, 83 rows remain for supervised calibration.
- 240 source ZIPs have no 2025 record and are excluded from latest-period output, but their existing historical rows remain in the all-period file.
- Duplicate `(Zip Code, PERIOD_END)` rows: 0. Invalid period durations: 0.

No external market data was used.

## 2. Feature engineering and leakage control

The model uses the 12 contemporaneous level features: sale and list price, sale and list price per square foot, homes sold, pending sales, new listings, inventory, median days on market, average sale-to-list ratio, share sold above list, and share off market within two weeks. Short-horizon MoM and YoY changes were evaluated but were less reliable in cross-validation and increased temporal volatility.

Each feature is converted to a cross-sectional percentile rank within its exact `PERIOD_END`. This removes market-wide nominal drift, makes 2012 and 2025 values comparable, reduces outlier sensitivity, and keeps the score interpretation relative to the contemporaneous SoCal market. A trailing-365-day median representation was also evaluated explicitly.

Median imputation and standardization are fitted inside each cross-validation training fold through a scikit-learn `Pipeline`; held-out rows never influence fitted preprocessing. Final predictions are clipped to `[0, 100]`.

The fixed-Ridge representation screen below isolates the effect of adding volatile change features and trailing smoothing before model-family tuning:

| representation | feature_count | mae_mean | rmse_mean | r2_mean |
| --- | --- | --- | --- | --- |
| Current levels | 12 | 7.172 | 9.479 | 0.621 |
| Trailing-year levels | 12 | 7.296 | 9.636 | 0.604 |
| Trailing-year levels + MoM + YoY | 36 | 7.810 | 10.090 | 0.560 |
| Trailing-year levels + YoY | 24 | 7.876 | 10.238 | 0.547 |
| Current levels + YoY | 24 | 7.998 | 10.901 | 0.501 |
| Current levels + MoM + YoY | 36 | 8.619 | 11.312 | 0.463 |

## 3. Model research and selection

The search covered a median baseline, Ridge, Elastic Net, Huber regression, partial least squares, RBF support-vector regression, random forest, extremely randomized trees, and histogram gradient boosting. The design reflects the following evidence:

- [Ridge](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) shrinks coefficients and is robust to correlated predictors.
- [Elastic Net](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00503.x) combines L1 and L2 penalties and encourages grouped selection among correlated features.
- [Huber regression](https://scikit-learn.org/stable/modules/linear_model.html#huber-regression) reduces the influence of large residuals in small samples.
- [PLS regression](https://scikit-learn.org/stable/modules/cross_decomposition.html) is a supervised latent-component method for correlated predictors.
- [Random forests](https://doi.org/10.1023/A:1010933404324) and [gradient boosting](https://doi.org/10.1214/aos/1013203451) provide nonlinear alternatives.
- [RBF SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) provides a regularized nonlinear kernel benchmark suited to small samples.
- Nested CV is used because evaluating the same CV run used for tuning can be optimistically biased; the [Varma and Simon study](https://pubmed.ncbi.nlm.nih.gov/16504092/) shows that tuning must be repeated inside each outer fold.
- The preprocessing design follows scikit-learn's [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Five repeats of five-fold outer CV were used, with four-fold inner tuning by MAE. The one-standard-error rule selects the least complex model statistically close to the best mean outer-fold MAE.

| candidate | outer_mae_mean | outer_mae_se | oof_mae | oof_rmse | oof_r2 | selected |
| --- | --- | --- | --- | --- | --- | --- |
| RBF SVR \| current levels | 7.085 | 0.290 | 6.703 | 9.534 | 0.664 |  |
| Elastic net \| current levels | 7.292 | 0.253 | 7.151 | 9.715 | 0.651 |  |
| Ridge \| current levels | 7.294 | 0.248 | 7.173 | 9.753 | 0.649 | Yes |
| PLS \| current levels | 7.313 | 0.224 | 7.209 | 9.410 | 0.673 |  |
| Huber \| current levels | 7.322 | 0.233 | 7.190 | 9.640 | 0.657 |  |
| Extra trees \| current levels | 7.339 | 0.276 | 7.267 | 9.621 | 0.658 |  |
| Huber \| trailing-year levels | 7.367 | 0.236 | 7.134 | 9.823 | 0.644 |  |
| Ridge \| trailing-year levels | 7.466 | 0.238 | 7.260 | 10.000 | 0.631 |  |
| Random forest \| current levels | 7.749 | 0.306 | 7.647 | 9.958 | 0.634 |  |
| Histogram gradient boosting \| current levels | 8.094 | 0.354 | 7.584 | 10.088 | 0.624 |  |
| Dummy median \| current levels | 13.197 | 0.532 | 13.204 | 17.235 | -0.097 |  |

Final full-sample tuning parameters: `{'model__alpha': 10.0}`.

## 4. Reference-score validation

The evaluation file contains repeated nested-CV predictions, not in-sample fitted values. This is the most defensible estimate available without a separate labeled test set.

- OOF MAE: 7.173
- OOF RMSE: 9.753
- OOF R-squared: 0.649
- OOF Spearman correlation: 0.826
- Fitted full-sample MAE (descriptive only): 6.326

### Geographic stress test

Holding out each reference city in turn is deliberately harder than random CV. Aggregate city-held-out MAE is 9.073, RMSE is 12.392, and R-squared is 0.433. These results quantify the risk of transporting labels from a small set of representative cities.

| held_out_city | rows | mae | rmse | r2 | spearman |
| --- | --- | --- | --- | --- | --- |
| Riverside | 7 | 4.934 | 5.808 | -1.328 | 0.775 |
| Irvine | 3 | 5.369 | 5.791 | -0.188 | 1.000 |
| Santa Ana | 4 | 6.030 | 6.769 | 0.015 | 0.000 |
| San Diego | 23 | 7.243 | 9.732 | -0.661 | 0.508 |
| San Bernardino | 4 | 8.818 | 11.634 | -2.247 | 1.000 |
| Los Angeles | 42 | 11.343 | 15.044 | 0.277 | 0.765 |

## 5. Explainability

Permutation importance is measured on outer validation folds, so it reflects the increase in held-out MAE after shuffling a feature. Standardized coefficients show the direction and approximate score contribution of a one-standard-deviation percentile-rank movement. Correlated features can share importance, so individual values should not be interpreted causally. See scikit-learn's [permutation-importance guidance](https://scikit-learn.org/stable/modules/permutation_importance.html).

| feature | permutation_mae_increase_mean | standardized_coefficient_mean | positive_coefficient_share |
| --- | --- | --- | --- |
| MEDIAN_DOM | 2.745 | -5.693 | 0.000 |
| INVENTORY | 1.872 | -4.178 | 0.000 |
| AVG_SALE_TO_LIST | 0.727 | 2.772 | 1.000 |
| OFF_MARKET_IN_TWO_WEEKS | 0.670 | 2.871 | 1.000 |
| SOLD_ABOVE_LIST | 0.590 | 2.217 | 1.000 |
| MEDIAN_SALE_PRICE | 0.499 | -1.683 | 0.040 |
| PENDING_SALES | 0.467 | 2.394 | 1.000 |
| HOMES_SOLD | 0.393 | 1.936 | 0.960 |
| NEW_LISTINGS | 0.262 | 1.772 | 1.000 |
| MEDIAN_LIST_PPSF | 0.147 | 0.132 | 0.440 |

## 6. Latest-period sanity checks

The following Spearman correlations test whether scores co-move with recognizable market signals. They are diagnostics, not constraints; inventory is especially ambiguous because it can proxy both supply pressure and market size.

| signal | expected_direction | spearman_correlation | direction_aligned |
| --- | --- | --- | --- |
| Sold above list share | positive | 0.714 | True |
| Average sale-to-list ratio | positive | 0.754 | True |
| Median days on market | negative | -0.776 | True |
| Inventory | negative | -0.060 | True |
| List-price drop proxy | negative | -0.047 | True |

## 7. Full-period scoring and temporal stability

All 12,445 existing ZIP-period rows are scored; missing time points are not created. The example trend uses ZIP 92612.

- Median absolute consecutive change: 5.16 points
- 95th percentile absolute consecutive change: 19.15 points
- Share of consecutive changes above 15 points: 10.26%
- Median absolute approximately year-over-year change: 5.77 points
- 95th percentile absolute approximately year-over-year change: 19.99 points
- Share of approximately year-over-year changes above 15 points: 11.86%

The percentile-rank representation is the main protection against nominal price drift. The score remains descriptive and contemporaneous; it is not a forecast.

## 8. Distribution-shift bonus

The selected reference ZIPs are compared with the other latest-2025 scoring ZIPs using per-feature standardized mean differences, two-sample Kolmogorov-Smirnov statistics, Wasserstein distance, and adversarial validation. The [KS test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) compares empirical distributions. The adversarial classifier's repeated-CV AUC is **0.759 ± 0.063**; 0.5 would indicate no detectable separation. 2 selected features have an absolute standardized mean difference above 0.5, and 12.5% of non-calibration latest ZIPs exceed the calibration sample's 95th-percentile nearest-neighbor distance threshold.

| feature | standardized_mean_difference | ks_statistic | ks_p_value | wasserstein_distance |
| --- | --- | --- | --- | --- |
| MEDIAN_PPSF | -0.656 | 0.317 | 0.000 | 0.179 |
| MEDIAN_LIST_PPSF | -0.586 | 0.302 | 0.000 | 0.162 |
| MEDIAN_SALE_PRICE | -0.330 | 0.166 | 0.037 | 0.095 |
| INVENTORY | -0.308 | 0.179 | 0.020 | 0.090 |
| OFF_MARKET_IN_TWO_WEEKS | 0.308 | 0.174 | 0.025 | 0.087 |
| NEW_LISTINGS | -0.243 | 0.123 | 0.224 | 0.071 |
| MEDIAN_LIST_PRICE | -0.224 | 0.151 | 0.075 | 0.067 |
| AVG_SALE_TO_LIST | 0.198 | 0.162 | 0.045 | 0.059 |

This is evidence of possible covariate shift, not proof that the score relationship changes. Under covariate shift, ordinary CV can misrepresent deployment risk; see [Sugiyama et al.](https://www.jmlr.org/papers/v8/sugiyama07a.html). Recommended production actions are to collect labels from underrepresented counties, monitor the same diagnostics by refresh, flag distant ZIPs for review, and evaluate importance weighting or regional recalibration only after obtaining a defensible target-domain validation set.

## 9. Outputs

- `outputs/scores_latest_select_zips_eval.csv`: valid reference ZIPs with honest OOF predictions.
- `outputs/scores_latest_all_zips.csv`: all ZIPs with a 2025 record, using each ZIP's latest 2025 row.
- `outputs/scores_all_periods.csv`: every existing source ZIP-period row.
- `outputs/model_comparison.csv`, `outputs/feature_importance.csv`, `outputs/sanity_checks.csv`, `outputs/feature_shift_diagnostics.csv`, and supporting validation diagnostics.
- `figures/`: model comparison, validation, top/bottom ZIPs, example trend, feature importance, sanity checks, stability, and distribution-shift plots.
- `outputs/market_score_model.joblib`: fitted preprocessing pipeline, selected features, and metadata.

## 10. Assumptions and limitations

1. Only 83 usable labels are available, concentrated in six cities. CV uncertainty and geographic transport risk remain material.
2. The reference score is treated as the supervised target; its construction and label uncertainty are unknown.
3. Percentile ranks are batch-relative. Scoring a new period requires a contemporaneous cohort of ZIP feature rows.
4. The model is associative, not causal. Coefficients and permutation importance do not establish that changing a market metric will change the true underlying market quality.
5. Clipping guarantees the 0-100 bounds but can compress extreme predictions.
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
