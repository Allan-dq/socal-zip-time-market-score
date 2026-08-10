"""Project-wide configuration and feature definitions."""

from __future__ import annotations

RANDOM_STATE = 42
SCORE_MIN = 0.0
SCORE_MAX = 100.0
TRAILING_WINDOW = "365D"

RAW_FEATURES = [
    "MEDIAN_SALE_PRICE",
    "MEDIAN_SALE_PRICE_MOM",
    "MEDIAN_SALE_PRICE_YOY",
    "MEDIAN_LIST_PRICE",
    "MEDIAN_LIST_PRICE_MOM",
    "MEDIAN_LIST_PRICE_YOY",
    "MEDIAN_PPSF",
    "MEDIAN_PPSF_MOM",
    "MEDIAN_PPSF_YOY",
    "MEDIAN_LIST_PPSF",
    "MEDIAN_LIST_PPSF_MOM",
    "MEDIAN_LIST_PPSF_YOY",
    "HOMES_SOLD",
    "HOMES_SOLD_MOM",
    "HOMES_SOLD_YOY",
    "PENDING_SALES",
    "PENDING_SALES_MOM",
    "PENDING_SALES_YOY",
    "NEW_LISTINGS",
    "NEW_LISTINGS_MOM",
    "NEW_LISTINGS_YOY",
    "INVENTORY",
    "INVENTORY_MOM",
    "INVENTORY_YOY",
    "MEDIAN_DOM",
    "MEDIAN_DOM_MOM",
    "MEDIAN_DOM_YOY",
    "AVG_SALE_TO_LIST",
    "AVG_SALE_TO_LIST_MOM",
    "AVG_SALE_TO_LIST_YOY",
    "SOLD_ABOVE_LIST",
    "SOLD_ABOVE_LIST_MOM",
    "SOLD_ABOVE_LIST_YOY",
    "OFF_MARKET_IN_TWO_WEEKS",
    "OFF_MARKET_IN_TWO_WEEKS_MOM",
    "OFF_MARKET_IN_TWO_WEEKS_YOY",
]

LEVEL_FEATURES = [
    feature for feature in RAW_FEATURES if not feature.endswith(("_MOM", "_YOY"))
]
LEVEL_AND_YOY_FEATURES = [
    feature for feature in RAW_FEATURES if not feature.endswith("_MOM")
]

CURRENT_LEVEL_FEATURES = [f"RANK__{feature}" for feature in LEVEL_FEATURES]
CURRENT_LEVEL_YOY_FEATURES = [
    f"RANK__{feature}" for feature in LEVEL_AND_YOY_FEATURES
]
CURRENT_ALL_FEATURES = [f"RANK__{feature}" for feature in RAW_FEATURES]
TRAILING_LEVEL_FEATURES = [f"TRAILING__{feature}" for feature in LEVEL_FEATURES]
TRAILING_LEVEL_YOY_FEATURES = [
    f"TRAILING__{feature}" for feature in LEVEL_AND_YOY_FEATURES
]
TRAILING_ALL_FEATURES = [f"TRAILING__{feature}" for feature in RAW_FEATURES]

REQUIRED_SOCAL_COLUMNS = {
    "PERIOD_BEGIN",
    "PERIOD_END",
    "PERIOD_DURATION",
    "Zip Code",
    "City",
    "County",
    *RAW_FEATURES,
}
REQUIRED_REFERENCE_COLUMNS = {"Zip Code", "City", "County", "Market Score"}

RESEARCH_SOURCES = {
    "ridge": "https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification",
    "elastic_net": "https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00503.x",
    "random_forest": "https://doi.org/10.1023/A:1010933404324",
    "gradient_boosting": "https://doi.org/10.1214/aos/1013203451",
    "svr": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
    "pls": "https://scikit-learn.org/stable/modules/cross_decomposition.html",
    "huber": "https://scikit-learn.org/stable/modules/linear_model.html#huber-regression",
    "nested_cv": "https://pubmed.ncbi.nlm.nih.gov/16504092/",
    "leakage": "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
    "permutation_importance": "https://scikit-learn.org/stable/modules/permutation_importance.html",
    "covariate_shift": "https://www.jmlr.org/papers/v8/sugiyama07a.html",
    "ks_test": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html",
}
