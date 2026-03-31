# detector.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest


def _zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    z = np.abs(stats.zscore(series.dropna()))
    flags = pd.Series(False, index=series.index)
    flags.iloc[series.dropna().index] = z > threshold
    return flags


def _iqr(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def _isolation_forest(df: pd.DataFrame, numeric_cols: list, contamination: float = 0.01) -> pd.Series:
    model = IsolationForest(contamination=contamination, random_state=42)
    scores = model.fit_predict(df[numeric_cols].fillna(0))
    return pd.Series(scores == -1, index=df.index)


def _rules(df: pd.DataFrame, rules: dict) -> pd.Series:
    """
    rules = {"column_name": {"gt": 10000}, "other_col": {"lt": 0}}
    Supported operators: gt, lt, gte, lte, eq
    """
    flags = pd.Series(False, index=df.index)
    ops = {"gt": lambda s, v: s > v, "lt": lambda s, v: s < v,
           "gte": lambda s, v: s >= v, "lte": lambda s, v: s <= v,
           "eq": lambda s, v: s == v}
    for col, conditions in rules.items():
        if col not in df.columns:
            continue
        for op, val in conditions.items():
            if op in ops:
                flags |= ops[op](df[col], val)
    return flags


def run(df: pd.DataFrame, profile: dict, rules: dict = None,
        z_threshold: float = 3.0, contamination: float = 0.01) -> pd.DataFrame:
    """
    Run all three detection layers. Returns df with added columns:
      flag_zscore, flag_iqr, flag_isoforest, flag_rules, methods_flagged, anomaly_score
    """
    numeric_cols = list(profile["columns"].keys())
    # keep only cols that exist in df
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    results = df.copy()

    # Layer 1 — statistical
    zscore_flags = pd.Series(False, index=df.index)
    iqr_flags = pd.Series(False, index=df.index)
    for col in numeric_cols:
        zscore_flags |= _zscore(df[col], z_threshold)
        iqr_flags |= _iqr(df[col])

    results["flag_zscore"] = zscore_flags
    results["flag_iqr"] = iqr_flags

    # Layer 2 — Isolation Forest
    results["flag_isoforest"] = _isolation_forest(df, numeric_cols, contamination)

    # Layer 3 — rules
    results["flag_rules"] = _rules(df, rules or {})

    # Ensemble: count how many methods fired per row
    flag_cols = ["flag_zscore", "flag_iqr", "flag_isoforest", "flag_rules"]
    results["methods_flagged"] = results[flag_cols].sum(axis=1)

    # Anomaly score: 0.0–1.0 based on methods fired (max 4)
    results["anomaly_score"] = (results["methods_flagged"] / len(flag_cols)).round(4)

    return results