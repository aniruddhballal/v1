# baseline.py
import json
import numpy as np
import pandas as pd
from pathlib import Path

def learn(df: pd.DataFrame, window: int = 30) -> dict:
    """
    Learn what 'normal' looks like for this dataset.
    Returns a baseline profile dict. Saves to JSON if save_path given.
    """
    numeric_cols = [c for c in df.attrs.get("numeric_cols", df.select_dtypes(include=[np.number]).columns.tolist()) if c in df.columns]
    is_timeseries = df.attrs.get("is_timeseries", False)
    profile = {"is_timeseries": is_timeseries, "window": window, "columns": {}}

    for col in numeric_cols:
        series = df[col].dropna()
        col_profile = {
            "mean": series.mean(),
            "std": series.std(),
            "median": series.median(),
            "p5": series.quantile(0.05),
            "p95": series.quantile(0.95),
            "q1": series.quantile(0.25),
            "q3": series.quantile(0.75),
        }
        col_profile["iqr"] = col_profile["q3"] - col_profile["q1"]

        if is_timeseries and len(series) >= window:
            rolling = series.rolling(window)
            col_profile["rolling_mean"] = rolling.mean().dropna().tolist()
            col_profile["rolling_std"] = rolling.std().dropna().tolist()

        profile["columns"][col] = col_profile

    return profile


def save(profile: dict, path: str):
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def load_profile(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summary(profile: dict):
    """Print a human-readable summary of the baseline."""
    print(f"Baseline profile ({'time-series' if profile['is_timeseries'] else 'tabular'})\n")
    for col, stats in profile["columns"].items():
        print(f"  {col}:")
        print(f"    mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"p5={stats['p5']:.4f}  p95={stats['p95']:.4f}  IQR={stats['iqr']:.4f}")