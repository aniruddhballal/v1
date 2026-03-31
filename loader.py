# loader.py
import pandas as pd
import numpy as np
from pathlib import Path

def load(filepath: str, drop_threshold: float = 0.5, normalize: bool = True) -> pd.DataFrame:
    """
    Load CSV or JSON, handle missing values, normalize numeric columns.
    Returns a clean DataFrame with metadata in df.attrs.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No file at {filepath}")

    if path.suffix == ".csv":
        df = pd.read_csv(filepath)
    elif path.suffix == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .csv or .json")

    # Drop columns that are mostly empty
    df = df.dropna(thresh=int(len(df) * (1 - drop_threshold)), axis=1)

    # Fill remaining missing values: median for numeric, mode for categorical
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

    # Detect if time-series (has a parseable datetime column)
    is_timeseries = False
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower() or "timestamp" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
                is_timeseries = True
                df.attrs["time_col"] = col
                break
            except Exception:
                pass

    # Normalize numeric columns to [0, 1]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if normalize and numeric_cols:
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / \
                           (df[numeric_cols].max() - df[numeric_cols].min() + 1e-8)

    df.attrs["is_timeseries"] = is_timeseries
    df.attrs["numeric_cols"] = numeric_cols
    df.attrs["source"] = str(path)

    return df