# explainer.py
import pandas as pd
import numpy as np


def explain_row(row: pd.Series, profile: dict, original_df: pd.DataFrame) -> dict:
    """
    Generate a human-readable explanation for a single flagged row.
    Returns a dict with row index, flagged columns, explanation string, and confidence.
    """
    idx = row.name
    explanations = []
    numeric_cols = list(profile["columns"].keys())

    # Z-score explanation
    if row.get("flag_zscore"):
        for col in numeric_cols:
            if col not in original_df.columns:
                continue
            col_mean = profile["columns"][col]["mean"]
            col_std = profile["columns"][col]["std"]
            if col_std == 0:
                continue
            val = original_df.loc[idx, col]
            z = abs(val - col_mean) / col_std
            if z > 3.0:
                direction = "above" if val > col_mean else "below"
                explanations.append(
                    f"{col} is {z:.1f}σ {direction} mean "
                    f"(mean={col_mean:.4f}, value={val:.4f})"
                )

    # IQR explanation
    if row.get("flag_iqr"):
        for col in numeric_cols:
            if col not in original_df.columns:
                continue
            q1 = profile["columns"][col]["q1"]
            q3 = profile["columns"][col]["q3"]
            iqr = profile["columns"][col]["iqr"]
            val = original_df.loc[idx, col]
            if val < q1 - 1.5 * iqr or val > q3 + 1.5 * iqr:
                bound = "below Q1-1.5×IQR" if val < q1 else "above Q3+1.5×IQR"
                explanations.append(f"{col}={val:.4f} is {bound} (Q1={q1:.4f}, Q3={q3:.4f})")

    # Isolation Forest explanation
    if row.get("flag_isoforest"):
        score = row.get("anomaly_score", 0)
        explanations.append(
            f"Isolation Forest flagged this row "
            f"(ensemble score={score:.2f})"
        )

    # Rule-based explanation
    if row.get("flag_rules"):
        explanations.append("Triggered a user-defined threshold rule")

    explanation_str = "; ".join(explanations) if explanations else "Flagged by ensemble (no single dominant reason)"

    return {
        "row": int(idx),
        "anomaly_score": float(row.get("anomaly_score", 0)),
        "methods_flagged": int(row.get("methods_flagged", 0)),
        "explanation": explanation_str,
        "flags": {
            "zscore": bool(row.get("flag_zscore", False)),
            "iqr": bool(row.get("flag_iqr", False)),
            "isolation_forest": bool(row.get("flag_isoforest", False)),
            "rules": bool(row.get("flag_rules", False)),
        }
    }


def run(detection_result: pd.DataFrame, profile: dict,
        original_df: pd.DataFrame, min_score: float = 0.25, limit: int = 500) -> list[dict]:
    flagged = detection_result[detection_result["anomaly_score"] >= min_score]
    flagged = flagged.nlargest(limit, "anomaly_score")
    explanations = [explain_row(row, profile, original_df) for _, row in flagged.iterrows()]
    return sorted(explanations, key=lambda x: x["anomaly_score"], reverse=True)


def summary(explanations: list[dict]) -> str:
    """Print a readable summary to stdout."""
    if not explanations:
        return "No anomalies detected above threshold."
    lines = [f"{'='*60}", f"  {len(explanations)} anomalies detected", f"{'='*60}"]
    for e in explanations:
        lines.append(f"\nRow {e['row']}  score={e['anomaly_score']:.2f}  methods={e['methods_flagged']}")
        lines.append(f"  → {e['explanation']}")
    return "\n".join(lines)