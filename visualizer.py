# visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for CLI


def plot(df: pd.DataFrame, explanations: list[dict], profile: dict,
         output_path: str = "output/anomalies.png", show: bool = False):
    """
    Plot each numeric column as a time-series or index plot,
    with anomalous rows marked as red dots.
    Saves to output_path as PNG.
    """
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    numeric_cols = [c for c in profile["columns"].keys() if c in df.columns]
    flagged_rows = {e["row"] for e in explanations}
    time_col = df.attrs.get("time_col", None)
    x = df[time_col] if time_col and time_col in df.columns else df.index

    n = len(numeric_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)
    fig.suptitle("Anomaly Detection Results", fontsize=14, fontweight="bold", y=1.01)

    for i, col in enumerate(numeric_cols):
        ax = axes[i][0]
        ax.plot(x, df[col], color="#4a90d9", linewidth=0.8, label=col)

        # Mark anomalies
        if flagged_rows:
            flagged_idx = [r for r in flagged_rows if r in df.index]
            ax.scatter(
                x.iloc[flagged_idx] if hasattr(x, "iloc") else [x[r] for r in flagged_idx],
                df[col].iloc[flagged_idx],
                color="red", zorder=5, s=40, label="anomaly"
            )

        # Draw mean line
        mean_val = profile["columns"][col]["mean"]
        ax.axhline(mean_val, color="gray", linestyle="--", linewidth=0.7, alpha=0.6, label="mean")

        ax.set_ylabel(col, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.xlabel("Time" if time_col else "Index", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path