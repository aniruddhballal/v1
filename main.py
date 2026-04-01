# main.py
import argparse
import json
import os
import sys

import pandas as pd

from loader import load
from baseline import learn, save as save_profile, load_profile
from detector import run as detect
from explainer import run as explain, summary
from visualizer import plot


def parse_rules(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("rules", {})


def main():
    parser = argparse.ArgumentParser(
        description="v1 Anomaly Detection Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/finance/transactions.csv
  python main.py --input data/finance/transactions.csv --output output/ --contamination 0.02
  python main.py --input data.csv --baseline-path saved_baseline.json
        """
    )
    parser.add_argument("--input",           required=True,  help="Path to CSV or JSON input file")
    parser.add_argument("--output",          default="output", help="Output directory (default: output/)")
    parser.add_argument("--baseline-path",   default=None,   help="Load existing baseline JSON instead of recomputing")
    parser.add_argument("--save-baseline",   default=None,   help="Save computed baseline to this path")
    parser.add_argument("--config",          default="config.yaml", help="Path to config.yaml with rules")
    parser.add_argument("--contamination",   type=float, default=0.01, help="Isolation Forest contamination (default: 0.01)")
    parser.add_argument("--z-threshold",     type=float, default=3.0,  help="Z-score threshold (default: 3.0)")
    parser.add_argument("--min-score",       type=float, default=0.25, help="Min anomaly score to report (default: 0.25)")
    parser.add_argument("--no-plot",         action="store_true",      help="Skip chart generation")
    parser.add_argument("--exclude-cols", default="", help="Comma-separated columns to drop before detection (e.g. Time,Class)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    print(f"[1/5] Loading data from {args.input}")
    df = load(args.input)
    print(f"      {len(df)} rows × {len(df.columns)} columns | "
          f"time-series={df.attrs.get('is_timeseries', False)}")
    if args.exclude_cols:
        drop = [c.strip() for c in args.exclude_cols.split(",") if c.strip() in df.columns]
        df = df.drop(columns=drop)
        print(f"      Dropped columns: {drop}")

    print("[2/5] Building baseline")
    if args.baseline_path and os.path.exists(args.baseline_path):
        profile = load_profile(args.baseline_path)
        print(f"      Loaded from {args.baseline_path}")
    else:
        profile = learn(df)
        if args.save_baseline:
            save_profile(profile, args.save_baseline)
            print(f"      Saved to {args.save_baseline}")

    print("[3/5] Running detection")
    rules = parse_rules(args.config)
    result = detect(df, profile, rules=rules,
                    z_threshold=args.z_threshold,
                    contamination=args.contamination)
    n_flagged = (result["anomaly_score"] >= args.min_score).sum()
    print(f"      {n_flagged} rows flagged (score ≥ {args.min_score})")

    print("[4/5] Generating explanations")
    explanations = explain(result, profile, df, min_score=args.min_score, limit=500)
    print(summary(explanations))

    json_path = os.path.join(args.output, f"{stem}_anomalies.json")
    with open(json_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"\n      Saved → {json_path}")

    if not args.no_plot:
        print("[5/5] Generating chart")
        chart_path = os.path.join(args.output, f"{stem}_anomalies.png")
        plot(df, explanations, profile, output_path=chart_path)
        print(f"      Saved → {chart_path}")
    else:
        print("[5/5] Skipping chart (--no-plot)")

    print("\nDone.")


if __name__ == "__main__":
    main()