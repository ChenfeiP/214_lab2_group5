#!/usr/bin/env python3
"""
Compare baseline vs modified logistic regression experiments.

Expected input (per directory):
  - logistic_raw_results.csv
  - logistic_latent_results.csv
  - logistic_raw_plus_latent_results.csv

Each CSV is produced by logistic_experiments.py and typically contains:
  - test_image
  - auc
  - balanced_accuracy
  - f1
  - (plus confusion-matrix fields)
"""

import argparse
import os
from typing import Dict, Optional

import pandas as pd

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LR_RESULTS_BASELINE = os.path.join(
    _CODE, "results", "part3_logistic_regression", "results_baseline"
)
LR_RESULTS_MODIFIED = os.path.join(
    _CODE, "results", "part3_logistic_regression", "results_modified"
)
LR_COMPARISON_SUMMARY_CSV = os.path.join(
    _CODE, "results", "part3_logistic_regression", "logistic_comparison_summary.csv"
)


FEATURE_SETS = {
    "raw": "logistic_raw_results.csv",
    "latent": "logistic_latent_results.csv",
    "raw_plus_latent": "logistic_raw_plus_latent_results.csv",
}

METRICS = ["auc", "balanced_accuracy", "f1"]


def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def mean_metrics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    out = {}
    for m in METRICS:
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
            out[m] = float(df[m].mean())
        else:
            out[m] = None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_dir",
        default=LR_RESULTS_BASELINE,
        help="Directory with baseline logistic_*_results.csv (default: results/part3.../results_baseline)",
    )
    parser.add_argument(
        "--modified_dir",
        default=LR_RESULTS_MODIFIED,
        help="Directory with modified logistic CSVs (default: results/part3.../results_modified)",
    )
    parser.add_argument(
        "--output",
        default=LR_COMPARISON_SUMMARY_CSV,
        help="Output CSV path (default: results/part3_logistic_regression/logistic_comparison_summary.csv)",
    )
    args = parser.parse_args()

    rows = []

    print("=== Logistic regression comparison (LOIO) ===")
    print(f"baseline_dir: {args.baseline_dir}")
    print(f"modified_dir: {args.modified_dir}")

    for feature_set_name, filename in FEATURE_SETS.items():
        base_path = os.path.join(args.baseline_dir, filename)
        mod_path = os.path.join(args.modified_dir, filename)

        base_df = read_csv_if_exists(base_path)
        mod_df = read_csv_if_exists(mod_path)

        if base_df is None or mod_df is None:
            missing = []
            if base_df is None:
                missing.append(base_path)
            if mod_df is None:
                missing.append(mod_path)
            print(f"[WARN] Missing {feature_set_name} results: {', '.join(missing)}")
            continue

        base_means = mean_metrics(base_df)
        mod_means = mean_metrics(mod_df)

        print(f"\n--- Feature set: {feature_set_name} ---")
        for m in METRICS:
            b = base_means.get(m)
            md = mod_means.get(m)
            if b is None or md is None:
                print(f"{m:18s}: missing (baseline={b}, modified={md})")
                continue
            abs_diff = md - b
            rel_diff = abs_diff / b if b != 0 else None
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "metric": m,
                    "baseline_mean": b,
                    "modified_mean": md,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )
            print(f"{m:18s}: baseline={b:.6f} modified={md:.6f} abs_diff={abs_diff:+.6f}")

    summary_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary_df.to_csv(args.output, index=False)

    print(f"\nSaved: {args.output}")
    if summary_df.empty:
        print("[WARN] No rows were written (likely missing input CSVs or metrics).")


if __name__ == "__main__":
    main()

