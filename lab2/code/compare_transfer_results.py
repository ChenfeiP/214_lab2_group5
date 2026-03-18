#!/usr/bin/env python3
"""
Compare baseline vs modified transfer learning results.

Reads (per directory):
  - quick_probe_results.csv
  - latent_dim_comparison.csv

Outputs:
  - a human-readable CSV with baseline vs modified values + deltas
"""

import argparse
import os
import pandas as pd


def read_quick_probe(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if set(df.columns) >= {"metric", "value"}:
        return dict(zip(df["metric"].astype(str), df["value"]))
    return {}


def read_latent_dim_table(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    # Typically one row; keep the first.
    row = df.iloc[0].to_dict()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", default="transfer_learning_results_baseline")
    parser.add_argument("--modified_dir", default="transfer_learning_results_modified")
    parser.add_argument(
        "--output",
        default="transfer_learning_comparison_summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--data_relpath",
        default="",
        help="(Optional) If you run from a different working dir; prefix for relative paths.",
    )
    args = parser.parse_args()

    baseline_dir = os.path.join(args.data_relpath, args.baseline_dir).rstrip("/")
    modified_dir = os.path.join(args.data_relpath, args.modified_dir).rstrip("/")

    quick_probe_metrics = ["accuracy_mean", "accuracy_std", "roc_auc_mean", "roc_auc_std"]

    baseline_quick = read_quick_probe(os.path.join(baseline_dir, "quick_probe_results.csv"))
    modified_quick = read_quick_probe(os.path.join(modified_dir, "quick_probe_results.csv"))

    baseline_latent = read_latent_dim_table(os.path.join(baseline_dir, "latent_dim_comparison.csv"))
    modified_latent = read_latent_dim_table(os.path.join(modified_dir, "latent_dim_comparison.csv"))

    # Metrics comparison (use quick_probe as the primary source).
    rows = []
    for m in quick_probe_metrics:
        b = baseline_quick.get(m, None)
        md = modified_quick.get(m, None)
        if b is None or md is None:
            continue
        abs_diff = md - b
        rel_diff = None
        if b != 0:
            rel_diff = abs_diff / b
        rows.append(
            {
                "metric": m,
                "baseline_value": b,
                "modified_value": md,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )

    comparison_df = pd.DataFrame(rows)

    # Add a small header-style "context" block via extra columns.
    # (CSV consumers can ignore these.)
    latent_dim_b = baseline_latent.get("latent_dim", None)
    latent_dim_m = modified_latent.get("latent_dim", None)
    source_b = baseline_latent.get("source", None)
    source_m = modified_latent.get("source", None)
    comparison_df["baseline_latent_dim"] = latent_dim_b
    comparison_df["modified_latent_dim"] = latent_dim_m
    comparison_df["baseline_source"] = source_b
    comparison_df["modified_source"] = source_m

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    comparison_df.to_csv(args.output, index=False)

    print("\nQuick probe comparison (baseline vs modified):")
    if comparison_df.empty:
        print("- Missing quick_probe_results.csv in one/both directories.")
    else:
        # Print concise subset
        print(comparison_df[["metric", "baseline_value", "modified_value", "abs_diff"]].to_string(index=False))

    print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()

