#!/usr/bin/env python3
"""
Compare baseline vs modified transfer learning results.

Reads (per directory), first match:
  - quick_probe_results.csv, quick_probe_results_{baseline,modified}.csv
  - latent_dim_comparison.csv, latent_dim_comparison_{...}.csv

Outputs:
  - results/transfer_learning/comparisons/transfer_learning_comparison_summary.csv (default)

Usage (from lab2/code):
  python transfer_learning/compare_transfer_results.py
"""

import argparse
import os
from typing import List, Optional

import pandas as pd

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TL_RESULTS_BASELINE = os.path.join(
    _CODE, "results", "transfer_learning", "results_baseline"
)
TL_RESULTS_MODIFIED = os.path.join(
    _CODE, "results", "transfer_learning", "results_modified"
)
TL_COMPARISON_SUMMARY_CSV = os.path.join(
    _CODE,
    "results",
    "transfer_learning",
    "comparisons",
    "transfer_learning_comparison_summary.csv",
)


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_quick_probe_csv(results_dir: str) -> Optional[str]:
    return _first_existing(
        [
            os.path.join(results_dir, "quick_probe_results.csv"),
            os.path.join(results_dir, "quick_probe_results_baseline.csv"),
            os.path.join(results_dir, "quick_probe_results_modified.csv"),
        ]
    )


def resolve_latent_dim_csv(results_dir: str) -> Optional[str]:
    return _first_existing(
        [
            os.path.join(results_dir, "latent_dim_comparison.csv"),
            os.path.join(results_dir, "latent_dim_comparison_baseline.csv"),
            os.path.join(results_dir, "latent_dim_comparison_modified.csv"),
        ]
    )


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
    parser.add_argument(
        "--baseline_dir",
        default=TL_RESULTS_BASELINE,
        help="Directory with baseline quick_probe / latent_dim CSVs",
    )
    parser.add_argument(
        "--modified_dir",
        default=TL_RESULTS_MODIFIED,
        help="Directory with modified quick_probe / latent_dim CSVs",
    )
    parser.add_argument(
        "--output",
        default=TL_COMPARISON_SUMMARY_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--data_relpath",
        default="",
        help="(Optional) Prefix if running outside lab2/code with relative dirs.",
    )
    args = parser.parse_args()

    baseline_dir = os.path.join(args.data_relpath, args.baseline_dir).rstrip("/")
    modified_dir = os.path.join(args.data_relpath, args.modified_dir).rstrip("/")

    quick_probe_metrics = ["accuracy_mean", "accuracy_std", "roc_auc_mean", "roc_auc_std"]

    b_qp = resolve_quick_probe_csv(baseline_dir)
    m_qp = resolve_quick_probe_csv(modified_dir)
    baseline_quick = read_quick_probe(b_qp) if b_qp else {}
    modified_quick = read_quick_probe(m_qp) if m_qp else {}

    b_ld = resolve_latent_dim_csv(baseline_dir)
    m_ld = resolve_latent_dim_csv(modified_dir)
    baseline_latent = read_latent_dim_table(b_ld) if b_ld else {}
    modified_latent = read_latent_dim_table(m_ld) if m_ld else {}

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
        print(
            "- Missing quick probe CSV in one/both dirs. "
            "Expected quick_probe_results.csv or quick_probe_results_{baseline,modified}.csv "
            f"under {baseline_dir} and {modified_dir}."
        )
    else:
        # Print concise subset
        print(comparison_df[["metric", "baseline_value", "modified_value", "abs_diff"]].to_string(index=False))

    print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()

