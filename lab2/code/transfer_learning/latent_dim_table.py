#!/usr/bin/env python3
"""
Generate latent dimension comparison table for B deliverable.
Usage (from lab2/code):
  python transfer_learning/latent_dim_table.py --variant modified
"""

import argparse
import os

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

TL_RESULTS_BASELINE = os.path.join(
    _CODE, "results", "transfer_learning", "results_baseline"
)
TL_RESULTS_MODIFIED = os.path.join(
    _CODE, "results", "transfer_learning", "results_modified"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("baseline", "modified"),
        default="modified",
    )
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--probe_csv", default=None)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--source", default="finetune_final")
    args = parser.parse_args()

    res_dir = TL_RESULTS_BASELINE if args.variant == "baseline" else TL_RESULTS_MODIFIED
    output = args.output or os.path.join(res_dir, "latent_dim_comparison.csv")
    probe_csv = args.probe_csv or os.path.join(res_dir, "quick_probe_results.csv")

    row = {"latent_dim": args.latent_dim, "source": args.source}

    if os.path.exists(probe_csv):
        df = pd.read_csv(probe_csv)
        for _, r in df.iterrows():
            row[r["metric"]] = r["value"]

    table = pd.DataFrame([row])
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    table.to_csv(output, index=False)
    print("Latent dim comparison:")
    print(table.to_string(index=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
