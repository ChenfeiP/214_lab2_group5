#!/usr/bin/env python3
"""
Generate latent dimension comparison table for B deliverable.
Usage:
  python latent_dim_table.py -o results/latent_dim_comparison.csv
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="results/latent_dim_comparison.csv")
    parser.add_argument("--probe_csv", default="results/quick_probe_results.csv")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--source", default="finetune_final")
    args = parser.parse_args()

    row = {"latent_dim": args.latent_dim, "source": args.source}

    if os.path.exists(args.probe_csv):
        df = pd.read_csv(args.probe_csv)
        for _, r in df.iterrows():
            row[r["metric"]] = r["value"]

    table = pd.DataFrame([row])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    table.to_csv(args.output, index=False)
    print("Latent dim comparison:")
    print(table.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
