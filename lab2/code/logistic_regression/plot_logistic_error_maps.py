#!/usr/bin/env python3
"""
Plot spatial misclassification maps from logistic_experiments *preds.csv.

Each row should have: y, x, label, image_id, prob_cloud, pred, is_error

Usage (from lab2/code):
  python logistic_regression/plot_logistic_error_maps.py \\
    --preds_csv results/part3_logistic_regression/results_modified/logistic_latent_preds.csv \\
    --out_dir results/part3_logistic_regression/results_modified \\
    --prefix latent
"""

import argparse
import os
import sys

import pandas as pd

_LR_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_LR_DIR)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
if _LR_DIR not in sys.path:
    sys.path.insert(0, _LR_DIR)

from logistic_experiments import save_error_maps  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_csv", required=True, help="e.g. logistic_raw_preds.csv")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write PNGs (subfolders use prefix internally if needed)",
    )
    parser.add_argument(
        "--prefix",
        default="preds",
        help="Tag for plot titles / saved under out_dir/error_maps/<prefix>/",
    )
    args = parser.parse_args()

    if not os.path.exists(args.preds_csv):
        raise FileNotFoundError(args.preds_csv)

    df = pd.read_csv(args.preds_csv)
    required = {"y", "x", "is_error", "image_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"preds CSV missing columns: {missing}")

    save_error_maps(df, args.out_dir, args.prefix)
    print(f"Saved error maps under {args.out_dir}/error_maps/{args.prefix}/")


if __name__ == "__main__":
    main()
