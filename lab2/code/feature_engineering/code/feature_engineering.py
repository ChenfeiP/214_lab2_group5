#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


LABELED_IMAGE_IDS = ["O013257", "O013490", "O012791"]

FEATURE_COLUMNS = [
    "NDAI",
    "SD",
    "CORR",
    "Radiance DF",
    "Radiance CF",
    "Radiance BF",
    "Radiance AF",
    "Radiance AN",
]

NPZ_COLUMNS = [
    "y",
    "x",
    "NDAI",
    "SD",
    "CORR",
    "Radiance DF",
    "Radiance CF",
    "Radiance BF",
    "Radiance AF",
    "Radiance AN",
    "label",
]


def load_and_prepare_data(image_dir):
    """Load the 3 labeled images, keep labels in {-1, +1}, and add helper columns."""
    frames = []

    for image_id in LABELED_IMAGE_IDS:
        arr = np.load(image_dir / f"{image_id}.npz")["arr_0"]
        df = pd.DataFrame(arr, columns=NPZ_COLUMNS)
        df["image_id"] = image_id
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin([-1, 1])].copy()
    df["label_binary"] = (df["label"] == 1).astype(int)
    df["class_name"] = df["label_binary"].map({0: "Non-cloud", 1: "Cloud"})

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_cohens_d(cloud_values, non_cloud_values):
    """Compute Cohen's d for cloud minus non-cloud."""
    if len(cloud_values) < 2 or len(non_cloud_values) < 2:
        return np.nan

    cloud_sd = np.std(cloud_values, ddof=1)
    non_cloud_sd = np.std(non_cloud_values, ddof=1)
    pooled_var = (
        (len(cloud_values) - 1) * cloud_sd**2
        + (len(non_cloud_values) - 1) * non_cloud_sd**2
    ) / (len(cloud_values) + len(non_cloud_values) - 2)

    if pooled_var <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_var)
    return (np.mean(cloud_values) - np.mean(non_cloud_values)) / pooled_sd


def compute_auc(y_true, scores):
    """
    Compute single-feature ROC AUC.
    Use max(auc, 1 - auc) so features with reversed direction still get credit.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
    if pd.Series(scores).nunique(dropna=True) < 2:
        return 0.5

    auc = roc_auc_score(y_true, scores)
    return max(auc, 1 - auc)


def compute_feature_metrics(df, feature):
    """Compute summary metrics for one original feature."""
    temp = df[[feature, "label_binary", "image_id"]].dropna().copy()

    cloud = temp.loc[temp["label_binary"] == 1, feature].to_numpy()
    non_cloud = temp.loc[temp["label_binary"] == 0, feature].to_numpy()

    cloud_mean = np.mean(cloud)
    non_cloud_mean = np.mean(non_cloud)
    diff = cloud_mean - non_cloud_mean
    cohens_d = compute_cohens_d(cloud, non_cloud)

    results = {
        "feature": feature,
        "cloud_mean": cloud_mean,
        "non_cloud_mean": non_cloud_mean,
        "difference_in_means": diff,
        "cohens_d": cohens_d,
        "abs_cohens_d": abs(cohens_d) if not np.isnan(cohens_d) else np.nan,
        "roc_auc": compute_auc(temp["label_binary"], temp[feature]),
    }

    per_image_aucs = []
    for image_id in LABELED_IMAGE_IDS:
        temp_image = temp[temp["image_id"] == image_id]
        image_auc = compute_auc(temp_image["label_binary"], temp_image[feature])
        results[f"roc_auc_{image_id}"] = image_auc
        per_image_aucs.append(image_auc)

    results["mean_per_image_roc_auc"] = np.nanmean(per_image_aucs)
    return results


def evaluate_all_features(df):
    """Rank the 8 original features by single-feature ROC AUC."""
    rows = [compute_feature_metrics(df, feature) for feature in FEATURE_COLUMNS]
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["roc_auc", "abs_cohens_d"], ascending=[False, False]
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def main(image_dir, output_dir):
    """Run the Part 2.1 quantitative feature-ranking pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(image_dir)
    summary = evaluate_all_features(df)
    summary.to_csv(output_dir / "feature_ranking_summary.csv", index=False)

    print("\nFeature ranking summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary table to: {output_dir / 'feature_ranking_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank original MISR features for cloud vs non-cloud separation."
    )
    parser.add_argument("--image_dir", type=Path, default=Path("../../../image_data_float32"))
    parser.add_argument(
        "--output_dir", type=Path, default=Path("../results/feature_engineering_part21")
    )
    args = parser.parse_args()
    main(args.image_dir, args.output_dir)

