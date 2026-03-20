#!/usr/bin/env python3
# Example usage: python feature_engineering_plot.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns


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

TOP_FEATURES = ["NDAI", "CORR", "SD"]  
EXAMPLE_IMAGE_ID = "O013257"           
LABELED_IMAGE_IDS = ["O013257", "O013490", "O012791"]

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


def load_combined_plot_data(image_dir):
    """
    Load the 3 labeled MISR images into one DataFrame.
    This keeps labels {-1, 0, +1} so spatial label maps can show unlabeled pixels.
    """
    frames = []
    for image_id in LABELED_IMAGE_IDS:
        arr = np.load(image_dir / f"{image_id}.npz")["arr_0"]
        df = pd.DataFrame(arr, columns=NPZ_COLUMNS)
        df["image_id"] = image_id
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def prepare_labeled_plot_data(df):
    """
    Return:
    - full_df: keeps labels {-1, 0, +1}
    - labeled_df: keeps only labels {-1, +1} for class-comparison plots
    """
    full_df = df.copy()

    numeric_cols = ["x", "y", "label"] + FEATURE_COLUMNS
    for col in numeric_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

    full_df = full_df.dropna(subset=["x", "y", "label", "image_id"]).copy()
    full_df["label"] = full_df["label"].astype(int)
    full_df["class_name"] = full_df["label"].map(
        {-1: "Non-cloud", 0: "Unlabeled", 1: "Cloud"}
    )

    labeled_df = full_df[full_df["label"].isin([-1, 1])].copy()
    return full_df, labeled_df


def sanitize_filename(name):
    """Convert a feature name into a simple filename."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def sample_for_plotting(df, group_cols, max_per_group=40000, random_state=42):
    """Downsample large groups for faster plotting."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    sampled = []
    for _, group_df in df.groupby(group_cols, sort=False, dropna=False):
        if len(group_df) > max_per_group:
            group_df = group_df.sample(n=max_per_group, random_state=random_state)
        sampled.append(group_df)

    if not sampled:
        return df.iloc[0:0].copy()
    return pd.concat(sampled, ignore_index=True)


def build_spatial_grid(image_df, value_col):
    """
    Convert one image from long form into a 2D grid for plotting.
    Returns grid, x_values, y_values.
    """
    temp = image_df[["x", "y", value_col]].dropna().copy()
    x_values = np.sort(temp["x"].astype(int).unique())
    y_values = np.sort(temp["y"].astype(int).unique())

    x_to_idx = {x: i for i, x in enumerate(x_values)}
    y_to_idx = {y: i for i, y in enumerate(y_values)}

    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for x_val, y_val, value in temp[["x", "y", value_col]].to_numpy():
        grid[y_to_idx[int(y_val)], x_to_idx[int(x_val)]] = float(value)

    return grid, x_values, y_values


def plot_label_map(full_df, image_id, output_dir):
    """Save one spatial label map for an example image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_df = full_df[full_df["image_id"] == image_id].copy()
    label_grid, x_values, y_values = build_spatial_grid(image_df, "label")

    cmap = ListedColormap(["#4C78A8", "#D9D9D9", "#E45756"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        label_grid,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="auto",
        extent=[x_values.min(), x_values.max(), y_values.max(), y_values.min()],
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["Non-cloud", "Unlabeled", "Cloud"])

    ax.set_title(f"Spatial label map: {image_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_dir / f"{image_id}_label_map.png", dpi=220)
    plt.close(fig)


def plot_feature_distribution(labeled_df, feature, output_dir):
    """Save one boxplot + density comparison figure for a feature."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = labeled_df[[feature, "class_name"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = sample_for_plotting(plot_df, "class_name", max_per_group=40000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.boxplot(
        data=plot_df,
        x="class_name",
        y=feature,
        order=["Non-cloud", "Cloud"],
        hue="class_name",
        dodge=False,
        legend=False,
        showfliers=False,
        palette={"Non-cloud": "#4C78A8", "Cloud": "#E45756"},
        ax=axes[0],
    )
    axes[0].set_title(f"{feature}: Cloud vs Non-cloud")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel(feature)

    sns.histplot(
        data=plot_df,
        x=feature,
        hue="class_name",
        hue_order=["Non-cloud", "Cloud"],
        bins=50,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        palette={"Non-cloud": "#4C78A8", "Cloud": "#E45756"},
        ax=axes[1],
    )
    axes[1].set_title(f"{feature}: Density comparison")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(output_dir / f"{sanitize_filename(feature)}_distribution.png", dpi=220)
    plt.close(fig)


def plot_top3_distributions(labeled_df, output_dir):
    """Save class-comparison plots for the selected top 3 features."""
    for feature in TOP_FEATURES:
        plot_feature_distribution(labeled_df, feature, output_dir)


def plot_top3_per_image(labeled_df, output_dir):
    """Save one 3-panel figure comparing Cloud vs Non-cloud across image_id."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(TOP_FEATURES), figsize=(6 * len(TOP_FEATURES), 5))
    if len(TOP_FEATURES) == 1:
        axes = [axes]

    for ax, feature in zip(axes, TOP_FEATURES):
        plot_df = labeled_df[["image_id", "class_name", feature]].dropna().copy()
        plot_df = sample_for_plotting(
            plot_df,
            ["image_id", "class_name"],
            max_per_group=20000,
        )

        sns.boxplot(
            data=plot_df,
            x="image_id",
            y=feature,
            hue="class_name",
            hue_order=["Non-cloud", "Cloud"],
            showfliers=False,
            palette={"Non-cloud": "#4C78A8", "Cloud": "#E45756"},
            ax=ax,
        )
        ax.set_title(f"{feature} by image and class")
        ax.set_xlabel("Image ID")
        ax.set_ylabel(feature)

        if ax is axes[0]:
            ax.legend(title="Class")
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    fig.tight_layout()
    fig.savefig(output_dir / "top3_per_image_comparison.png", dpi=220)
    plt.close(fig)


def plot_label_and_feature_heatmap(full_df, image_id, feature, output_dir):
    """Save a 2-panel figure with label map and one feature heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_df = full_df[full_df["image_id"] == image_id].copy()
    label_grid, x_values, y_values = build_spatial_grid(image_df, "label")
    feature_grid, _, _ = build_spatial_grid(image_df, feature)

    label_cmap = ListedColormap(["#4C78A8", "#D9D9D9", "#E45756"])
    label_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], label_cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    im0 = axes[0].imshow(
        label_grid,
        cmap=label_cmap,
        norm=label_norm,
        origin="upper",
        aspect="auto",
        extent=[x_values.min(), x_values.max(), y_values.max(), y_values.min()],
    )
    cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[-1, 0, 1])
    cbar0.ax.set_yticklabels(["Non-cloud", "Unlabeled", "Cloud"])
    axes[0].set_title(f"{image_id}: label map")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im1 = axes[1].imshow(
        feature_grid,
        cmap="viridis",
        origin="upper",
        aspect="auto",
        extent=[x_values.min(), x_values.max(), y_values.max(), y_values.min()],
    )
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title(f"{image_id}: {feature}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{image_id}_{sanitize_filename(feature)}_label_vs_feature.png",
        dpi=220,
    )
    plt.close(fig)


def plot_all_feature_distributions(labeled_df, output_dir):
    """Save class-comparison distributions for all 8 original features."""
    for feature in FEATURE_COLUMNS:
        plot_feature_distribution(labeled_df, feature, output_dir)


def plot_feature_correlation_heatmap(labeled_df, output_dir):
    """Compute and save the 8-feature correlation heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_df = labeled_df[FEATURE_COLUMNS].dropna().copy()
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation heatmap: original 8 features")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_correlation_heatmap.png", dpi=220)
    plt.close(fig)


def plot_top3_pairwise_scatter(labeled_df, output_dir):
    """Save pairwise scatter plots for the 3 selected top features."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = labeled_df[TOP_FEATURES + ["class_name"]].dropna().copy()
    plot_df = sample_for_plotting(plot_df, "class_name", max_per_group=3000)

    g = sns.pairplot(
        plot_df,
        vars=TOP_FEATURES,
        hue="class_name",
        hue_order=["Non-cloud", "Cloud"],
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.45, "s": 14},
        diag_kws={"bins": 30, "alpha": 0.6},
        palette={"Non-cloud": "#4C78A8", "Cloud": "#E45756"},
    )
    g.figure.suptitle("Top-3 feature pairwise scatter plots", y=1.02)
    g.savefig(output_dir / "top3_pairwise_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(g.figure)


def main(image_dir, output_dir):
    """Run the full figure-saving pipeline."""
    sns.set_theme(style="whitegrid", context="talk")

    full_df, labeled_df = prepare_labeled_plot_data(load_combined_plot_data(image_dir))

    label_map_dir = output_dir / "label_maps"
    top3_dist_dir = output_dir / "top3_feature_distributions"
    all8_dist_dir = output_dir / "all_feature_distributions"
    per_image_dir = output_dir / "top3_per_image"
    heatmap_compare_dir = output_dir / "label_feature_heatmaps"
    corr_dir = output_dir / "correlation_heatmap"
    pairwise_dir = output_dir / "top3_pairwise_scatter"

    plot_label_map(full_df, EXAMPLE_IMAGE_ID, label_map_dir)
    plot_top3_distributions(labeled_df, top3_dist_dir)
    plot_top3_per_image(labeled_df, per_image_dir)

    for feature in TOP_FEATURES:
        plot_label_and_feature_heatmap(full_df, EXAMPLE_IMAGE_ID, feature, heatmap_compare_dir)

    plot_all_feature_distributions(labeled_df, all8_dist_dir)
    plot_feature_correlation_heatmap(labeled_df, corr_dir)
    plot_top3_pairwise_scatter(labeled_df, pairwise_dir)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate feature-engineering plots from labeled MISR data."
    )
    parser.add_argument("--image_dir", type=Path, default=Path("../../../image_data_float32"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../results/feature_engineering_plots"),
    )
    args = parser.parse_args()
    main(args.image_dir, args.output_dir)
