#!/usr/bin/env python3
"""
Generate PCA and t-SNE plots of embeddings for B deliverables.
Requires: image*_ae*.csv from get_embedding.py and labeled npz under lab2/data.
Usage (from lab2/code):
  python transfer_learning/visualize_embeddings.py --variant modified
"""

import argparse
import os
from typing import List, Optional, Set

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

TL_RESULTS_BASELINE = os.path.join(
    _CODE, "results", "transfer_learning", "results_baseline"
)
TL_RESULTS_MODIFIED = os.path.join(
    _CODE, "results", "transfer_learning", "results_modified"
)


def _lab_data_dir() -> str:
    primary = os.path.normpath(os.path.join(_CODE, "..", "data"))
    alt = os.path.normpath(os.path.join(_CODE, "..", "image_data_float32"))

    def has_npz(d: str) -> bool:
        try:
            return any(name.endswith(".npz") for name in os.listdir(d))
        except OSError:
            return False

    if os.path.isdir(primary) and has_npz(primary):
        return primary
    if os.path.isdir(alt):
        return alt
    return primary


def _embedding_csv_candidates(results_dir: str, index_1based: int, variant: str) -> List[str]:
    stem = f"image{index_1based}_ae"
    names: List[str] = [f"{stem}.csv"]
    if variant == "baseline":
        names.insert(0, f"{stem}_baseline.csv")
    elif variant == "modified":
        names.insert(0, f"{stem}_modified.csv")
    seen: Set[str] = set()
    ordered: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return [os.path.join(results_dir, n) for n in ordered]


def _first_existing_embedding_csv(
    results_dir: str, index_1based: int, variant: str
) -> Optional[str]:
    for p in _embedding_csv_candidates(results_dir, index_1based, variant):
        if os.path.exists(p):
            return p
    return None


def load_labeled_embeddings(results_dir, image_data_dir, variant):
    """Load embeddings + expert labels for the 3 labeled images.
    Align by (y, x): CSV and npz can have different row counts, so we join on coordinates."""
    labeled_ids = ["O013257", "O013490", "O012791"]
    all_emb, all_labels = [], []
    for i, img_id in enumerate(labeled_ids):
        csv_path = _first_existing_embedding_csv(results_dir, i + 1, variant)
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        ae_cols = [c for c in df.columns if c.startswith("ae")]
        npz_path = os.path.join(image_data_dir, f"{img_id}.npz")
        if not os.path.exists(npz_path) or df.shape[0] == 0:
            continue
        data = np.load(npz_path)
        key = list(data.files)[0]
        arr = data[key]
        if arr.shape[1] != 11:
            continue
        # Build label lookup by (y, x); npz columns: 0=y, 1=x, 10=label
        labels_full = arr[:, -1]
        ys_npz, xs_npz = arr[:, 0].astype(int), arr[:, 1].astype(int)
        valid_npz = labels_full != 0
        label_df = pd.DataFrame({"y": ys_npz[valid_npz], "x": xs_npz[valid_npz], "label": labels_full[valid_npz]})
        merged = df.merge(label_df, on=["y", "x"], how="inner")
        if merged.shape[0] == 0:
            continue
        emb = merged[ae_cols].values
        labels = merged["label"].values  # keep +1 / -1 for plot (cloud / no cloud)
        all_emb.append(emb)
        all_labels.append(labels)
    if not all_emb:
        return None, None
    return np.vstack(all_emb), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("baseline", "modified"),
        default="modified",
    )
    parser.add_argument("-i", "--input_dir", default=None)
    parser.add_argument("-d", "--data_dir", default=None)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()

    variant = args.variant
    input_dir = args.input_dir or (
        TL_RESULTS_BASELINE if variant == "baseline" else TL_RESULTS_MODIFIED
    )
    data_dir = args.data_dir or _lab_data_dir()
    output = args.output or os.path.join(input_dir, "pca_tsne.png")

    emb, labels = load_labeled_embeddings(input_dir, data_dir, variant)
    if emb is None:
        print("No embedding CSVs found. Run get_embedding.py first.")
        return

    if len(emb) > args.sample:
        idx = np.random.default_rng(42).choice(len(emb), args.sample, replace=False)
        emb, labels = emb[idx], labels[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(emb)
    cloud = labels == 1
    no_cloud = labels == -1
    if cloud.any():
        axes[0].scatter(X_pca[cloud, 0], X_pca[cloud, 1], c="blue", alpha=0.5, s=5, label="cloud")
    if no_cloud.any():
        axes[0].scatter(X_pca[no_cloud, 0], X_pca[no_cloud, 1], c="orange", alpha=0.5, s=5, label="no cloud")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("PCA of Embeddings")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb) - 1))
    X_tsne = tsne.fit_transform(emb)
    if cloud.any():
        axes[1].scatter(X_tsne[cloud, 0], X_tsne[cloud, 1], c="blue", alpha=0.5, s=5, label="cloud")
    if no_cloud.any():
        axes[1].scatter(X_tsne[no_cloud, 0], X_tsne[no_cloud, 1], c="orange", alpha=0.5, s=5, label="no cloud")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].set_title("t-SNE of Embeddings")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
