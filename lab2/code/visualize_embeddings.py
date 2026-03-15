#!/usr/bin/env python3
"""
Generate PCA and t-SNE plots of embeddings for B deliverables.
Requires: results/image1_ae.csv, image2_ae.csv, image3_ae.csv (from get_embedding.py)
          and corresponding npz files with expert labels.
Usage:
  python visualize_embeddings.py -o results/pca_tsne.png
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_labeled_embeddings(results_dir="results", image_data_dir="../image_data_float32"):
    """Load embeddings + expert labels for the 3 labeled images."""
    labeled_ids = ["O013257", "O013490", "O012791"]
    all_emb, all_labels = [], []
    for i, img_id in enumerate(labeled_ids):
        csv_path = os.path.join(results_dir, f"image{i+1}_ae.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        ae_cols = [c for c in df.columns if c.startswith("ae")]
        emb = df[ae_cols].values
        npz_path = os.path.join(image_data_dir, f"{img_id}.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            key = list(data.files)[0]
            arr = data[key]
            if arr.shape[1] == 11:
                labels = arr[:, -1]
                valid = labels != 0
                emb, labels = emb[valid], labels[valid]
                all_emb.append(emb)
                all_labels.append(labels)
        else:
            all_emb.append(emb)
            all_labels.append(np.zeros(len(emb)))
    if not all_emb:
        return None, None
    return np.vstack(all_emb), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="results")
    parser.add_argument("-d", "--data_dir", default="../image_data_float32")
    parser.add_argument("-o", "--output", default="results/pca_tsne.png")
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()

    emb, labels = load_labeled_embeddings(args.input_dir, args.data_dir)
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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
