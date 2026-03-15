#!/usr/bin/env python3
"""
Quick probe: train a simple classifier on embeddings to predict cloud vs no-cloud.
Reports accuracy/AUC as evidence that embeddings are useful (B deliverable).
Usage:
  python quick_probe.py -i results -o results/quick_probe_results.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def load_labeled_embeddings(results_dir, image_data_dir):
    """Load embeddings + labels for the 3 labeled images (O013257, O013490, O012791).
    Align by (y, x): CSV and npz can have different row counts, so we join on coordinates."""
    labeled_ids = ["O013257", "O013490", "O012791"]
    all_emb, all_labels = [], []
    for i, img_id in enumerate(labeled_ids):
        csv_path = os.path.join(results_dir, f"image{i+1}_ae.csv")
        if not os.path.exists(csv_path):
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
        # Only keep labeled pixels (label != 0)
        valid_npz = labels_full != 0
        label_df = pd.DataFrame({"y": ys_npz[valid_npz], "x": xs_npz[valid_npz], "label": labels_full[valid_npz]})
        # Merge CSV (y, x, ae*) with labels on (y, x) so row counts match
        merged = df.merge(label_df, on=["y", "x"], how="inner")
        if merged.shape[0] == 0:
            continue
        emb = merged[ae_cols].values
        labels = (merged["label"].values == 1).astype(int)  # +1 -> 1, -1 -> 0
        all_emb.append(emb)
        all_labels.append(labels)
    if not all_emb:
        return None, None
    return np.vstack(all_emb), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="results")
    parser.add_argument("-d", "--data_dir", default="../image_data_float32")
    parser.add_argument("-o", "--output", default="results/quick_probe_results.csv")
    args = parser.parse_args()

    X, y = load_labeled_embeddings(args.input_dir, args.data_dir)
    if X is None:
        print("No embedding CSVs found. Run get_embedding.py first.")
        return

    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42)

    acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")

    results = pd.DataFrame({
        "metric": ["accuracy_mean", "accuracy_std", "roc_auc_mean", "roc_auc_std"],
        "value": [acc.mean(), acc.std(), auc.mean(), auc.std()],
    })
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    print("Quick probe results:")
    print(results.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
