#!/usr/bin/env python3
"""
Quick probe: train a simple classifier on embeddings to predict cloud vs no-cloud.
Reports accuracy/AUC as evidence that embeddings are useful (B deliverable).
Usage (from lab2/code):
  python transfer_learning/quick_probe.py --variant modified
  python transfer_learning/quick_probe.py --variant baseline
"""

import argparse
import os
from typing import List, Optional, Set

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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
    """Load embeddings + labels for the 3 labeled images (O013257, O013490, O012791).
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
    parser.add_argument(
        "--variant",
        choices=("baseline", "modified"),
        default="modified",
        help="Which TL results folder and embedding CSV naming to use.",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default=None,
        help="Directory with image*_ae*.csv (default: results/transfer_learning/results_<variant>)",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        default=None,
        help="Directory with O013257.npz etc. (default: lab2/data or image_data_float32)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV (default: <input_dir>/quick_probe_results.csv)",
    )
    args = parser.parse_args()

    variant = args.variant
    input_dir = args.input_dir or (
        TL_RESULTS_BASELINE if variant == "baseline" else TL_RESULTS_MODIFIED
    )
    data_dir = args.data_dir or _lab_data_dir()
    output = args.output or os.path.join(input_dir, "quick_probe_results.csv")

    X, y = load_labeled_embeddings(input_dir, data_dir, variant)
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
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    results.to_csv(output, index=False)
    print("Quick probe results:")
    print(results.to_string(index=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
