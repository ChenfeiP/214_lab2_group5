#!/usr/bin/env python3
"""
Image-level logistic regression experiments to evaluate transfer learning embeddings.

This script:
1. Loads the 3 labeled MISR images from ../image_data_float32/*.npz
2. Merges raw features with autoencoder embeddings (image*_ae.csv)
3. Runs Leave-One-Image-Out CV logistic regression with 3 feature sets:
   - raw only
   - latent only (autoencoder embeddings)
   - raw + latent
4. Saves per-fold results, per-pixel predictions, coefficient tables, top-|coef|
   bar plots (standardized features), and spatial misclassification maps under OUT_DIR.

IMPORTANT: Files/paths you must have or adjust:
- FEATURE_NAMES: names and order of raw features in the npz (after y,x)

Usage (from lab2/code/):
    python logistic_experiments.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


RAW_NPZ_PATHS = {
    "image1": "../image_data_float32/O013257.npz",
    "image2": "../image_data_float32/O013490.npz",
    "image3": "../image_data_float32/O012791.npz",
}

EMBEDDING_DIR = "transfer_learning_results/images"

FEATURE_NAMES = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]

# Output directory for all logistic experiment results.
OUT_DIR = "logistic_experiments_results"
os.makedirs(OUT_DIR, exist_ok=True)


def load_labeled_npz(path: str, feature_names):
    """
    Load a labeled MISR npz file and return a DataFrame with columns:
    y, x, <raw features...>, label

    We assume the array has columns:
        [y, x] + feature_names + [label]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw labeled npz not found: {path}")

    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        key = list(arr.files)[0]
        arr = arr[key]

    arr = np.asarray(arr)
    expected_cols = 2 + len(feature_names) + 1
    if arr.shape[1] != expected_cols:
        raise ValueError(
            f"{path} has {arr.shape[1]} columns, expected {expected_cols} "
            f"for [y, x] + {len(feature_names)} features + label."
        )

    cols = ["y", "x"] + feature_names + ["label"]
    df = pd.DataFrame(arr, columns=cols)

    # Map labels from {-1, 0, +1} to {0, 1}, dropping unlabeled (0)
    if df["label"].dtype != int and df["label"].dtype != np.int64:
        df["label"] = df["label"].astype(int)
    valid = df["label"] != 0
    df = df.loc[valid].copy()
    df["label"] = (df["label"] == 1).astype(int)
    return df


def merge_with_ae(raw_df: pd.DataFrame, ae_csv_path: str, image_id: str):
    """
    Merge raw labeled DataFrame with autoencoder embeddings on (y, x).

    raw_df: columns y, x, raw features..., label
    ae_csv_path: path to image*_ae.csv with columns y, x, ae0..ae(k-1)
    """
    if not os.path.exists(ae_csv_path):
        raise FileNotFoundError(f"Embedding CSV not found: {ae_csv_path}")

    ae_df = pd.read_csv(ae_csv_path)
    ae_cols = [c for c in ae_df.columns if c.startswith("ae")]

    merged = raw_df.merge(ae_df, on=["y", "x"], how="inner").copy()
    if merged.empty:
        raise ValueError(f"Merge on (y, x) produced 0 rows for {image_id}.")

    merged["image_id"] = image_id
    return merged, ae_cols


def _coef_dataframe_from_pipeline(model: Pipeline, feature_cols, test_image: str):
    """
    Extract coefficients from StandardScaler + LogisticRegression pipeline.
    - coef_scaled: weight on standardized features (mean 0, variance 1).
    - coef_per_unit_original: approximate d(logit)/d(raw_feature) = coef_scaled / scale.
    """
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    coef = np.asarray(clf.coef_).ravel()
    scale = np.asarray(scaler.scale_).ravel()
    # avoid div by zero if a column is constant
    scale_safe = np.where(scale > 1e-12, scale, 1.0)
    coef_orig = coef / scale_safe
    intercept = float(np.asarray(clf.intercept_).ravel()[0])

    rows = []
    for j, name in enumerate(feature_cols):
        rows.append(
            {
                "test_image": test_image,
                "feature": name,
                "coef_scaled": float(coef[j]),
                "scaler_scale": float(scale[j]),
                "coef_per_unit_original": float(coef_orig[j]),
                "abs_coef_scaled": float(abs(coef[j])),
            }
        )
    rows.append(
        {
            "test_image": test_image,
            "feature": "__intercept__",
            "coef_scaled": intercept,
            "scaler_scale": np.nan,
            "coef_per_unit_original": intercept,
            "abs_coef_scaled": np.nan,
        }
    )
    return pd.DataFrame(rows)


def save_coef_topk_plots(coef_df: pd.DataFrame, out_dir: str, prefix: str, top_k: int = 10):
    """
    For each LOIO fold (test_image), bar plot top-k features by |coef_scaled|
    (coefficients on standardized inputs, i.e. after StandardScaler).
    """
    plot_root = os.path.join(out_dir, "coef_plots", prefix)
    os.makedirs(plot_root, exist_ok=True)

    for test_img in sorted(coef_df["test_image"].unique()):
        sub = coef_df[
            (coef_df["test_image"] == test_img) & (coef_df["feature"] != "__intercept__")
        ].copy()
        if sub.empty:
            continue
        sub = sub.nlargest(top_k, "abs_coef_scaled")

        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * top_k)))
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub["coef_scaled"].to_numpy(), color="steelblue", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["feature"].tolist(), fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Coefficient (standardized features)")
        ax.set_title(f"{prefix}: top {top_k} |coef| — held-out {test_img}")
        ax.invert_yaxis()
        fig.tight_layout()
        safe = str(test_img).replace("/", "_")
        fig.savefig(
            os.path.join(plot_root, f"{safe}_top{top_k}_coef_scaled.png"),
            dpi=150,
        )
        plt.close(fig)


def save_error_maps(preds_df: pd.DataFrame, out_dir: str, prefix: str):
    """
    Spatial misclassification maps: one PNG per image_id (is_error on y-x grid).
    """
    def _ik(val):
        return int(round(float(val)))

    map_root = os.path.join(out_dir, "error_maps", prefix)
    os.makedirs(map_root, exist_ok=True)

    for img_id in sorted(preds_df["image_id"].unique()):
        sub = preds_df[preds_df["image_id"] == img_id]
        ys = sub["y"].to_numpy()
        xs = sub["x"].to_numpy()
        err = sub["is_error"].to_numpy(dtype=float)
        uy = np.sort(np.unique([_ik(v) for v in ys]))
        ux = np.sort(np.unique([_ik(v) for v in xs]))
        y2i = {int(y): i for i, y in enumerate(uy)}
        x2i = {int(x): i for i, x in enumerate(ux)}
        grid = np.full((len(uy), len(ux)), np.nan, dtype=float)
        for y, x, e in zip(ys, xs, err):
            grid[y2i[_ik(y)], x2i[_ik(x)]] = e

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="upper",
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"{prefix}: misclassification (1=error) — {img_id}")
        ax.set_xlabel("column index (sorted x)")
        ax.set_ylabel("row index (sorted y)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe = str(img_id).replace("/", "_")
        fig.savefig(
            os.path.join(map_root, f"{safe}_error_map.png"),
            dpi=150,
        )
        plt.close(fig)


def run_loio_logistic(df: pd.DataFrame, feature_cols, threshold: float = 0.5):
    """
    Leave-One-Image-Out logistic regression.

    df must contain:
      - 'image_id': which image each pixel belongs to
      - 'label': 0/1 (non-cloud / cloud)
      - feature columns specified in feature_cols
    """
    image_ids = sorted(df["image_id"].unique())

    fold_results = []
    all_preds = []
    all_coef_frames = []

    for test_image in image_ids:
        train_df = df[df["image_id"] != test_image].copy()
        test_df = df[df["image_id"] == test_image].copy()

        X_train = train_df[feature_cols].to_numpy()
        y_train = train_df["label"].astype(int).to_numpy()

        X_test = test_df[feature_cols].to_numpy()
        y_test = test_df["label"].astype(int).to_numpy()

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="liblinear",
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        coef_df_fold = _coef_dataframe_from_pipeline(model, feature_cols, test_image)
        all_coef_frames.append(coef_df_fold)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= threshold).astype(int)

        auc = roc_auc_score(y_test, prob)
        bal_acc = balanced_accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        fold_results.append(
            {
                "test_image": test_image,
                "auc": auc,
                "balanced_accuracy": bal_acc,
                "f1": f1,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

        tmp = test_df[["y", "x", "label", "image_id"]].copy()
        tmp["prob_cloud"] = prob
        tmp["pred"] = pred
        tmp["is_error"] = (tmp["pred"] != tmp["label"]).astype(int)
        all_preds.append(tmp)

    results_df = pd.DataFrame(fold_results)
    preds_df = pd.concat(all_preds, ignore_index=True)
    coef_df = pd.concat(all_coef_frames, ignore_index=True)
    return results_df, preds_df, coef_df


def main():
    # ----- Step 1: load 3 labeled images -----
    raw_dfs = {}
    for image_id, path in RAW_NPZ_PATHS.items():
        raw_dfs[image_id] = load_labeled_npz(path, FEATURE_NAMES)

    # ----- Step 2: merge with embeddings -----
    df_list = []
    ae_cols = None
    for idx, image_id in enumerate(sorted(RAW_NPZ_PATHS.keys()), start=1):
        raw_df = raw_dfs[image_id]
        ae_path = os.path.join(EMBEDDING_DIR, f"image{idx}_ae.csv")
        merged, ae_cols = merge_with_ae(raw_df, ae_path, image_id)
        df_list.append(merged)

    df_all = pd.concat(df_list, ignore_index=True)

    raw_features = FEATURE_NAMES
    latent_features = ae_cols
    raw_plus_latent = raw_features + latent_features

    print(f"Total labeled pixels after merge: {df_all.shape[0]}")
    print(f"Raw feature columns: {raw_features}")
    print(f"Latent feature columns: {latent_features}")

    # ----- Step 3: run 3 feature-set versions -----
    print("\n=== RAW ONLY ===")
    res_raw, pred_raw, coef_raw = run_loio_logistic(df_all, raw_features)
    print(res_raw)
    print("\nRAW ONLY (mean over folds):")
    print(res_raw.mean(numeric_only=True))

    print("\n=== LATENT ONLY ===")
    res_latent, pred_latent, coef_latent = run_loio_logistic(df_all, latent_features)
    print(res_latent)
    print("\nLATENT ONLY (mean over folds):")
    print(res_latent.mean(numeric_only=True))

    print("\n=== RAW + LATENT ===")
    res_both, pred_both, coef_both = run_loio_logistic(df_all, raw_plus_latent)
    print(res_both)
    print("\nRAW + LATENT (mean over folds):")
    print(res_both.mean(numeric_only=True))

    # ----- Step 4: save results and predictions -----
    res_raw.to_csv(
        os.path.join(OUT_DIR, "logistic_raw_results.csv"), index=False
    )
    res_latent.to_csv(
        os.path.join(OUT_DIR, "logistic_latent_results.csv"), index=False
    )
    res_both.to_csv(
        os.path.join(OUT_DIR, "logistic_raw_plus_latent_results.csv"), index=False
    )

    pred_raw.to_csv(
        os.path.join(OUT_DIR, "logistic_raw_preds.csv"), index=False
    )
    pred_latent.to_csv(
        os.path.join(OUT_DIR, "logistic_latent_preds.csv"), index=False
    )
    pred_both.to_csv(
        os.path.join(OUT_DIR, "logistic_raw_plus_latent_preds.csv"), index=False
    )

    # Coefficients (per fold) and interpretability plots
    coef_raw.to_csv(os.path.join(OUT_DIR, "logistic_raw_coefs.csv"), index=False)
    coef_latent.to_csv(os.path.join(OUT_DIR, "logistic_latent_coefs.csv"), index=False)
    coef_both.to_csv(
        os.path.join(OUT_DIR, "logistic_raw_plus_latent_coefs.csv"), index=False
    )

    save_coef_topk_plots(coef_raw, OUT_DIR, "raw", top_k=10)
    save_coef_topk_plots(coef_latent, OUT_DIR, "latent", top_k=10)
    save_coef_topk_plots(coef_both, OUT_DIR, "raw_plus_latent", top_k=10)

    save_error_maps(pred_raw, OUT_DIR, "raw")
    save_error_maps(pred_latent, OUT_DIR, "latent")
    save_error_maps(pred_both, OUT_DIR, "raw_plus_latent")

    print("\nSaved logistic experiment results to logistic_experiments_results/.")
    print("Also saved coef CSVs under logistic_*_coefs.csv, coef_plots/, error_maps/.")


if __name__ == "__main__":
    main()

