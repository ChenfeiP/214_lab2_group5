"""Run sanity-check predictions on unlabeled images with the saved best RF model.

Requires:
- the saved RF model from part3_random_forest.py
- metadata JSON describing which feature columns were used
- handcrafted features from the unlabeled npz
- optional AE feature npz for unlabeled images if best model uses ae_* columns
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part3_data_utils import load_unlabeled_dataframe


def load_unlabeled_ae_dataframe(ae_feature_npz: str) -> pd.DataFrame:
    data = np.load(ae_feature_npz, allow_pickle=True)
    X = data["X"]
    groups = data["groups"].astype(int)
    image_names = data["image_names"]
    df = pd.DataFrame(X, columns=[f"ae_{i}" for i in range(X.shape[1])])
    df["group_id"] = groups
    df["image_name"] = [image_names[g] for g in groups]
    return df


def save_probability_map(df_img: pd.DataFrame, image_name: str, outpath: str):
    ys = df_img["y_coord"].to_numpy()
    xs = df_img["x_coord"].to_numpy()
    probs = df_img["cloud_prob"].to_numpy()

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    canvas = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan, dtype=np.float32)
    canvas[ys - y0, xs - x0] = probs

    plt.figure(figsize=(6, 5))
    plt.imshow(canvas)
    plt.colorbar(label="Predicted cloud probability")
    plt.title(image_name)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--unlabeled-paths", nargs="+", required=True)
    parser.add_argument("--outdir", default="results/part3_random_forest/unlabeled_predictions")
    parser.add_argument("--unlabeled-ae-features", default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model = joblib.load(args.model)
    with open(args.metadata, "r") as f:
        metadata = json.load(f)
    feature_cols = metadata["best_feature_cols"]

    df = load_unlabeled_dataframe(args.unlabeled_paths)

    if any(c.startswith("ae_") for c in feature_cols):
        if args.unlabeled_ae_features is None:
            raise ValueError("Best model uses AE features; provide --unlabeled-ae-features.")
        ae_df = load_unlabeled_ae_dataframe(args.unlabeled_ae_features)
        if len(df) != len(ae_df):
            raise ValueError(f"Row mismatch: handcrafted={len(df)} ae={len(ae_df)}")
        for c in [col for col in ae_df.columns if col.startswith("ae_")]:
            df[c] = ae_df[c].to_numpy()

    X = df[feature_cols].to_numpy(dtype=np.float32)
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    df["cloud_prob"] = probs
    df["cloud_pred"] = preds

    df.to_csv(os.path.join(args.outdir, "unlabeled_predictions.csv"), index=False)

    for image_name, df_img in df.groupby("image_name"):
        safe_name = os.path.splitext(image_name)[0]
        save_probability_map(df_img, image_name, os.path.join(args.outdir, f"{safe_name}_probability_map.png"))

    print("Saved unlabeled predictions to", args.outdir)


if __name__ == "__main__":
    main()
