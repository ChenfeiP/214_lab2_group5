"""Train and evaluate Random Forest models for Part 3.

This script compares three feature sets required by your workflow:
1. handcrafted only
2. autoencoder latent features only
3. handcrafted + autoencoder features

It uses GroupKFold with image_id groups, which is important because the lab
only has 3 labeled images and patch-level random splits would leak spatially
related patches across train/test. See lab instructions for the emphasis on
careful split choice and using autoencoder features. fileciteturn4file1 fileciteturn4file4
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

from part3_data_utils import HANDCRAFTED_DEFAULT, assemble_feature_sets


@dataclass
class FoldMetrics:
    fold: int
    feature_set: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    n_train: int
    n_test: int
    held_out_group: int


def build_candidate_models(random_state: int) -> Dict[str, RandomForestClassifier]:
    return {
        "rf_small": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "rf_medium": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "rf_regularized": RandomForestClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_one_model(X, y, groups, model, feature_set_name: str) -> List[FoldMetrics]:
    gkf = GroupKFold(n_splits=3)
    metrics: List[FoldMetrics] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_test = groups[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        held_out_group = int(np.unique(g_test)[0])
        metrics.append(
            FoldMetrics(
                fold=fold_idx,
                feature_set=feature_set_name,
                accuracy=float(accuracy_score(y_test, y_pred)),
                precision=float(precision_score(y_test, y_pred, zero_division=0)),
                recall=float(recall_score(y_test, y_pred, zero_division=0)),
                f1=float(f1_score(y_test, y_pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y_test, y_prob)),
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                held_out_group=held_out_group,
            )
        )
    return metrics


def save_roc_plot(model, X, y, groups, title: str, outpath: str):
    gkf = GroupKFold(n_splits=3)
    plt.figure(figsize=(6, 5))
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        aucs.append(auc)
        plt.plot(fpr, tpr, label=f"Fold {fold_idx} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC: {title} (mean AUC={np.mean(aucs):.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_feature_importance_plot(model, X, y, feature_names: List[str], outpath: str):
    model.fit(X, y)
    importances = pd.DataFrame({
        "feature": feature_names,
        "mdi_importance": model.feature_importances_,
    }).sort_values("mdi_importance", ascending=False)

    top = importances.head(min(20, len(importances))).iloc[::-1]
    plt.figure(figsize=(7, max(4, 0.35 * len(top))))
    plt.barh(top["feature"], top["mdi_importance"])
    plt.xlabel("MDI importance")
    plt.title("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    csv_path = os.path.splitext(outpath)[0] + ".csv"
    importances.to_csv(csv_path, index=False)


def save_permutation_importance_plot(model, X, y, feature_names: List[str], outpath: str, random_state: int):
    model.fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": feature_names,
        "perm_mean": result.importances_mean,
        "perm_std": result.importances_std,
    }).sort_values("perm_mean", ascending=False)

    top = imp.head(min(20, len(imp))).iloc[::-1]
    plt.figure(figsize=(7, max(4, 0.35 * len(top))))
    plt.barh(top["feature"], top["perm_mean"], xerr=top["perm_std"])
    plt.xlabel("Permutation importance (AUC drop)")
    plt.title("Permutation importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    csv_path = os.path.splitext(outpath)[0] + ".csv"
    imp.to_csv(csv_path, index=False)


def save_confusion_matrices(model, X, y, groups, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    gkf = GroupKFold(n_splits=3)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4, 4))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix Fold {fold_idx}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"confusion_fold_{fold_idx}.png"), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-features", required=True, help="Path to results/part3_latent_vectors.npz")
    parser.add_argument("--labeled-paths", nargs="+", required=True, help="Three labeled .npz image paths")
    parser.add_argument("--outdir", default="results/part3_random_forest")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--handcrafted-features", nargs="*", default=HANDCRAFTED_DEFAULT)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df, feature_sets = assemble_feature_sets(
        labeled_paths=args.labeled_paths,
        ae_feature_npz=args.ae_features,
        handcrafted_features=args.handcrafted_features,
    )

    y = df["y_binary"].to_numpy()
    groups = df["group_id"].to_numpy()

    candidate_models = build_candidate_models(args.random_state)

    all_metrics = []
    summary_rows = []
    best_score = -np.inf
    best_payload = None

    for feature_set_name, feature_cols in feature_sets.items():
        X = df[feature_cols].to_numpy(dtype=np.float32)
        for model_name, model in candidate_models.items():
            fold_metrics = evaluate_one_model(X, y, groups, model, feature_set_name)
            all_metrics.extend([asdict(m) | {"model_name": model_name} for m in fold_metrics])
            mean_auc = float(np.mean([m.roc_auc for m in fold_metrics]))
            mean_f1 = float(np.mean([m.f1 for m in fold_metrics]))
            summary_rows.append({
                "model_name": model_name,
                "feature_set": feature_set_name,
                "mean_roc_auc": mean_auc,
                "mean_f1": mean_f1,
                "mean_accuracy": float(np.mean([m.accuracy for m in fold_metrics])),
                "mean_precision": float(np.mean([m.precision for m in fold_metrics])),
                "mean_recall": float(np.mean([m.recall for m in fold_metrics])),
            })

            if mean_auc > best_score:
                best_score = mean_auc
                best_payload = {
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "feature_cols": feature_cols,
                    "model": model,
                }

    fold_df = pd.DataFrame(all_metrics)
    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_roc_auc", "mean_f1"], ascending=False)
    fold_df.to_csv(os.path.join(args.outdir, "rf_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(args.outdir, "rf_model_summary.csv"), index=False)

    if best_payload is None:
        raise RuntimeError("No model evaluated.")

    best_model = best_payload["model"]
    best_feature_cols = best_payload["feature_cols"]
    best_feature_set = best_payload["feature_set"]
    best_model_name = best_payload["model_name"]
    X_best = df[best_feature_cols].to_numpy(dtype=np.float32)

    save_roc_plot(
        best_model,
        X_best,
        y,
        groups,
        title=f"{best_model_name} + {best_feature_set}",
        outpath=os.path.join(args.outdir, "best_model_roc.png"),
    )
    save_feature_importance_plot(
        best_model,
        X_best,
        y,
        best_feature_cols,
        outpath=os.path.join(args.outdir, "best_model_mdi_importance.png"),
    )
    save_permutation_importance_plot(
        best_model,
        X_best,
        y,
        best_feature_cols,
        outpath=os.path.join(args.outdir, "best_model_permutation_importance.png"),
        random_state=args.random_state,
    )
    save_confusion_matrices(
        best_model,
        X_best,
        y,
        groups,
        outdir=os.path.join(args.outdir, "confusion_matrices"),
    )

    # Fit final model on all labeled data and save it.
    best_model.fit(X_best, y)
    joblib.dump(best_model, os.path.join(args.outdir, "best_random_forest.joblib"))

    metadata = {
        "best_model_name": best_model_name,
        "best_feature_set": best_feature_set,
        "best_feature_cols": best_feature_cols,
        "random_state": args.random_state,
        "labeled_paths": args.labeled_paths,
        "summary_top_row": summary_df.iloc[0].to_dict(),
    }
    with open(os.path.join(args.outdir, "best_model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved results to", args.outdir)
    print(summary_df)


if __name__ == "__main__":
    main()
