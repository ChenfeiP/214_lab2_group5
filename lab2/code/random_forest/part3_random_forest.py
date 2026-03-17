"""Train and evaluate Random Forest models for Part 3."""

import argparse
import json
import os
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")
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
            n_jobs=1,
        ),
        "rf_medium": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "rf_regularized": RandomForestClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def evaluate_one_model(X, y, groups, model, feature_set_name: str, model_name: str) -> List[FoldMetrics]:
    print(f"\n[INFO] Evaluating model={model_name}, feature_set={feature_set_name}")
    gkf = GroupKFold(n_splits=3)
    metrics: List[FoldMetrics] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(
            f"[INFO]  Fold {fold_idx}/3 | "
            f"train={len(train_idx)} test={len(test_idx)} | "
            f"held_out_group={int(np.unique(groups[test_idx])[0])}"
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_test = groups[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        held_out_group = int(np.unique(g_test)[0])
        fold_result = FoldMetrics(
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
        metrics.append(fold_result)

        print(
            f"[INFO]   Fold {fold_idx} done | "
            f"acc={fold_result.accuracy:.4f}, "
            f"f1={fold_result.f1:.4f}, "
            f"auc={fold_result.roc_auc:.4f}"
        )

    print(f"[INFO] Finished model={model_name}, feature_set={feature_set_name}")
    return metrics


def save_roc_plot(model, X, y, groups, title: str, outpath: str):
    print(f"[INFO] Saving ROC plot to {outpath}")
    gkf = GroupKFold(n_splits=3)

    roc_rows = []
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        aucs.append(auc)

        roc_rows.append(
            pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "fold": f"Fold {fold_idx} (AUC={auc:.3f})",
            })
        )

    roc_df = pd.concat(roc_rows, ignore_index=True)

    plt.figure(figsize=(7, 6))
    sns.lineplot(data=roc_df, x="fpr", y="tpr", hue="fold", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {title}\nMean AUC = {np.mean(aucs):.3f}")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def save_feature_importance_plot(model, X, y, feature_names: List[str], outpath: str):
    print(f"[INFO] Saving MDI feature importance to {outpath}")
    model.fit(X, y)

    importances = pd.DataFrame({
        "feature": feature_names,
        "mdi_importance": model.feature_importances_,
    }).sort_values("mdi_importance", ascending=False)

    top = importances.head(min(20, len(importances))).sort_values("mdi_importance", ascending=True)

    plt.figure(figsize=(8, max(5, 0.4 * len(top))))
    sns.barplot(data=top, x="mdi_importance", y="feature", orient="h")
    plt.xlabel("MDI Importance")
    plt.ylabel("")
    plt.title("Random Forest Feature Importance (MDI)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()

    csv_path = os.path.splitext(outpath)[0] + ".csv"
    importances.to_csv(csv_path, index=False)


def save_permutation_importance_plot(model, X, y, feature_names: List[str], outpath: str, random_state: int):
    print(f"[INFO] Saving permutation importance to {outpath}")
    model.fit(X, y)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=3,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=1,
    )

    imp = pd.DataFrame({
        "feature": feature_names,
        "perm_mean": result.importances_mean,
        "perm_std": result.importances_std,
    }).sort_values("perm_mean", ascending=False)

    top = imp.head(min(20, len(imp))).sort_values("perm_mean", ascending=True)

    plt.figure(figsize=(8, max(5, 0.4 * len(top))))
    plt.barh(
        top["feature"],
        top["perm_mean"],
        xerr=top["perm_std"],
        alpha=0.9,
        capsize=3,
    )
    plt.xlabel("Permutation Importance (AUC Drop)")
    plt.ylabel("")
    plt.title("Permutation Importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()

    csv_path = os.path.splitext(outpath)[0] + ".csv"
    imp.to_csv(csv_path, index=False)


def save_confusion_matrices(model, X, y, groups, outdir: str):
    print(f"[INFO] Saving confusion matrices to {outdir}")
    os.makedirs(outdir, exist_ok=True)
    gkf = GroupKFold(n_splits=3)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"[INFO]  Building confusion matrix for fold {fold_idx}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            square=True,
            xticklabels=["Non-cloud", "Cloud"],
            yticklabels=["Non-cloud", "Cloud"],
        )
        plt.title(f"Confusion Matrix - Fold {fold_idx}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"confusion_fold_{fold_idx}.png"), dpi=220, bbox_inches="tight")
        plt.close()


def main():
    print("[INFO] Parsing command-line arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-features", required=True, help="Path to results/part3_latent_vectors.npz")
    parser.add_argument("--labeled-paths", nargs="+", required=True, help="Three labeled .npz image paths")
    parser.add_argument("--outdir", default="results/part3_random_forest")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--handcrafted-features", nargs="*", default=HANDCRAFTED_DEFAULT)
    args = parser.parse_args()

    print("[INFO] Arguments:")
    print(f"       ae_features        = {args.ae_features}")
    print(f"       labeled_paths      = {args.labeled_paths}")
    print(f"       outdir             = {args.outdir}")
    print(f"       random_state       = {args.random_state}")
    print(f"       handcrafted_feats  = {args.handcrafted_features}")

    os.makedirs(args.outdir, exist_ok=True)

    print("\n[INFO] Assembling handcrafted / AE / combined feature sets...")
    df, feature_sets = assemble_feature_sets(
        labeled_paths=args.labeled_paths,
        ae_feature_npz=args.ae_features,
        handcrafted_features=args.handcrafted_features,
    )

    print("[INFO] Feature assembly complete.")
    print(f"       DataFrame shape = {df.shape}")
    print(f"       Feature sets    = {list(feature_sets.keys())}")

    y = df["y_binary"].to_numpy()
    groups = df["group_id"].to_numpy()

    print("[INFO] Label distribution:")
    unique_y, count_y = np.unique(y, return_counts=True)
    print(f"       {dict(zip(unique_y.tolist(), count_y.tolist()))}")

    print("[INFO] Group distribution:")
    unique_g, count_g = np.unique(groups, return_counts=True)
    print(f"       {dict(zip(unique_g.tolist(), count_g.tolist()))}")

    print("\n[INFO] Building candidate Random Forest models...")
    candidate_models = build_candidate_models(args.random_state)
    print(f"[INFO] Candidate models: {list(candidate_models.keys())}")

    all_metrics = []
    summary_rows = []
    best_score = -np.inf
    best_payload = None

    print("\n[INFO] Starting model comparison...")
    for feature_set_name, feature_cols in feature_sets.items():
        print(f"\n[INFO] Working on feature_set={feature_set_name}")
        print(f"[INFO] Number of features = {len(feature_cols)}")
        X = df[feature_cols].to_numpy(dtype=np.float32)

        for model_name, model in candidate_models.items():
            fold_metrics = evaluate_one_model(X, y, groups, model, feature_set_name, model_name)
            all_metrics.extend([asdict(m) | {"model_name": model_name} for m in fold_metrics])

            mean_auc = float(np.mean([m.roc_auc for m in fold_metrics]))
            mean_f1 = float(np.mean([m.f1 for m in fold_metrics]))
            mean_acc = float(np.mean([m.accuracy for m in fold_metrics]))
            mean_prec = float(np.mean([m.precision for m in fold_metrics]))
            mean_rec = float(np.mean([m.recall for m in fold_metrics]))

            print(
                f"[INFO] Summary | model={model_name}, feature_set={feature_set_name}, "
                f"mean_auc={mean_auc:.4f}, mean_f1={mean_f1:.4f}, mean_acc={mean_acc:.4f}"
            )

            summary_rows.append({
                "model_name": model_name,
                "feature_set": feature_set_name,
                "mean_roc_auc": mean_auc,
                "mean_f1": mean_f1,
                "mean_accuracy": mean_acc,
                "mean_precision": mean_prec,
                "mean_recall": mean_rec,
            })

            if mean_auc > best_score:
                best_score = mean_auc
                best_payload = {
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "feature_cols": feature_cols,
                    "model": model,
                }
                print(
                    f"[INFO] New best model found: {model_name} + {feature_set_name} "
                    f"(mean_auc={mean_auc:.4f})"
                )

    print("\n[INFO] Saving fold-level and summary CSV files...")
    fold_df = pd.DataFrame(all_metrics)
    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_roc_auc", "mean_f1"], ascending=False)
    fold_csv = os.path.join(args.outdir, "rf_fold_metrics.csv")
    summary_csv = os.path.join(args.outdir, "rf_model_summary.csv")
    fold_df.to_csv(fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[INFO] Saved {fold_csv}")
    print(f"[INFO] Saved {summary_csv}")

    if best_payload is None:
        raise RuntimeError("No model evaluated.")

    best_model = best_payload["model"]
    best_feature_cols = best_payload["feature_cols"]
    best_feature_set = best_payload["feature_set"]
    best_model_name = best_payload["model_name"]
    X_best = df[best_feature_cols].to_numpy(dtype=np.float32)

    print("\n[INFO] Best model selected:")
    print(f"       model_name   = {best_model_name}")
    print(f"       feature_set  = {best_feature_set}")
    print(f"       num_features = {len(best_feature_cols)}")
    print(f"       best_score   = {best_score:.4f}")

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

    print("\n[INFO] Fitting best model on all labeled data...")
    best_model.fit(X_best, y)

    model_path = os.path.join(args.outdir, "best_random_forest.joblib")
    joblib.dump(best_model, model_path)
    print(f"[INFO] Saved trained best model to {model_path}")

    metadata = {
        "best_model_name": best_model_name,
        "best_feature_set": best_feature_set,
        "best_feature_cols": best_feature_cols,
        "random_state": args.random_state,
        "labeled_paths": args.labeled_paths,
        "summary_top_row": summary_df.iloc[0].to_dict(),
    }
    metadata_path = os.path.join(args.outdir, "best_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata to {metadata_path}")

    print("\n[INFO] Finished.")
    print(f"[INFO] All results saved to: {args.outdir}")
    print("\n[INFO] Final summary table:")
    print(summary_df)


if __name__ == "__main__":
    main()