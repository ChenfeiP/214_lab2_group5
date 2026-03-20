"""Train and evaluate LDA models for Part 3."""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from random_forest.part3_data_utils import (
    HANDCRAFTED_DEFAULT,
    assemble_feature_sets,
    assemble_unlabeled_feature_sets,
)

sns.set_theme(context="paper", style="ticks")

mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "figure.figsize": (6.4, 4.8),
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 1.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "legend.frameon": False,
})

PALETTE_MAIN = sns.color_palette("deep")


@dataclass
class FoldMetrics:
    fold: int
    feature_set: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    n_train: int
    n_test: int
    held_out_group: int


def apply_publication_axes_style(ax, add_grid: bool = False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    if add_grid:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    else:
        ax.grid(False)

    return ax


def finalize_figure(outpath: str):
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    base, _ = os.path.splitext(outpath)
    plt.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close()


def build_candidate_models() -> Dict[str, LinearDiscriminantAnalysis]:
    return {
        "lda_shrinkage": LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        ),
    }


def get_oof_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    df_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[FoldMetrics]]:
    gkf = GroupKFold(n_splits=3)
    pred_rows = []
    metrics: List[FoldMetrics] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_test = groups[test_idx]

        local_model = clone(model)
        local_model.fit(X_train, y_train)
        y_pred = local_model.predict(X_test)
        y_prob = local_model.predict_proba(X_test)[:, 1]

        held_out_group = int(np.unique(g_test)[0])

        metrics.append(
            FoldMetrics(
                fold=fold_idx,
                feature_set="",
                model_name="",
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

        fold_df = df_meta.iloc[test_idx].copy().reset_index(drop=True)
        fold_df["fold"] = fold_idx
        fold_df["y_true"] = y_test
        fold_df["y_pred"] = y_pred
        fold_df["y_prob"] = y_prob
        fold_df["correct"] = (y_test == y_pred).astype(int)
        pred_rows.append(fold_df)

    pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)
    return pred_df, metrics


def evaluate_one_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
    feature_set_name: str,
    model_name: str,
    df_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[FoldMetrics]]:
    print(f"\n[INFO] Evaluating model={model_name}, feature_set={feature_set_name}")
    pred_df, metrics = get_oof_predictions(model, X, y, groups, df_meta=df_meta)

    for i, metric in enumerate(metrics, start=1):
        metric.feature_set = feature_set_name
        metric.model_name = model_name
        print(
            f"[INFO]  Fold {i}/3 | held_out_group={metric.held_out_group} | "
            f"acc={metric.accuracy:.4f}, f1={metric.f1:.4f}, auc={metric.roc_auc:.4f}"
        )

    return pred_df, metrics


def save_roc_plot_from_oof(pred_df: pd.DataFrame, outpath: str):
    roc_rows = []
    aucs = []

    for fold, fold_df in pred_df.groupby("fold"):
        fpr, tpr, _ = roc_curve(fold_df["y_true"], fold_df["y_prob"])
        auc = roc_auc_score(fold_df["y_true"], fold_df["y_prob"])
        aucs.append(auc)
        roc_rows.append(
            pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "fold": f"Fold {fold} (AUC={auc:.3f})",
            })
        )

    roc_df = pd.concat(roc_rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sns.lineplot(
        data=roc_df,
        x="fpr",
        y="tpr",
        hue="fold",
        palette="deep",
        linewidth=1.8,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.0, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves for Best LDA\nMean AUC = {np.mean(aucs):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title=None, loc="lower right", frameon=False)
    apply_publication_axes_style(ax, add_grid=False)
    finalize_figure(outpath)


def save_confusion_matrices_from_oof(pred_df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    for fold, fold_df in pred_df.groupby("fold"):
        cm = confusion_matrix(fold_df["y_true"], fold_df["y_pred"])

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=["Non-cloud", "Cloud"],
            yticklabels=["Non-cloud", "Cloud"],
            annot_kws={"size": 10, "weight": "semibold"},
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix (Fold {fold})")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        finalize_figure(os.path.join(outdir, f"confusion_fold_{fold}.png"))


def save_coefficient_plot(model, X: np.ndarray, y: np.ndarray, feature_names: List[str], outpath: str):
    model = clone(model)
    model.fit(X, y)

    coefs = np.ravel(model.coef_)
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    top = coef_df.head(min(20, len(coef_df))).sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(7.2, max(4.8, 0.34 * len(top))))
    colors = [PALETTE_MAIN[0] if value >= 0 else PALETTE_MAIN[3] for value in top["coefficient"]]
    ax.barh(
        top["feature"],
        top["coefficient"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axvline(0.0, color="0.4", linewidth=1.0)
    ax.set_xlabel("LDA coefficient")
    ax.set_ylabel("")
    ax.set_title("LDA Coefficients for Best Feature Set")
    apply_publication_axes_style(ax, add_grid=True)
    finalize_figure(outpath)

    csv_path = os.path.splitext(outpath)[0] + ".csv"
    coef_df.to_csv(csv_path, index=False)


def save_unlabeled_predictions(
    best_model,
    best_feature_set: str,
    best_feature_cols: List[str],
    outdir: str,
    unlabeled_paths: List[str],
    handcrafted_features: List[str],
    unlabeled_ae_features: str = None,
):
    unlabeled_dir = os.path.join(outdir, "unlabeled_sanity_check")
    os.makedirs(unlabeled_dir, exist_ok=True)

    unlabeled_df, unlabeled_feature_sets = assemble_unlabeled_feature_sets(
        unlabeled_paths=unlabeled_paths,
        handcrafted_features=handcrafted_features,
        ae_feature_npz=unlabeled_ae_features,
    )

    if best_feature_set not in unlabeled_feature_sets:
        print(
            f"[WARN] Skipping unlabeled prediction because feature set '{best_feature_set}' "
            f"is unavailable for unlabeled data."
        )
        return

    X_unlabeled = unlabeled_df[best_feature_cols].to_numpy(dtype=np.float32)
    pred_prob = best_model.predict_proba(X_unlabeled)[:, 1]
    pred_label = (pred_prob >= 0.5).astype(int)

    out = unlabeled_df[["image_name", "group_id", "y_coord", "x_coord"]].copy()
    out["pred_cloud"] = pred_label
    out["pred_prob_cloud"] = pred_prob
    out.to_csv(os.path.join(unlabeled_dir, "unlabeled_predictions.csv"), index=False)

    summary = (
        out.groupby("image_name")
        .agg(
            n_pixels=("pred_cloud", "size"),
            predicted_cloud_fraction=("pred_cloud", "mean"),
            mean_cloud_probability=("pred_prob_cloud", "mean"),
            std_cloud_probability=("pred_prob_cloud", "std"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(unlabeled_dir, "unlabeled_prediction_summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-features", required=True, help="Path to labeled AE latent vectors .npz")
    parser.add_argument("--labeled-paths", nargs="+", required=True, help="Three labeled .npz image paths")
    parser.add_argument("--outdir", default=os.path.join(SCRIPT_DIR, "results", "part3_lda"))
    parser.add_argument("--handcrafted-features", nargs="*", default=HANDCRAFTED_DEFAULT)
    parser.add_argument("--unlabeled-paths", nargs="*", default=None, help="Optional unlabeled image paths")
    parser.add_argument(
        "--unlabeled-ae-features",
        default=None,
        help="Optional AE latent vectors for unlabeled images if best model uses AE features",
    )
    args = parser.parse_args()

    print("[INFO] Arguments:")
    print(f"       ae_features         = {args.ae_features}")
    print(f"       labeled_paths       = {args.labeled_paths}")
    print(f"       outdir              = {args.outdir}")
    print(f"       handcrafted_feats   = {args.handcrafted_features}")
    print(f"       unlabeled_paths     = {args.unlabeled_paths}")
    print(f"       unlabeled_ae_feats  = {args.unlabeled_ae_features}")

    os.makedirs(args.outdir, exist_ok=True)

    print("\n[INFO] Assembling handcrafted / AE / combined feature sets...")
    df, feature_sets = assemble_feature_sets(
        labeled_paths=args.labeled_paths,
        ae_feature_npz=args.ae_features,
        handcrafted_features=args.handcrafted_features,
    )

    y = df["y_binary"].to_numpy()
    groups = df["group_id"].to_numpy()
    meta_cols = ["image_name", "group_id", "y_coord", "x_coord"]

    print(f"[INFO] DataFrame shape = {df.shape}")
    print(f"[INFO] Feature sets = {list(feature_sets.keys())}")
    print(f"[INFO] Label distribution = {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"[INFO] Group distribution = {dict(zip(*np.unique(groups, return_counts=True)))}")

    candidate_models = build_candidate_models()

    all_metrics = []
    summary_rows = []
    best_score = -np.inf
    best_payload = None

    print("\n[INFO] Starting LDA comparison...")
    for feature_set_name, feature_cols in feature_sets.items():
        X = df[feature_cols].to_numpy(dtype=np.float32)
        df_meta = df[meta_cols].copy()

        for model_name, model in candidate_models.items():
            pred_df, fold_metrics = evaluate_one_model(
                X=X,
                y=y,
                groups=groups,
                model=model,
                feature_set_name=feature_set_name,
                model_name=model_name,
                df_meta=df_meta,
            )

            for metric in fold_metrics:
                all_metrics.append(asdict(metric))

            mean_auc = float(np.mean([metric.roc_auc for metric in fold_metrics]))
            mean_f1 = float(np.mean([metric.f1 for metric in fold_metrics]))
            mean_acc = float(np.mean([metric.accuracy for metric in fold_metrics]))
            mean_prec = float(np.mean([metric.precision for metric in fold_metrics]))
            mean_rec = float(np.mean([metric.recall for metric in fold_metrics]))

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
                    "model": clone(model),
                    "pred_df": pred_df.copy(),
                }
                print(
                    f"[INFO] New best model found: {model_name} + {feature_set_name} "
                    f"(mean_auc={mean_auc:.4f})"
                )

    fold_df = pd.DataFrame(all_metrics)
    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_roc_auc", "mean_f1"], ascending=False)

    fold_df.to_csv(os.path.join(args.outdir, "lda_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(args.outdir, "lda_model_summary.csv"), index=False)

    if best_payload is None:
        raise RuntimeError("No LDA model evaluated.")

    best_model = best_payload["model"]
    best_feature_cols = best_payload["feature_cols"]
    best_feature_set = best_payload["feature_set"]
    best_model_name = best_payload["model_name"]
    best_pred_df = best_payload["pred_df"]

    X_best = df[best_feature_cols].to_numpy(dtype=np.float32)

    print("\n[INFO] Best model selected:")
    print(f"       model_name   = {best_model_name}")
    print(f"       feature_set  = {best_feature_set}")
    print(f"       num_features = {len(best_feature_cols)}")
    print(f"       best_score   = {best_score:.4f}")

    best_pred_df.to_csv(os.path.join(args.outdir, "best_lda_oof_predictions.csv"), index=False)
    save_roc_plot_from_oof(best_pred_df, os.path.join(args.outdir, "best_lda_roc.png"))
    save_confusion_matrices_from_oof(best_pred_df, os.path.join(args.outdir, "confusion_matrices"))
    save_coefficient_plot(
        best_model,
        X_best,
        y,
        best_feature_cols,
        os.path.join(args.outdir, "best_lda_coefficients.png"),
    )

    print("\n[INFO] Fitting best LDA on all labeled data...")
    best_model.fit(X_best, y)
    joblib.dump(best_model, os.path.join(args.outdir, "best_lda.joblib"))

    metadata = {
        "best_model_name": best_model_name,
        "best_feature_set": best_feature_set,
        "best_feature_cols": best_feature_cols,
        "labeled_paths": args.labeled_paths,
        "summary_top_row": summary_df.iloc[0].to_dict(),
    }
    with open(os.path.join(args.outdir, "best_lda_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if args.unlabeled_paths:
        save_unlabeled_predictions(
            best_model=best_model,
            best_feature_set=best_feature_set,
            best_feature_cols=best_feature_cols,
            outdir=args.outdir,
            unlabeled_paths=args.unlabeled_paths,
            handcrafted_features=args.handcrafted_features,
            unlabeled_ae_features=args.unlabeled_ae_features,
        )

    print("\n[INFO] Finished.")
    print(f"[INFO] All results saved to: {args.outdir}")
    print("\n[INFO] Final summary table:")
    print(summary_df)


if __name__ == "__main__":
    main()
