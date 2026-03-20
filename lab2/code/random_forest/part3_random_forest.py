import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
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

from part3_data_utils import (
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
PALETTE_SOFT = sns.color_palette("muted")


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


def build_candidate_models(random_state: int) -> Dict[str, RandomForestClassifier]:
    return {
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

        fold_metrics = FoldMetrics(
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
        metrics.append(fold_metrics)

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
    X,
    y,
    groups,
    model,
    feature_set_name: str,
    model_name: str,
    df_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[FoldMetrics]]:
    print(f"\n[INFO] Evaluating model={model_name}, feature_set={feature_set_name}")
    pred_df, metrics = get_oof_predictions(model, X, y, groups, df_meta=df_meta)

    for i, m in enumerate(metrics, start=1):
        m.feature_set = feature_set_name
        m.model_name = model_name
        print(
            f"[INFO]  Fold {i}/3 | held_out_group={m.held_out_group} | "
            f"acc={m.accuracy:.4f}, f1={m.f1:.4f}, auc={m.roc_auc:.4f}"
        )

    print(f"[INFO] Finished model={model_name}, feature_set={feature_set_name}")
    return pred_df, metrics


def save_roc_plot_from_oof(pred_df: pd.DataFrame, title: str, outpath: str):
    print(f"[INFO] Saving ROC plot to {outpath}")
    roc_rows = []
    aucs = []

    for fold, g in pred_df.groupby("fold"):
        fpr, tpr, _ = roc_curve(g["y_true"], g["y_prob"])
        auc = roc_auc_score(g["y_true"], g["y_prob"])
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
    ax.set_title(f"ROC Curves for Best Random Forest\nMean AUC = {np.mean(aucs):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title=None, loc="lower right", frameon=False)
    apply_publication_axes_style(ax, add_grid=False)
    finalize_figure(outpath)


def save_feature_importance_plot(model, X, y, feature_names: List[str], outpath: str):
    print(f"[INFO] Saving MDI feature importance to {outpath}")
    model = clone(model)
    model.fit(X, y)

    importances = pd.DataFrame({
        "feature": feature_names,
        "mdi_importance": model.feature_importances_,
    }).sort_values("mdi_importance", ascending=False)

    importances["display_feature"] = importances["feature"].astype(str)
    top = importances.head(min(20, len(importances))).sort_values("mdi_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7.2, max(4.8, 0.34 * len(top))))
    sns.barplot(
        data=top,
        x="mdi_importance",
        y="display_feature",
        color=PALETTE_MAIN[0],
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_xlabel("MDI Importance")
    ax.set_ylabel("")
    ax.set_title("Feature Importance from Random Forest (MDI)")
    apply_publication_axes_style(ax, add_grid=True)
    finalize_figure(outpath)

    csv_path = os.path.splitext(outpath)[0] + ".csv"
    importances.drop(columns=["display_feature"]).to_csv(csv_path, index=False)


def save_permutation_importance_plot(model, X, y, feature_names: List[str], outpath: str, random_state: int):
    print(f"[INFO] Saving permutation importance to {outpath}")
    model = clone(model)
    model.fit(X, y)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=1,
    )

    imp = pd.DataFrame({
        "feature": feature_names,
        "perm_mean": result.importances_mean,
        "perm_std": result.importances_std,
    }).sort_values("perm_mean", ascending=False)

    imp["display_feature"] = imp["feature"].astype(str)
    top = imp.head(min(20, len(imp))).sort_values("perm_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7.2, max(4.8, 0.34 * len(top))))
    ax.barh(
        top["display_feature"],
        top["perm_mean"],
        xerr=top["perm_std"],
        color=PALETTE_MAIN[1],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.8, "capsize": 2.5},
    )
    ax.set_xlabel("Permutation Importance (Decrease in AUC)")
    ax.set_ylabel("")
    ax.set_title("Permutation Feature Importance")
    apply_publication_axes_style(ax, add_grid=True)
    finalize_figure(outpath)

    csv_path = os.path.splitext(outpath)[0] + ".csv"
    imp.drop(columns=["display_feature"]).to_csv(csv_path, index=False)


def save_confusion_matrices_from_oof(pred_df: pd.DataFrame, outdir: str):
    print(f"[INFO] Saving confusion matrices to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    for fold, g in pred_df.groupby("fold"):
        cm = confusion_matrix(g["y_true"], g["y_pred"])

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


def save_error_analysis(
    pred_df: pd.DataFrame,
    full_df: pd.DataFrame,
    feature_cols: List[str],
    outdir: str,
):
    print(f"[INFO] Saving error analysis to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    errors = pred_df[pred_df["correct"] == 0].copy()
    errors.to_csv(os.path.join(outdir, "misclassified_rows.csv"), index=False)

    error_by_image = (
        pred_df.groupby("image_name")["correct"]
        .apply(lambda s: float(1.0 - s.mean()))
        .reset_index(name="error_rate")
        .sort_values("error_rate", ascending=False)
    )
    error_by_image.to_csv(os.path.join(outdir, "error_rate_by_image.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    sns.barplot(
        data=error_by_image,
        x="image_name",
        y="error_rate",
        color=PALETTE_MAIN[2],
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_ylabel("Misclassification Rate")
    ax.set_xlabel("")
    ax.set_title("Misclassification Rate by Labeled Image")
    plt.xticks(rotation=25, ha="right")
    apply_publication_axes_style(ax, add_grid=True)
    finalize_figure(os.path.join(outdir, "error_rate_by_image.png"))

    conf_df = pred_df.copy()
    conf_df["prediction_status"] = conf_df["correct"].map({1: "Correct", 0: "Misclassified"})

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sns.kdeplot(
        data=conf_df,
        x="y_prob",
        hue="prediction_status",
        common_norm=False,
        fill=False,
        linewidth=1.8,
        palette=[PALETTE_MAIN[3], PALETTE_MAIN[0]],
        ax=ax,
    )
    ax.set_xlabel("Predicted Probability of Cloud")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence by Classification Outcome")
    apply_publication_axes_style(ax, add_grid=False)
    finalize_figure(os.path.join(outdir, "confidence_correct_vs_error.png"))

    merged = pred_df.merge(
        full_df[["image_name", "group_id", "y_coord", "x_coord"] + feature_cols],
        on=["image_name", "group_id", "y_coord", "x_coord"],
        how="left",
    )

    numeric_candidates = [c for c in feature_cols if c in merged.columns]
    top_features = numeric_candidates[: min(6, len(numeric_candidates))]

    summary_rows = []
    for feat in top_features:
        grp = merged.groupby("correct")[feat].agg(["mean", "std", "median"]).reset_index()
        grp["feature"] = feat
        summary_rows.append(grp)

    if summary_rows:
        pd.concat(summary_rows, axis=0, ignore_index=True).to_csv(
            os.path.join(outdir, "feature_distribution_correct_vs_error.csv"),
            index=False,
        )

    for feat in top_features:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        sns.boxplot(
            data=merged,
            x="correct",
            y=feat,
            palette=[PALETTE_SOFT[3], PALETTE_SOFT[0]],
            width=0.55,
            fliersize=2.5,
            linewidth=0.8,
            ax=ax,
        )
        ax.set_xticklabels(["Misclassified", "Correct"])
        ax.set_xlabel("")
        ax.set_title(f"{feat} Distribution: Correct vs Misclassified")
        apply_publication_axes_style(ax, add_grid=False)
        finalize_figure(os.path.join(outdir, f"error_boxplot_{feat}.png"))


def save_stability_analysis(
    best_model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outdir: str,
    base_random_state: int,
):
    print(f"[INFO] Saving stability analysis to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    seeds = [base_random_state + i for i in range(10)]
    rows = []

    for seed in seeds:
        model = clone(best_model)
        model.set_params(random_state=seed)

        gkf = GroupKFold(n_splits=3)
        fold_aucs = []
        fold_f1s = []
        fold_accs = []

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            fold_aucs.append(roc_auc_score(y_test, y_prob))
            fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))
            fold_accs.append(accuracy_score(y_test, y_pred))

        rows.append({
            "seed": seed,
            "mean_auc": float(np.mean(fold_aucs)),
            "mean_f1": float(np.mean(fold_f1s)),
            "mean_accuracy": float(np.mean(fold_accs)),
        })

    stab_df = pd.DataFrame(rows)
    stab_df.to_csv(os.path.join(outdir, "rf_seed_stability.csv"), index=False)

    stab_long = stab_df.melt(
        id_vars="seed",
        value_vars=["mean_auc", "mean_f1", "mean_accuracy"],
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sns.boxplot(
        data=stab_long,
        x="metric",
        y="score",
        palette=PALETTE_SOFT[:3],
        width=0.55,
        linewidth=0.8,
        fliersize=2.5,
        ax=ax,
    )
    sns.stripplot(
        data=stab_long,
        x="metric",
        y="score",
        color="black",
        size=3,
        alpha=0.55,
        jitter=0.12,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cross-validated score")
    ax.set_title("Stability Across Random Seeds")
    ax.set_xticklabels(["AUC", "F1", "Accuracy"])
    apply_publication_axes_style(ax, add_grid=True)
    finalize_figure(os.path.join(outdir, "rf_seed_stability_boxplot.png"))


def save_perturbation_analysis(
    best_model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    outdir: str,
    random_state: int,
):
    print(f"[INFO] Saving perturbation analysis to {outdir}")
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(random_state)
    model = clone(best_model)
    model.fit(X, y)

    base_prob = model.predict_proba(X)[:, 1]
    base_pred = (base_prob >= 0.5).astype(int)

    feature_std = X.std(axis=0, ddof=0)
    noise_levels = [0.01, 0.03, 0.05, 0.10]
    rows = []

    for eps in noise_levels:
        noise = rng.normal(loc=0.0, scale=eps * np.maximum(feature_std, 1e-8), size=X.shape)
        X_pert = X + noise
        pert_prob = model.predict_proba(X_pert)[:, 1]
        pert_pred = (pert_prob >= 0.5).astype(int)

        flip_rate = float(np.mean(base_pred != pert_pred))
        mean_abs_prob_shift = float(np.mean(np.abs(base_prob - pert_prob)))
        corr = float(np.corrcoef(base_prob, pert_prob)[0, 1])

        rows.append({
            "noise_scale_fraction_of_std": eps,
            "flip_rate": flip_rate,
            "mean_abs_prob_shift": mean_abs_prob_shift,
            "prob_correlation": corr,
        })

    pert_df = pd.DataFrame(rows)
    pert_df.to_csv(os.path.join(outdir, "perturbation_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sns.lineplot(
        data=pert_df,
        x="noise_scale_fraction_of_std",
        y="flip_rate",
        marker="o",
        linewidth=1.8,
        markersize=5,
        color=PALETTE_MAIN[4],
        ax=ax,
    )
    ax.set_ylabel("Prediction Flip Rate")
    ax.set_xlabel("Noise scale relative to feature SD")
    ax.set_title("Prediction Stability Under Input Perturbation")
    apply_publication_axes_style(ax, add_grid=True)
    finalize_figure(os.path.join(outdir, "perturbation_flip_rate.png"))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sns.lineplot(
        data=pert_df,
        x="noise_scale_fraction_of_std",
        y="mean_abs_prob_shift",
        marker="o",
        linewidth=1.8,
        markersize=5,
        color=PALETTE_MAIN[5],
        ax=ax,
    )
    ax.set_ylabel("Mean Absolute Probability Shift")
    ax.set_xlabel("Noise scale relative to feature SD")
    ax.set_title("Probability Shift Under Input Perturbation")
    apply_publication_axes_style(ax, add_grid=True)
    finalize_figure(os.path.join(outdir, "perturbation_prob_shift.png"))


def save_unlabeled_predictions(
    best_model: RandomForestClassifier,
    best_feature_set: str,
    best_feature_cols: List[str],
    outdir: str,
    unlabeled_paths: List[str],
    handcrafted_features: List[str],
    unlabeled_ae_features: str = None,
):
    print("[INFO] Running sanity-check prediction on unlabeled images...")
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

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sns.barplot(
        data=summary,
        x="image_name",
        y="predicted_cloud_fraction",
        color=PALETTE_MAIN[0],
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )
    plt.xticks(rotation=25, ha="right")
    ax.set_ylabel("Fraction Predicted as Cloud")
    ax.set_xlabel("")
    ax.set_title("Sanity Check Predictions on Unlabeled Images")
    apply_publication_axes_style(ax, add_grid=True)
    finalize_figure(os.path.join(unlabeled_dir, "unlabeled_cloud_fraction.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-features", required=True, help="Path to labeled AE latent vectors .npz")
    parser.add_argument("--labeled-paths", nargs="+", required=True, help="Three labeled .npz image paths")
    parser.add_argument("--outdir", default="results/part3_random_forest")
    parser.add_argument("--random-state", type=int, default=42)
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
    print(f"       random_state        = {args.random_state}")
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

    print(f"[INFO] DataFrame shape = {df.shape}")
    print(f"[INFO] Feature sets = {list(feature_sets.keys())}")
    print(f"[INFO] Label distribution = {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"[INFO] Group distribution = {dict(zip(*np.unique(groups, return_counts=True)))}")

    candidate_models = build_candidate_models(args.random_state)

    all_metrics = []
    summary_rows = []
    best_score = -np.inf
    best_payload = None

    meta_cols = ["image_name", "group_id", "y_coord", "x_coord"]

    print("\n[INFO] Starting model comparison...")
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

            for m in fold_metrics:
                all_metrics.append(asdict(m))

            mean_auc = float(np.mean([m.roc_auc for m in fold_metrics]))
            mean_f1 = float(np.mean([m.f1 for m in fold_metrics]))
            mean_acc = float(np.mean([m.accuracy for m in fold_metrics]))
            mean_prec = float(np.mean([m.precision for m in fold_metrics]))
            mean_rec = float(np.mean([m.recall for m in fold_metrics]))

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
    best_pred_df = best_payload["pred_df"]

    X_best = df[best_feature_cols].to_numpy(dtype=np.float32)

    print("\n[INFO] Best model selected:")
    print(f"       model_name   = {best_model_name}")
    print(f"       feature_set  = {best_feature_set}")
    print(f"       num_features = {len(best_feature_cols)}")
    print(f"       best_score   = {best_score:.4f}")

    best_pred_df.to_csv(os.path.join(args.outdir, "best_model_oof_predictions.csv"), index=False)

    save_roc_plot_from_oof(
        best_pred_df,
        title=f"{best_model_name} + {best_feature_set}",
        outpath=os.path.join(args.outdir, "best_model_roc.png"),
    )
    save_confusion_matrices_from_oof(
        best_pred_df,
        outdir=os.path.join(args.outdir, "confusion_matrices"),
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
    save_error_analysis(
        pred_df=best_pred_df,
        full_df=df,
        feature_cols=best_feature_cols,
        outdir=os.path.join(args.outdir, "error_analysis"),
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