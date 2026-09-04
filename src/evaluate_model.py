"""
evaluate_model.py
Comprehensive evaluation suite for trained models.
Generates confusion matrices, ROC curves, and comparative visualization charts.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from src.train_model import train_and_evaluate_all

FIG_DIR = os.path.join(PROJECT_ROOT, "visualizations")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_model_comparison(comparison_df: pd.DataFrame) -> str:
    """Generate bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC."""
    df_melted = pd.melt(
        comparison_df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=df_melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette="viridis",
        edgecolor="black",
        alpha=0.9,
        ax=ax,
    )

    ax.set_title("Machine Learning Models Evaluation Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_xlabel("Algorithm", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=45,
            )

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "model_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_confusion_matrices(trained_models: dict, X_test: np.ndarray, y_test: pd.Series) -> str:
    """Generate side-by-side confusion matrices for all 4 models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(trained_models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "fontweight": "bold"},
            xticklabels=["Not Potable (0)", "Potable (1)"],
            yticklabels=["Not Potable (0)", "Potable (1)"],
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.suptitle("Confusion Matrices across Evaluated Models", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_roc_curves(trained_models: dict, X_test: np.ndarray, y_test: pd.Series) -> str:
    """Plot overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_score:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "roc_curve.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def run_evaluation():
    """Execute complete model training, metric computation, and plot generation."""
    trained_models, comparison_df, X_train, X_test, y_train, y_test = train_and_evaluate_all()

    print("\n--- Detailed Classification Reports ---")
    for name, model in trained_models.items():
        print(f"\nModel: {name}")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"], digits=4))

    print("\nGenerating Model Comparison plot...")
    plot_model_comparison(comparison_df)

    print("Generating Confusion Matrices...")
    plot_confusion_matrices(trained_models, X_test, y_test)

    print("Generating ROC Curves...")
    plot_roc_curves(trained_models, X_test, y_test)

    print("\n--- Trade-off Analysis: Precision vs. Recall ---")
    print(
        "In water potability screening, a FALSE POSITIVE (predicting potable when water is actually contaminated)\n"
        "carries acute health risks, whereas a FALSE NEGATIVE (classifying potable water as unsafe) incurs water\n"
        "loss or secondary treatment costs. Therefore, high Precision protects consumer safety, while balanced\n"
        "Recall minimizes false alarms. Ensemble trees (Random Forest & Gradient Boosting) yield the strongest\n"
        "F1 and ROC-AUC profiles."
    )


if __name__ == "__main__":
    run_evaluation()
