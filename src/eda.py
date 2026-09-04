"""
eda.py
Exploratory Data Analysis and visualization generator for Water Quality dataset.
Saves high-resolution figures to the visualizations/ directory.
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

from src.data_loader import load_dataset, validate_dataset

# Set global visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
FIG_DIR = os.path.join(PROJECT_ROOT, "visualizations")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_missing_values(df: pd.DataFrame) -> str:
    """Visualize missing value percentage per feature."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(missing_pct.index, missing_pct.values, color="#e74c3c", edgecolor="black", alpha=0.85)
    ax.set_title("Missing Values Percentage by Water Quality Parameter", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Missing Percentage (%)", fontsize=11)
    ax.set_xlabel("Parameter", fontsize=11)
    ax.set_ylim(0, max(missing_pct.values) * 1.25)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}% ({int(height * len(df) / 100)})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "missing_values.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_target_distribution(df: pd.DataFrame) -> str:
    """Visualize target class balance with counts and percentages."""
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["Potability"].value_counts().sort_index()
    labels = ["Not Potable (0)", "Potable (1)"]
    colors = ["#e74c3c", "#2ecc71"]

    # Bar plot
    bars = ax_bar.bar(labels, counts.values, color=colors, edgecolor="black", alpha=0.85)
    ax_bar.set_title("Water Potability Sample Counts", fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("Number of Samples", fontsize=11)
    ax_bar.set_ylim(0, max(counts.values) * 1.15)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax_bar.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Donut / Pie plot
    wedges, texts, autotexts = ax_pie.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="black"),
        textprops=dict(fontsize=11),
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax_pie.set_title("Class Ratio", fontsize=12, fontweight="bold")

    plt.suptitle("Water Potability Target Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "target_distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_feature_distributions(df: pd.DataFrame) -> str:
    """Plot histograms and KDE distributions for all 9 numerical features split by Potability."""
    features = [c for c in df.columns if c != "Potability"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.histplot(
            data=df,
            x=feature,
            hue="Potability",
            kde=True,
            palette={0: "#e74c3c", 1: "#2ecc71"},
            ax=ax,
            bins=30,
            alpha=0.4,
            element="step",
        )
        ax.set_title(f"Distribution of {feature}", fontsize=11, fontweight="bold")
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Density / Count", fontsize=10)
        # Custom legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=["Not Potable (0)", "Potable (1)"], title="Potability", fontsize=8)

    plt.suptitle("Physicochemical Feature Distributions by Potability Class", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "feature_distributions.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Plot Pearson correlation matrix across all features and target."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        vmin=-0.3,
        center=0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Pairwise Feature Correlation Matrix (Pearson)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def run_eda():
    """Run all EDA analyses and generate visualization artifacts."""
    print("Loading raw dataset for EDA...")
    df = load_dataset()
    print("Generating Missing Values plot...")
    plot_missing_values(df)
    print("Generating Target Distribution plot...")
    plot_target_distribution(df)
    print("Generating Feature Distributions plot...")
    plot_feature_distributions(df)
    print("Generating Correlation Heatmap...")
    plot_correlation_heatmap(df)
    print("EDA Visualizations generation complete!")


if __name__ == "__main__":
    run_eda()
