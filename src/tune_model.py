"""
tune_model.py
Hyperparameter tuning for top-performing ensemble candidate models.
Uses Stratified K-Fold Cross-Validation with GridSearchCV / RandomizedSearchCV.
Saves the optimal tuned model to models/best_model.pkl.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from src.data_preprocessing import preprocess_data

BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")


def tune_random_forest(X_train: np.ndarray, y_train: pd.Series) -> GridSearchCV:
    """Tune Random Forest Classifier using 5-Fold Stratified Cross-Validation."""
    print("\n--- Tuning Random Forest Classifier ---")
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)

    # Use ROC-AUC as scoring to balance sensitivity and specificity
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best RF Parameters: {grid_search.best_params_}")
    print(f"Best RF CV ROC-AUC: {grid_search.best_score_:.4f}")
    return grid_search


def tune_gradient_boosting(X_train: np.ndarray, y_train: pd.Series) -> GridSearchCV:
    """Tune Gradient Boosting Classifier using 5-Fold Stratified Cross-Validation."""
    print("\n--- Tuning Gradient Boosting Classifier ---")
    param_grid = {
        "n_estimators": [100, 150],
        "learning_rate": [0.03, 0.08, 0.12],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best GB Parameters: {grid_search.best_params_}")
    print(f"Best GB CV ROC-AUC: {grid_search.best_score_:.4f}")
    return grid_search


def evaluate_candidate(model, X_test: np.ndarray, y_test: pd.Series, name: str) -> dict:
    """Evaluate a candidate model on held-out test data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, y_prob)), 4) if y_prob is not None else np.nan,
    }
    return metrics


def run_tuning_and_save_best():
    """Execute hyperparameter tuning, compare candidates, and persist the champion model."""
    X_train, X_test, y_train, y_test, features, pipeline = preprocess_data(
        random_state=42, save_artifacts=False
    )

    rf_grid = tune_random_forest(X_train, y_train)
    gb_grid = tune_gradient_boosting(X_train, y_train)

    rf_best = rf_grid.best_estimator_
    gb_best = gb_grid.best_estimator_

    rf_metrics = evaluate_candidate(rf_best, X_test, y_test, "Tuned Random Forest")
    gb_metrics = evaluate_candidate(gb_best, X_test, y_test, "Tuned Gradient Boosting")

    comparison_df = pd.DataFrame([rf_metrics, gb_metrics])
    print("\n" + "=" * 60)
    print("      TUNED CANDIDATE PERFORMANCE ON TEST SET")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    # Champion selection: select model with highest ROC-AUC and F1
    if rf_metrics["ROC-AUC"] >= gb_metrics["ROC-AUC"]:
        champion_name = "Tuned Random Forest"
        champion_model = rf_best
        champion_metrics = rf_metrics
    else:
        champion_name = "Tuned Gradient Boosting"
        champion_model = gb_best
        champion_metrics = gb_metrics

    print(f"\nChampion Model Selected: {champion_name}")
    print(f"Metrics: {champion_metrics}")

    # Save champion model
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(champion_model, BEST_MODEL_PATH)
    print(f"Successfully persisted best model to: {BEST_MODEL_PATH}")

    return champion_model, champion_metrics


if __name__ == "__main__":
    run_tuning_and_save_best()
