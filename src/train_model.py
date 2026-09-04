"""
train_model.py
Trains baseline machine learning models on water quality dataset and outputs metrics.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict, Any, Tuple

from src.data_preprocessing import preprocess_data


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """Define the 4 benchmark models with reproducible random states."""
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=random_state, max_iter=1000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state, max_depth=6, min_samples_split=10
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state, n_estimators=150, max_depth=12
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state, n_estimators=120, learning_rate=0.08, max_depth=4
        ),
    }
    return models


def train_and_evaluate_all(
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Train all 4 models and compile comprehensive evaluation metrics.
    Returns:
        trained_models: dict of fitted models
        comparison_df: dataframe comparing metrics across all models
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test, features, pipeline = preprocess_data(
        random_state=random_state, save_artifacts=True
    )

    models = get_baseline_models(random_state=random_state)
    trained_models = {}
    records = []

    print("\nTraining and evaluating baseline models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        records.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
        })

    comparison_df = pd.DataFrame(records)
    return trained_models, comparison_df, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    trained_models, comparison_df, X_train, X_test, y_train, y_test = train_and_evaluate_all()
    print("\n" + "=" * 65)
    print("           BASELINE MODEL PERFORMANCE COMPARISON")
    print("=" * 65)
    print(comparison_df.to_string(index=False))
    print("=" * 65)
