"""
data_preprocessing.py
Reproducible, leak-free preprocessing pipeline for Water Quality Potability Prediction.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from src.data_loader import load_dataset, validate_dataset, DEFAULT_RAW_PATH



PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_water_quality.csv")
PIPELINE_PATH = os.path.join("models", "preprocessing_pipeline.pkl")


def build_preprocessor() -> Pipeline:
    """
    Build a Scikit-learn Pipeline with:
    1. Median Imputation (robust to skew/outliers and handles missing values)
    2. Standard Scaler (zero mean, unit variance for scale-sensitive models)
    """
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return preprocessor


def preprocess_data(
    raw_csv_path: str = DEFAULT_RAW_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, List[str], Pipeline]:
    """
    Execute end-to-end data preprocessing:
    - Load & validate raw dataset
    - Separate features (X) and target (y)
    - Stratified train/test split to maintain class balance
    - Fit preprocessor strictly on X_train (preventing data leakage)
    - Transform both X_train and X_test
    - Optionally save cleaned dataset and preprocessing pipeline artifact
    """
    # 1. Load dataset
    df = load_dataset(raw_csv_path)

    # 2. Validation & schema check
    report = validate_dataset(df)
    target_col = report["target_column"]
    feature_cols = report["feature_columns"]

    if target_col is None:
        raise ValueError("Target column 'Potability' not found in dataset.")

    # 3. Handle duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 4. Feature and Target Separation
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    # 5. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. Fit Preprocessing Pipeline strictly on X_train to avoid data leakage
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 7. Persistence
    if save_artifacts:
        # Save complete cleaned dataset (imputed, unscaled for EDA and readable analytics)
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        imputer_only = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            imputer_only.fit_transform(X),
            columns=feature_cols,
        )
        cleaned_df = pd.concat([X_imputed, y.reset_index(drop=True)], axis=1)
        cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Cleaned dataset saved to: {PROCESSED_DATA_PATH}")

        # Save fitted preprocessing pipeline for deployment/inference
        os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
        joblib.dump(preprocessor, PIPELINE_PATH)
        print(f"Fitted preprocessing pipeline saved to: {PIPELINE_PATH}")

    return X_train_processed, X_test_processed, y_train, y_test, feature_cols, preprocessor


if __name__ == "__main__":
    print("Running data preprocessing pipeline...")
    X_tr, X_te, y_tr, y_te, feats, pipe = preprocess_data()
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}")
    print(f"X_test shape:  {X_te.shape}, y_test shape:  {y_te.shape}")
    print(f"Feature count: {len(feats)} ({', '.join(feats)})")
    print(f"Train Potability distribution: {dict(pd.Series(y_tr).value_counts())}")
    print(f"Test Potability distribution:  {dict(pd.Series(y_te).value_counts())}")
