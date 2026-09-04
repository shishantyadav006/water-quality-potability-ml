"""
data_loader.py
Module for acquiring, loading, and validating water quality dataset.
"""

import os
import urllib.request
import pandas as pd
import numpy as np


DATA_URL = "https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/main/water_potability.csv"
DEFAULT_RAW_PATH = os.path.join("data", "raw", "water_quality.csv")

# Standard expected features in canonical naming
FEATURE_MAPPING = {
    "ph": "ph",
    "hardness": "Hardness",
    "solids": "Solids",
    "chloramines": "Chloramines",
    "sulfate": "Sulfate",
    "conductivity": "Conductivity",
    "organic_carbon": "Organic_carbon",
    "trihalomethanes": "Trihalomethanes",
    "turbidity": "Turbidity",
    "potability": "Potability",
}


def download_dataset(url: str = DATA_URL, dest_path: str = DEFAULT_RAW_PATH) -> str:
    """Download the water quality dataset if it does not exist locally."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading dataset from {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print(f"Dataset already present at {dest_path}")
    return dest_path


def load_dataset(csv_path: str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Load water quality CSV into a DataFrame with standardized column names.
    Adapts gracefully if column casings or slight name variants are provided.
    """
    if not os.path.exists(csv_path):
        download_dataset(dest_path=csv_path)

    df = pd.read_csv(csv_path)
    
    # Standardize column names (strip whitespace and map case-insensitively)
    col_map = {}
    for col in df.columns:
        norm = col.strip().lower()
        if norm in FEATURE_MAPPING:
            col_map[col] = FEATURE_MAPPING[norm]
        else:
            col_map[col] = col.strip()
    df = df.rename(columns=col_map)
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Perform thorough validation checks on the water quality dataset.
    Returns a dictionary of diagnostic metrics.
    """
    # 1. Target identification
    possible_targets = [c for c in df.columns if c.lower() in ["potability", "target", "safe", "is_potable"]]
    target_col = possible_targets[0] if possible_targets else None

    # 2. Features identification
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != target_col]

    # 3. Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_summary = {
        col: {"count": int(missing_counts[col]), "percentage": round(float(missing_pct[col]), 2)}
        for col in df.columns if missing_counts[col] > 0
    }

    # 4. Duplicates
    duplicate_count = int(df.duplicated().sum())

    # 5. Target distribution
    target_dist = {}
    if target_col and target_col in df:
        val_counts = df[target_col].value_counts().to_dict()
        target_dist = {str(k): int(v) for k, v in val_counts.items()}

    # 6. Physical validity checks
    # e.g., pH outside standard realistic aqueous range 0-14, negative values in physical concentrations
    invalid_checks = {}
    if "ph" in df:
        invalid_ph = int(((df["ph"] < 0) | (df["ph"] > 14)).sum())
        invalid_checks["ph_out_of_bounds_0_14"] = invalid_ph

    negative_features = {}
    for col in feature_cols:
        neg_count = int((df[col] < 0).sum())
        if neg_count > 0:
            negative_features[col] = neg_count
    invalid_checks["negative_values"] = negative_features

    report = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "target_column": target_col,
        "feature_columns": feature_cols,
        "missing_summary": missing_summary,
        "duplicate_rows": duplicate_count,
        "target_distribution": target_dist,
        "validity_checks": invalid_checks,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    return report


def print_validation_report(report: dict, df: pd.DataFrame) -> None:
    """Format and print the validation diagnostics cleanly."""
    print("=" * 60)
    print("        WATER QUALITY DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Rows:     {report['shape']['rows']}")
    print(f"Total Columns:  {report['shape']['columns']}")
    print(f"Target Column:  {report['target_column']}")
    print(f"Numerical Features ({len(report['feature_columns'])}): {', '.join(report['feature_columns'])}")
    print(f"Duplicate Rows: {report['duplicate_rows']}")

    print("\n--- Target Class Balance ---")
    for k, v in report['target_distribution'].items():
        label = "Potable (1)" if str(k) in ["1", "1.0"] else "Not Potable (0)"
        pct = (v / report['shape']['rows']) * 100
        print(f"  Class {k} ({label}): {v} samples ({pct:.1f}%)")

    print("\n--- Missing Values Detected ---")
    if report['missing_summary']:
        for col, stats in report['missing_summary'].items():
            print(f"  {col:18}: {stats['count']:4d} missing ({stats['percentage']:.2f}%)")
    else:
        print("  No missing values detected.")

    print("\n--- Physical Range Checks ---")
    for check, val in report['validity_checks'].items():
        print(f"  {check}: {val}")

    print("\n--- Descriptive Statistics Preview ---")
    print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]].to_string())
    print("=" * 60)


if __name__ == "__main__":
    download_path = download_dataset()
    data = load_dataset(download_path)
    val_report = validate_dataset(data)
    print_validation_report(val_report, data)
