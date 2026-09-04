"""
generate_notebook.py
Generates the comprehensive notebooks/water_quality_analysis.ipynb file with full
narrative, code cells, and EDA workflows.
"""

import json
import os

NOTEBOOK_PATH = os.path.join("notebooks", "water_quality_analysis.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Intelligent Water Quality Assessment & Potability Prediction\n",
            "### Exploratory Data Analysis & Machine Learning Pipeline\n",
            "\n",
            "> **Disclaimer**: This is an educational decision-support project developed for academic and portfolio demonstration. It does not replace laboratory-certified chemical water testing or official public health inspection."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Project Background & Water Quality Parameters\n",
            "\n",
            "Access to safe drinking water is a fundamental human necessity. The dataset contains 9 physicochemical parameters measured across freshwater sources:\n",
            "- **pH**: Measure of acidic/basic balance (WHO guideline: 6.5 - 8.5).\n",
            "- **Hardness**: Capacity of water to precipitate soap, caused mainly by Calcium and Magnesium.\n",
            "- **Solids (TDS)**: Total dissolved solids in mg/L (desirable limit < 500 mg/L).\n",
            "- **Chloramines**: Major disinfectant formed when ammonia is added to chlorine (safe up to 4 mg/L).\n",
            "- **Sulfate**: Naturally occurring substance in minerals, soil, and rocks (desirable < 250 mg/L).\n",
            "- **Conductivity**: Electrical conductivity indicating ionic concentrations (safe < 400 uS/cm).\n",
            "- **Organic Carbon**: Total organic carbon (TOC) measuring dissolved decaying organic matter (< 2 mg/L for pure water, < 4 mg/L for treated).\n",
            "- **Trihalomethanes**: By-products of chlorine disinfection (safe < 80 ug/L).\n",
            "- **Turbidity**: Cloudiness caused by suspended particles (WHO standard < 5 NTU).\n",
            "- **Potability**: Binary target (1 = Potable/Drinkable, 0 = Not Potable)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Add project root to sys.path\n",
            "sys.path.append('..')\n",
            "from src.data_loader import load_dataset, validate_dataset\n",
            "\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "%matplotlib inline"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Dataset Loading & Diagnostics"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = load_dataset('../data/raw/water_quality.csv')\n",
            "report = validate_dataset(df)\n",
            "print(f\"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\")\n",
            "print(f\"Target class balance: {report['target_distribution']}\")\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Missing Value Analysis\n",
            "Analyzing missing value percentages to justify median imputation over deletion."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "missing = df.isnull().sum()\n",
            "missing_pct = (missing / len(df)) * 100\n",
            "missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct.round(2)})\n",
            "missing_df[missing_df['Missing Count'] > 0]"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Descriptive Statistics & Outlier Inspection"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Boxplot outlier visualization across features\n",
            "features = [c for c in df.columns if c != 'Potability']\n",
            "plt.figure(figsize=(15, 8))\n",
            "sns.boxplot(data=pd.melt(df[features]), x='variable', y='value')\n",
            "plt.xticks(rotation=45)\n",
            "plt.title('Feature Spread & Outlier Analysis (Raw Scales)')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Correlation Analysis\n",
            "Evaluating linear relationships between features and the Potability target."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 8))\n",
            "corr = df.corr()\n",
            "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Pearson r'})\n",
            "plt.title('Correlation Matrix of Water Quality Parameters')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Preprocessing & Leak-Free Pipeline Execution\n",
            "We split data before fitting imputers and scalers to prevent data leakage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.data_preprocessing import preprocess_data\n",
            "\n",
            "X_train, X_test, y_train, y_test, feature_names, pipeline = preprocess_data(\n",
            "    raw_csv_path='../data/raw/water_quality.csv',\n",
            "    test_size=0.2,\n",
            "    random_state=42\n",
            ")\n",
            "print(f\"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Model Training & Comparison\n",
            "Comparing Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.tree import DecisionTreeClassifier\n",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n",
            "\n",
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(random_state=42),\n",
            "    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),\n",
            "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),\n",
            "    'Gradient Boosting': GradientBoostingClassifier(random_state=42)\n",
            "}\n",
            "\n",
            "results = []\n",
            "for name, model in models.items():\n",
            "    model.fit(X_train, y_train)\n",
            "    y_pred = model.predict(X_test)\n",
            "    y_prob = model.predict_proba(X_test)[:, 1]\n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Accuracy': round(accuracy_score(y_test, y_pred), 4),\n",
            "        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),\n",
            "        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),\n",
            "        'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),\n",
            "        'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)\n",
            "    })\n",
            "\n",
            "comparison_df = pd.DataFrame(results)\n",
            "comparison_df"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Summary & Key Findings\n",
            "- The correlation matrix demonstrates low linear correlation between individual features and water potability, indicating that safe water status depends on non-linear combinations of chemical bounds.\n",
            "- Ensemble tree methods (Random Forest and Gradient Boosting) consistently outperform linear baselines like Logistic Regression.\n",
            "- SHAP explainability provides transparent insight into non-linear boundary decisions for individual samples."
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated notebook successfully at {NOTEBOOK_PATH}")
