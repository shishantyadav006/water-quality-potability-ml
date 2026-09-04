"""
explain_model.py
Explainable AI (XAI) engine using SHAP for Water Potability Prediction.
Provides global feature importance and sample-level prediction explanations.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Union

from src.data_preprocessing import preprocess_data, DEFAULT_RAW_PATH

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessing_pipeline.pkl")
FIG_DIR = os.path.join(PROJECT_ROOT, "visualizations")


class WaterQualityExplainer:
    """Class encapsulation for Model inference and SHAP Explainability."""

    def __init__(self, model_path: str = MODEL_PATH, pipeline_path: str = PIPELINE_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found at: {pipeline_path}. Run preprocessing first.")

        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)
        self.feature_names = [
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ]
        self.explainer = shap.TreeExplainer(self.model)

    def compute_global_explanations(self, X_sample: np.ndarray, save_plot: bool = True) -> np.ndarray:
        """
        Compute global SHAP values across a background sample and save summary plot.
        """
        shap_vals = self.explainer.shap_values(X_sample)

        # For binary classification (samples, features, 2)
        if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            shap_for_potable = shap_vals[:, :, 1]
        elif isinstance(shap_vals, list):
            shap_for_potable = shap_vals[1]
        else:
            shap_for_potable = shap_vals

        if save_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_for_potable,
                features=X_sample,
                feature_names=self.feature_names,
                plot_type="dot",
                show=False,
            )
            plt.title("SHAP Global Feature Impact on Water Potability (Class 1)", fontsize=13, fontweight="bold", pad=15)
            plt.tight_layout()
            out_path = os.path.join(FIG_DIR, "shap_summary.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved global SHAP summary plot: {out_path}")

        return shap_for_potable

    def explain_prediction(self, raw_input: Union[Dict[str, float], pd.DataFrame]) -> Dict[str, Any]:
        """
        Provide local prediction and SHAP feature attributions for a single water sample.
        """
        if isinstance(raw_input, dict):
            input_df = pd.DataFrame([raw_input])
        else:
            input_df = raw_input.copy()

        # Ensure all required features are present
        for feat in self.feature_names:
            if feat not in input_df.columns:
                raise ValueError(f"Missing required parameter '{feat}' in input sample.")

        # Reorder to canonical sequence
        input_ordered = input_df[self.feature_names]

        # Apply preprocessor (imputation + scaling)
        X_scaled = self.pipeline.transform(input_ordered)

        # Prediction and confidence
        prediction_class = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0]
        prob_potable = float(probabilities[1])
        prob_not_potable = float(probabilities[0])

        confidence_pct = round((prob_potable if prediction_class == 1 else prob_not_potable) * 100, 2)
        label = "POTABLE" if prediction_class == 1 else "NOT POTABLE"

        # SHAP calculation for the sample
        shap_vals = self.explainer.shap_values(X_scaled)
        if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            sample_shap = shap_vals[0, :, 1]
        elif isinstance(shap_vals, list):
            sample_shap = shap_vals[1][0]
        else:
            sample_shap = shap_vals[0]

        # Compile feature factor contributions
        contributions = []
        for i, feat in enumerate(self.feature_names):
            val = float(input_ordered.iloc[0][feat])
            impact = float(sample_shap[i])
            direction = "Toward Potable (+)" if impact > 0 else "Toward Not Potable (-)"
            contributions.append({
                "feature": feat,
                "value": val,
                "shap_value": round(impact, 4),
                "abs_importance": round(abs(impact), 4),
                "direction": direction,
            })

        # Sort by absolute contribution magnitude
        contributions.sort(key=lambda x: x["abs_importance"], reverse=True)

        return {
            "prediction": prediction_class,
            "prediction_label": label,
            "potable_probability": round(prob_potable, 4),
            "not_potable_probability": round(prob_not_potable, 4),
            "confidence_pct": confidence_pct,
            "top_factors": contributions,
            "disclaimer": (
                "Educational decision-support inference. SHAP values reflect statistical pattern "
                "associations learned by the algorithm from training data, not certified laboratory "
                "or medical causation."
            ),
        }


def run_explainability():
    """Run global SHAP computation and verify single sample explanation."""
    print("Initializing Water Quality SHAP Explainer...")
    explainer = WaterQualityExplainer()

    print("Computing global SHAP summary across test set...")
    _, X_test, _, _, _, _ = preprocess_data(save_artifacts=False)
    explainer.compute_global_explanations(X_test, save_plot=True)

    print("\nTesting single-sample explanation:")
    sample = {
        "ph": 7.35,
        "Hardness": 204.0,
        "Solids": 20791.0,
        "Chloramines": 7.3,
        "Sulfate": 368.5,
        "Conductivity": 564.0,
        "Organic_carbon": 10.4,
        "Trihalomethanes": 86.9,
        "Turbidity": 3.0,
    }
    explanation = explainer.explain_prediction(sample)
    print(f"Prediction: {explanation['prediction_label']} ({explanation['confidence_pct']}% confidence)")
    print(f"Potable Probability: {explanation['potable_probability']:.2%}")
    print("\nTop Contributing Factors:")
    for f in explanation["top_factors"][:5]:
        print(f"  {f['feature']:16}: Value={f['value']:<8.2f} SHAP={f['shap_value']:>+7.4f} ({f['direction']})")


if __name__ == "__main__":
    run_explainability()
