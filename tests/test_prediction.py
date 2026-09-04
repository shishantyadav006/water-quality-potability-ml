"""
test_prediction.py
Unit and integration test suite for Water Quality Prediction & Explainability Pipeline.
Tests model persistence, input validation, output boundaries, and SHAP explanations.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from src.explain_model import WaterQualityExplainer, MODEL_PATH, PIPELINE_PATH


@pytest.fixture(scope="module")
def explainer():
    """Fixture providing initialized WaterQualityExplainer instance."""
    return WaterQualityExplainer()


@pytest.fixture
def valid_sample():
    """Standard valid physicochemical water sample."""
    return {
        "ph": 7.2,
        "Hardness": 195.0,
        "Solids": 21000.0,
        "Chloramines": 7.2,
        "Sulfate": 330.0,
        "Conductivity": 420.0,
        "Organic_carbon": 14.0,
        "Trihalomethanes": 65.0,
        "Turbidity": 3.9,
    }


def test_model_and_pipeline_files_exist():
    """Test 1: Verify model and preprocessing artifacts exist and load cleanly."""
    assert os.path.exists(MODEL_PATH), f"Model artifact missing at {MODEL_PATH}"
    assert os.path.exists(PIPELINE_PATH), f"Pipeline artifact missing at {PIPELINE_PATH}"

    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)

    assert hasattr(model, "predict"), "Loaded model lacks predict method"
    assert hasattr(pipeline, "transform"), "Loaded pipeline lacks transform method"


def test_prediction_for_valid_input(explainer, valid_sample):
    """Test 2: Valid input vector yields valid prediction and probability."""
    result = explainer.explain_prediction(valid_sample)

    assert "prediction" in result
    assert "prediction_label" in result
    assert "potable_probability" in result
    assert "confidence_pct" in result
    assert "top_factors" in result

    # Check probability bounds
    prob = result["potable_probability"]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} is out of bounds [0.0, 1.0]"
    assert 0.0 <= result["not_potable_probability"] <= 1.0
    assert abs((result["potable_probability"] + result["not_potable_probability"]) - 1.0) < 1e-4


def test_prediction_output_labels(explainer, valid_sample):
    """Test 3: Prediction label is strictly 'POTABLE' or 'NOT POTABLE'."""
    result = explainer.explain_prediction(valid_sample)
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["POTABLE", "NOT POTABLE"]


def test_missing_feature_rejected_gracefully(explainer):
    """Test 4: Reject input missing required physicochemical features."""
    incomplete_sample = {
        "ph": 7.0,
        "Hardness": 200.0,
        # Missing Solids, Sulfate, Chloramines, etc.
    }
    with pytest.raises(ValueError) as excinfo:
        explainer.explain_prediction(incomplete_sample)

    assert "Missing required parameter" in str(excinfo.value)


def test_shap_explanation_structure(explainer, valid_sample):
    """Test 5: Explainability engine produces contributions for all 9 features."""
    result = explainer.explain_prediction(valid_sample)
    factors = result["top_factors"]

    assert len(factors) == 9, f"Expected 9 feature factors, got {len(factors)}"

    feature_names_returned = {f["feature"] for f in factors}
    expected_names = set(explainer.feature_names)
    assert feature_names_returned == expected_names

    for factor in factors:
        assert isinstance(factor["shap_value"], float)
        assert factor["direction"] in ["Toward Potable (+)", "Toward Not Potable (-)"]
        assert factor["abs_importance"] >= 0.0


def test_imputation_handles_nan_in_pipeline(explainer, valid_sample):
    """Test 6: Preprocessing pipeline gracefully handles NaN inputs via median imputer."""
    sample_with_nan = valid_sample.copy()
    sample_with_nan["ph"] = np.nan
    sample_with_nan["Sulfate"] = np.nan

    # Should not raise an exception because the imputer handles NaNs
    result = explainer.explain_prediction(sample_with_nan)
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["POTABLE", "NOT POTABLE"]
