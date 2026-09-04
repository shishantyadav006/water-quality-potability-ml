"""
app.py
Streamlit Web Application: Intelligent Water Quality Assessment & Potability Prediction.
Features single-sample inference, Explainable AI (SHAP), batch CSV predictions, and analytics dashboard.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from src.explain_model import WaterQualityExplainer
from src.data_loader import load_dataset, validate_dataset

# Set Page Config
st.set_page_config(
    page_title="WaterPotability AI | Water Quality Assessment",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        margin-bottom: 10px;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .potable-badge {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        border: 2px solid #31C48D;
    }
    .not-potable-badge {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        border: 2px solid #F98080;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_explainer():
    """Load model and explainer once into cache."""
    try:
        return WaterQualityExplainer()
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None


@st.cache_data
def get_data():
    """Load processed dataset for visualization and statistics."""
    try:
        return load_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return None


def show_overview():
    """Render Project Overview and System Guide."""
    st.markdown("<div class='main-header'>💧 Intelligent Water Quality Assessment</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Machine Learning Decision-Support System with Transparent SHAP Explainability</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='disclaimer-box'>
        ⚠️ <strong>Educational & Research Disclaimer:</strong><br>
        This application is an educational decision-support prototype developed for academic demonstrations.
        Predictions are derived statistically from machine learning models trained on historical data.
        <strong>It is not a certified water testing device or medical instrument.</strong> Official potability
        determinations require laboratory chemical and microbiological assays in compliance with WHO / EPA regulations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🎯 Project Purpose & Capabilities")
        st.markdown(
            """
            This platform assesses the **physicochemical parameters** of freshwater samples to predict whether
            a sample is **Potable** (safe for human consumption) or **Not Potable**.

            Key system features:
            - **End-to-End Leak-Free Pipeline**: Scikit-learn Pipeline with median imputation and standard scaling.
            - **Tuned Ensemble ML**: High-capacity Random Forest Classifier tuned via 5-fold cross-validation.
            - **Transparent Explainable AI (SHAP)**: Every prediction highlights the specific parameters that swayed the decision.
            - **Batch Inference Support**: Upload laboratory test CSVs for bulk potability scoring.
            """
        )

        st.subheader("🧪 Supported Physicochemical Parameters")
        params_info = pd.DataFrame([
            {"Parameter": "pH", "Standard Reference": "6.5 – 8.5 (WHO)", "Significance": "Acid-base balance"},
            {"Parameter": "Hardness", "Standard Reference": "< 300 mg/L", "Significance": "Ca2+ / Mg2+ mineral content"},
            {"Parameter": "Solids (TDS)", "Standard Reference": "< 500 - 1000 ppm", "Significance": "Total dissolved solids"},
            {"Parameter": "Chloramines", "Standard Reference": "≤ 4.0 ppm", "Significance": "Disinfection residue"},
            {"Parameter": "Sulfate", "Standard Reference": "< 250 mg/L", "Significance": "Natural mineral salts"},
            {"Parameter": "Conductivity", "Standard Reference": "< 400 μS/cm", "Significance": "Ionic conductance"},
            {"Parameter": "Organic Carbon", "Standard Reference": "< 2.0 - 4.0 ppm", "Significance": "Decaying organic matter"},
            {"Parameter": "Trihalomethanes", "Standard Reference": "< 80 μg/L", "Significance": "Disinfection by-product"},
            {"Parameter": "Turbidity", "Standard Reference": "< 5.0 NTU", "Significance": "Water clarity & suspended matter"},
        ])
        st.table(params_info)

    with col2:
        st.subheader("📊 Dataset Overview")
        data = get_data()
        if data is not None:
            total_samples = len(data)
            potable_count = int((data["Potability"] == 1).sum())
            non_potable_count = int((data["Potability"] == 0).sum())

            c1, c2 = st.columns(2)
            c1.metric("Total Water Samples", f"{total_samples:,}")
            c2.metric("Features Analyzed", "9 Parameters")

            c3, c4 = st.columns(2)
            c3.metric("Potable Samples (1)", f"{potable_count:,} ({potable_count/total_samples:.1%})")
            c4.metric("Non-Potable (0)", f"{non_potable_count:,} ({non_potable_count/total_samples:.1%})")

            # Mini donut chart
            fig = px.pie(
                values=[non_potable_count, potable_count],
                names=["Not Potable (0)", "Potable (1)"],
                color=["Not Potable (0)", "Potable (1)"],
                color_discrete_map={"Not Potable (0)": "#EF4444", "Potable (1)": "#10B981"},
                hole=0.5,
                title="Class Balance in Dataset",
            )
            fig.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=260)
            st.plotly_chart(fig, use_container_width=True)


def show_prediction_page():
    """Interactive single-sample prediction interface with SHAP explanations."""
    st.markdown("<div class='main-header'>🔬 Water Quality Analyzer & Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Input chemical measurements to evaluate potability and generate AI explanations.</div>",
        unsafe_allow_html=True,
    )

    explainer = get_explainer()
    if explainer is None:
        st.error("Model engine is currently unavailable. Please train models first.")
        return

    # Presets for quick testing
    st.sidebar.markdown("### ⚡ Quick Presets")
    preset = st.sidebar.selectbox(
        "Load Sample Profile:",
        ["Custom Values", "Typical Clean Drinking Water", "Contaminated / Industrial Water", "High-Turbidity River Runoff"],
    )

    defaults = {
        "ph": 7.08,
        "Hardness": 196.4,
        "Solids": 20928.0,
        "Chloramines": 7.12,
        "Sulfate": 333.8,
        "Conductivity": 426.2,
        "Organic_carbon": 14.28,
        "Trihalomethanes": 66.4,
        "Turbidity": 3.97,
    }

    if preset == "Typical Clean Drinking Water":
        defaults = {
            "ph": 7.2,
            "Hardness": 160.0,
            "Solids": 14000.0,
            "Chloramines": 4.5,
            "Sulfate": 280.0,
            "Conductivity": 340.0,
            "Organic_carbon": 8.5,
            "Trihalomethanes": 45.0,
            "Turbidity": 2.2,
        }
    elif preset == "Contaminated / Industrial Water":
        defaults = {
            "ph": 4.2,
            "Hardness": 280.0,
            "Solids": 38000.0,
            "Chloramines": 10.5,
            "Sulfate": 420.0,
            "Conductivity": 650.0,
            "Organic_carbon": 22.0,
            "Trihalomethanes": 95.0,
            "Turbidity": 5.8,
        }
    elif preset == "High-Turbidity River Runoff":
        defaults = {
            "ph": 8.8,
            "Hardness": 230.0,
            "Solids": 32000.0,
            "Chloramines": 6.8,
            "Sulfate": 360.0,
            "Conductivity": 510.0,
            "Organic_carbon": 18.2,
            "Trihalomethanes": 80.0,
            "Turbidity": 6.4,
        }

    with st.form("water_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ph = st.number_input("pH (0.0 – 14.0)", min_value=0.0, max_value=14.0, value=float(defaults["ph"]), step=0.1, help="WHO standard: 6.5 to 8.5")
            hardness = st.number_input("Hardness (mg/L)", min_value=40.0, max_value=350.0, value=float(defaults["Hardness"]), step=1.0, help="Mineral hardness concentration")
            solids = st.number_input("Solids / TDS (ppm)", min_value=300.0, max_value=65000.0, value=float(defaults["Solids"]), step=100.0, help="Total dissolved solids")

        with col2:
            chloramines = st.number_input("Chloramines (ppm)", min_value=0.1, max_value=15.0, value=float(defaults["Chloramines"]), step=0.1, help="Disinfectant residual (safe < 4.0)")
            sulfate = st.number_input("Sulfate (mg/L)", min_value=120.0, max_value=500.0, value=float(defaults["Sulfate"]), step=1.0, help="Natural mineral salts")
            conductivity = st.number_input("Conductivity (μS/cm)", min_value=150.0, max_value=800.0, value=float(defaults["Conductivity"]), step=1.0, help="Electrical conductivity")

        with col3:
            organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=1.0, max_value=30.0, value=float(defaults["Organic_carbon"]), step=0.1, help="Total organic carbon")
            trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.5, max_value=130.0, value=float(defaults["Trihalomethanes"]), step=0.5, help="Disinfection by-product (< 80)")
            turbidity = st.number_input("Turbidity (NTU)", min_value=1.0, max_value=7.0, value=float(defaults["Turbidity"]), step=0.1, help="Optical cloudiness measure (< 5.0)")

        submit_btn = st.form_submit_button("🧪 Analyze Water Quality", use_container_width=True)

    if submit_btn:
        input_dict = {
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity,
        }

        with st.spinner("Analyzing physicochemical sample and computing SHAP attributions..."):
            try:
                result = explainer.explain_prediction(input_dict)
                is_potable = result["prediction"] == 1
                prob_potable = result["potable_probability"]
                conf_pct = result["confidence_pct"]

                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 1.4])

                with res_col1:
                    st.markdown("### 📋 Prediction Outcome")
                    if is_potable:
                        st.markdown(
                            f"<div class='potable-badge'>✅ POTABLE (DRINKABLE)</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='not-potable-badge'>❌ NOT POTABLE (UNSAFE)</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**Model Confidence:** `{conf_pct:.1f}%`")
                    st.markdown(f"**Potability Probability:** `{prob_potable:.2%}`")

                    # Probability gauge
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prob_potable * 100,
                            title={"text": "Potability Index (%)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#10B981" if is_potable else "#EF4444"},
                                "steps": [
                                    {"range": [0, 50], "color": "#FEE2E2"},
                                    {"range": [50, 100], "color": "#D1FAE5"},
                                ],
                                "threshold": {
                                    "line": {"color": "black", "width": 3},
                                    "thickness": 0.75,
                                    "value": 50,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=230, margin=dict(t=20, b=10, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with res_col2:
                    st.markdown("### 🔍 Explainable AI: SHAP Feature Contributions")
                    st.caption("Bar length shows how strongly each parameter pushed the model toward Potable (+) or Not Potable (-).")

                    top_factors = result["top_factors"]
                    factors_df = pd.DataFrame(top_factors)
                    factors_df["color"] = factors_df["shap_value"].apply(
                        lambda x: "#10B981" if x > 0 else "#EF4444"
                    )

                    fig_bar = px.bar(
                        factors_df.sort_values("shap_value", ascending=True),
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        color="color",
                        color_discrete_map="identity",
                        labels={"shap_value": "SHAP Attribution Impact", "feature": "Parameter"},
                        hover_data=["value", "direction"],
                    )
                    fig_bar.update_layout(
                        showlegend=False,
                        height=280,
                        margin=dict(t=20, b=20, l=20, r=20),
                        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="gray"),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("#### 🔬 Detailed Attribution Summary Table")
                display_table = pd.DataFrame([
                    {
                        "Parameter": f["feature"],
                        "Input Value": f"{f['value']:.2f}",
                        "SHAP Impact Score": f"{f['shap_value']:+.4f}",
                        "Influence Direction": f["direction"],
                    }
                    for f in top_factors
                ])
                st.dataframe(display_table, use_container_width=True)

            except Exception as ex:
                st.error(f"Prediction failed: {ex}")


def show_batch_prediction():
    """CSV Batch upload and inference section."""
    st.markdown("<div class='main-header'>📁 Batch Water Samples Inference</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Upload laboratory test CSV files to evaluate multiple samples simultaneously.</div>",
        unsafe_allow_html=True,
    )

    explainer = get_explainer()
    if explainer is None:
        st.error("Model engine is currently unavailable.")
        return

    # Downloadable template
    sample_template = pd.DataFrame([{
        "ph": 7.1,
        "Hardness": 195.0,
        "Solids": 21000.0,
        "Chloramines": 7.2,
        "Sulfate": 330.0,
        "Conductivity": 420.0,
        "Organic_carbon": 14.0,
        "Trihalomethanes": 65.0,
        "Turbidity": 3.9,
    }])
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="water_samples_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Water Quality CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded `{len(df)}` samples from uploaded file.")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", type="primary"):
                predictions = []
                potable_probs = []

                # Transform via pipeline
                X_proc = explainer.pipeline.transform(df[explainer.feature_names])
                preds = explainer.model.predict(X_proc)
                probs = explainer.model.predict_proba(X_proc)[:, 1]

                results_df = df.copy()
                results_df["Predicted_Potability"] = preds
                results_df["Potability_Label"] = ["Potable" if p == 1 else "Not Potable" for p in preds]
                results_df["Potability_Probability"] = [round(float(p), 4) for p in probs]

                st.success(f"Batch prediction completed for {len(results_df)} samples!")

                c1, c2 = st.columns(2)
                pot_cnt = int((preds == 1).sum())
                c1.metric("Predicted Potable", f"{pot_cnt} ({pot_cnt/len(preds):.1%})")
                c2.metric("Predicted Not Potable", f"{len(preds) - pot_cnt} ({(len(preds) - pot_cnt)/len(preds):.1%})")

                st.dataframe(results_df, use_container_width=True)

                st.download_button(
                    label="📥 Download Prediction Results (CSV)",
                    data=results_df.to_csv(index=False),
                    file_name="water_potability_predictions.csv",
                    mime="text/csv",
                )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")


def show_analytics():
    """Display system visualizations, model comparison, and EDA."""
    st.markdown("<div class='main-header'>📈 Model Performance & EDA Visualizations</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Detailed exploratory analysis and machine learning benchmark evaluations.</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📊 ML Model Benchmarks", "🔍 Exploratory Data Analysis"])

    with tab1:
        st.subheader("Model Evaluation & Comparison")
        col1, col2 = st.columns(2)

        comp_img_path = os.path.join(PROJECT_ROOT, "visualizations", "model_comparison.png")
        if os.path.exists(comp_img_path):
            with col1:
                st.image(comp_img_path, caption="Algorithm Comparison (Accuracy, Precision, Recall, F1, ROC-AUC)", use_container_width=True)

        roc_img_path = os.path.join(PROJECT_ROOT, "visualizations", "roc_curve.png")
        if os.path.exists(roc_img_path):
            with col2:
                st.image(roc_img_path, caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

        col3, col4 = st.columns(2)
        cm_img_path = os.path.join(PROJECT_ROOT, "visualizations", "confusion_matrix.png")
        if os.path.exists(cm_img_path):
            with col3:
                st.image(cm_img_path, caption="Confusion Matrices Across Models", use_container_width=True)

        shap_img_path = os.path.join(PROJECT_ROOT, "visualizations", "shap_summary.png")
        if os.path.exists(shap_img_path):
            with col4:
                st.image(shap_img_path, caption="Global SHAP Feature Attribution Impact", use_container_width=True)

    with tab2:
        st.subheader("Exploratory Data Analysis Visuals")
        col1, col2 = st.columns(2)

        target_img_path = os.path.join(PROJECT_ROOT, "visualizations", "target_distribution.png")
        if os.path.exists(target_img_path):
            with col1:
                st.image(target_img_path, caption="Target Class Distribution", use_container_width=True)

        corr_img_path = os.path.join(PROJECT_ROOT, "visualizations", "correlation_heatmap.png")
        if os.path.exists(corr_img_path):
            with col2:
                st.image(corr_img_path, caption="Pairwise Correlation Heatmap (Pearson)", use_container_width=True)

        col3, col4 = st.columns(2)
        dist_img_path = os.path.join(PROJECT_ROOT, "visualizations", "feature_distributions.png")
        if os.path.exists(dist_img_path):
            with col3:
                st.image(dist_img_path, caption="Feature Distributions by Potability Status", use_container_width=True)

        miss_img_path = os.path.join(PROJECT_ROOT, "visualizations", "missing_values.png")
        if os.path.exists(miss_img_path):
            with col4:
                st.image(miss_img_path, caption="Missing Values Breakdown", use_container_width=True)


def main():
    """Main application navigation."""
    st.sidebar.title("💧 WaterPotability AI")
    st.sidebar.markdown("**Decision-Support Platform**")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview & Purpose", "Single Sample Predictor", "Batch CSV Inference", "Analytics & Visualizations"],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "🎓 **Academic Major Project**\n"
        "Explainable Machine Learning for Water Quality Assessment.\n\n"
        "Built with Scikit-learn, SHAP, & Streamlit."
    )

    if page == "Overview & Purpose":
        show_overview()
    elif page == "Single Sample Predictor":
        show_prediction_page()
    elif page == "Batch CSV Inference":
        show_batch_prediction()
    elif page == "Analytics & Visualizations":
        show_analytics()


if __name__ == "__main__":
    main()
