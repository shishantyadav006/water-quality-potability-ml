# Intelligent Water Quality Assessment & Potability Prediction Using Explainable Machine Learning

[![Python 3.x](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-success.svg)](tests/)

---

## 1. Overview
Access to clean, potable drinking water is critical for global public health and environmental sustainability. **Intelligent Water Quality Assessment & Potability Prediction** is an end-to-end machine learning decision-support system designed to evaluate physicochemical water metrics and predict whether water is safe for human consumption (**Potable**) or hazardous (**Not Potable**).

Crucially, the system moves beyond black-box classification by integrating **Explainable AI (SHAP)**, providing transparent, feature-level attributions that describe *why* a particular sample is deemed safe or unsafe.

---

## 2. Problem Statement
Traditional water quality testing involves extensive chemical reagents and microbiological incubation periods that take hours or days to yield results. In contrast, in-line or field probes can rapidly log physicochemical parameters (pH, turbidity, conductivity, sulfate, etc.), but interpreting the combined non-linear interactions of these parameters requires intelligent decision support. 

Key challenges addressed:
- **Non-linear parameter relationships**: Safe water depends on joint chemical equilibrium rather than independent thresholds.
- **Moderate class imbalance**: In natural freshwater datasets, potable samples are typically a minority class (~39%).
- **Interpretability gap**: High-stakes environmental and health applications require transparent explanation of predictions before decision-makers act.

---

## 3. Objectives
1. Build a reproducible, leak-free data preprocessing pipeline handling missing values via median imputation and standard scaling.
2. Conduct comprehensive Exploratory Data Analysis (EDA) investigating distributions, correlations, and class imbalances.
3. Train, benchmark, and evaluate four core machine learning classifiers (Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting).
4. Perform Stratified 5-Fold Cross-Validation hyperparameter tuning on top candidate models.
5. Implement global and local Explainable AI using SHAP (SHapley Additive exPlanations) to explain feature influence.
6. Develop a multi-tab Streamlit dashboard enabling real-time single-sample inference, batch CSV scoring, and interactive analytics.
7. Implement automated testing and GitHub-ready project architecture.

---

## 4. Dataset
The project utilizes the standardized **Water Potability Dataset** containing **3,276 water samples** and **9 physicochemical features** with a binary target:

| Feature | Unit | WHO / EPA Standard Reference | Description |
| :--- | :--- | :--- | :--- |
| **ph** | Dimensionless | 6.5 – 8.5 | Acid-base balance of aqueous solution |
| **Hardness** | mg/L | < 300 mg/L (desirable) | Concentration of Calcium & Magnesium salts |
| **Solids (TDS)** | ppm | < 500 – 1000 ppm | Total dissolved solids & mineral content |
| **Chloramines** | ppm | ≤ 4.0 ppm | Residual disinfectant concentration |
| **Sulfate** | mg/L | < 250 mg/L | Naturally occurring mineral salts |
| **Conductivity** | μS/cm | < 400 μS/cm | Electrical conductance indicating dissolved ions |
| **Organic_carbon** | ppm | < 2.0 – 4.0 ppm | Total organic carbon from decaying matter |
| **Trihalomethanes** | μg/L | < 80 μg/L | By-products of chlorine-based disinfection |
| **Turbidity** | NTU | < 5.0 NTU | Optical clarity and suspended sediment measure |
| **Potability** | Binary | 0 = Unsafe, 1 = Safe | Target variable (61.0% Not Potable, 39.0% Potable) |

Missing values were observed in:
- `ph`: 491 missing (14.99%)
- `Sulfate`: 781 missing (23.84%)
- `Trihalomethanes`: 162 missing (4.95%)

---

## 5. Technologies Used
- **Programming Language**: Python 3.14+
- **Data Manipulation & Math**: NumPy, Pandas, SciPy
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-Learn (Pipelines, Imputers, Ensembles, GridSearchCV)
- **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations)
- **Model Persistence**: Joblib
- **Web Interface**: Streamlit
- **Testing**: Pytest

---

## 6. Project Architecture
```
water-quality-potability-ml/
│
├── data/
│   ├── raw/
│   │   └── water_quality.csv            # Original raw dataset
│   └── processed/
│       └── cleaned_water_quality.csv    # Imputed, validated dataset
│
├── notebooks/
│   ├── generate_notebook.py             # Generator script
│   └── water_quality_analysis.ipynb     # Jupyter EDA & Modeling Notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Acquisition & validation module
│   ├── data_preprocessing.py            # Leak-free pipeline & split
│   ├── train_model.py                   # Baseline benchmark training
│   ├── evaluate_model.py                # Multi-metric evaluation & plots
│   ├── tune_model.py                    # Cross-validated hyperparameter tuning
│   └── explain_model.py                 # SHAP explainability engine
│
├── models/
│   ├── best_model.pkl                   # Tuned champion model
│   └── preprocessing_pipeline.pkl       # Fitted pipeline (imputer + scaler)
│
├── visualizations/
│   ├── missing_values.png               # Missing data percentages
│   ├── feature_distributions.png        # KDE & histograms by class
│   ├── correlation_heatmap.png          # Pearson correlation matrix
│   ├── target_distribution.png          # Class balance counts & ratio
│   ├── model_comparison.png             # Benchmark comparison bar chart
│   ├── confusion_matrix.png             # 4-model confusion matrices
│   ├── roc_curve.png                    # Multi-model ROC comparison
│   └── shap_summary.png                 # Global SHAP beeswarm summary
│
├── app/
│   └── app.py                           # Streamlit multi-page web application
│
├── tests/
│   └── test_prediction.py               # Automated pytest suite
│
├── requirements.txt                     # Project dependencies
├── README.md                            # Project documentation
├── .gitignore                           # Git ignore rules
└── LICENSE                              # MIT License
```

---

## 7. Data Preprocessing
To strictly prevent **data leakage**, data preparation follows strict ML engineering best practices:
1. **Stratified Split**: An 80/20 train/test split (`random_state=42`) is executed first, ensuring both splits contain exactly 39% potable and 61% non-potable samples.
2. **Median Imputation**: `SimpleImputer(strategy='median')` is fitted strictly on `X_train` and transforms `X_test`. Median is selected over mean to resist distribution skew and outliers.
3. **Standardization**: `StandardScaler()` is fitted strictly on imputed `X_train` to normalize scale variance across features (e.g., Solids in tens of thousands vs Turbidity in single digits).
4. **Scikit-Learn Pipeline**: Imputation and scaling are encapsulated into a single serialized `Pipeline` object (`models/preprocessing_pipeline.pkl`) used identically during batch and single-sample inference.

---

## 8. Exploratory Data Analysis
Key findings from EDA:
- **Low Linear Correlation**: Pearson correlation coefficients between individual features and `Potability` are low ($|r| < 0.05$), proving that potability cannot be separated by simple linear thresholds.
- **Multimodal & Skewed Distributions**: Solids exhibits positive skewness, while pH and Sulfate display Gaussian-like distributions centered near potable neutrality.
- **Class Imbalance**: Non-potable instances (1,998) outnumber potable instances (1,278), necessitating stratified sampling and ROC-AUC / F1 evaluation.

---

## 9. Machine Learning Models
Four supervised classification algorithms were trained and benchmarked:
1. **Logistic Regression** (L2 penalty, balanced class weight): Linear baseline to assess linear separability.
2. **Decision Tree Classifier** (`max_depth=6`): Non-linear tree baseline providing interpretable decision boundaries.
3. **Random Forest Classifier** (`n_estimators=150`, `max_depth=12`): Bagging ensemble reducing variance across randomized decision trees.
4. **Gradient Boosting Classifier** (`n_estimators=120`, `learning_rate=0.08`): Boosting ensemble focusing iteratively on hard-to-classify samples.

---

## 10. Model Evaluation & Comparison

Held-out test set performance ($N=656$ samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 52.44% | 41.46% | 53.12% | 0.4658 | 0.5284 |
| **Decision Tree** | 64.63% | 60.91% | 26.17% | 0.3661 | 0.5982 |
| **Random Forest (Baseline)** | **67.53%** | **72.16%** | 27.34% | 0.3966 | **0.6588** |
| **Gradient Boosting** | 65.85% | 64.55% | 27.73% | 0.3880 | 0.6512 |
| **Tuned Random Forest (Champion)** | 63.11% | 53.27% | **44.53%** | **0.4851** | **0.6611** |

### Precision vs. Recall Trade-Off
In water potability screening:
- **False Positive (Type I Error)**: The model predicts water is *Potable*, but it is actually contaminated. This presents severe human health hazards (waterborne illness).
- **False Negative (Type II Error)**: The model predicts water is *Unsafe*, but it is actually potable. This causes resource wastage and unnecessary secondary filtration.
- Therefore, tuning with balanced class weights significantly elevates **Recall** (from 27.3% to 44.5%) and **F1-Score** (from 0.39 to 0.485), achieving the highest overall **ROC-AUC (0.6611)**.

---

## 11. Explainable AI (SHAP)
The project integrates **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:
- **Global Explanations**: Identifies overall parameter importance across the population. Solids, Sulfate, pH, and Chloramines consistently rank as the top drivers of potability classification.
- **Local Explanations**: When an individual sample is submitted, the system computes sample-specific SHAP values:
  - Factors with **positive SHAP values** push the prediction towards **Potable**.
  - Factors with **negative SHAP values** push the prediction towards **Not Potable**.
- **Important Distinction**: SHAP describes the *mathematical attributions learned by the model* from the training dataset. It reflects statistical associations rather than direct medical or biochemical causation.

---

## 12. Streamlit Application
The web application provides an intuitive graphical interface organized into four dedicated views:
1. **Overview & Purpose**: Educational framing, parameter reference tables, and dataset distribution metrics.
2. **Single Sample Predictor**: Form with sliders and numeric inputs (calibrated against empirical dataset bounds), real-time potability badge, confidence meter, and dynamic Plotly horizontal bar chart of SHAP factors.
3. **Batch CSV Inference**: Upload lab test CSV files, execute vector predictions, preview results with summary metrics, and download the tagged CSV.
4. **Analytics & Visualizations**: High-resolution gallery displaying ROC curves, confusion matrices, correlation heatmaps, and SHAP summaries.

---

## 13. Installation

### Prerequisites
- Python 3.10+ (tested through Python 3.14)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/water-quality-potability-ml.git
cd water-quality-potability-ml

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 14. How to Run

### Step 1: Execute End-to-End Pipeline
```bash
# Preprocess data and export cleaned artifacts
python src/data_preprocessing.py

# Run multi-model evaluation and generate charts
python src/evaluate_model.py

# Run hyperparameter tuning and save champion model
python src/tune_model.py

# Compute global SHAP explanations
python src/explain_model.py
```

### Step 2: Launch Streamlit Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### Step 3: Run Automated Tests
```bash
pytest tests/test_prediction.py -v
```

---

## 15. Screenshots
Visualizations generated by the pipeline are stored in `visualizations/`:
- `visualizations/missing_values.png`
- `visualizations/target_distribution.png`
- `visualizations/feature_distributions.png`
- `visualizations/correlation_heatmap.png`
- `visualizations/model_comparison.png`
- `visualizations/confusion_matrix.png`
- `visualizations/roc_curve.png`
- `visualizations/shap_summary.png`

---

## 16. Results
- **Ensemble Dominance**: Tree ensembles consistently outperformed linear classifiers, confirming that water potability involves complex multi-parameter interactions.
- **Explainability**: SHAP explanations proved effective at identifying whether an unsafe prediction was primarily driven by excessive solids, low pH, or abnormal sulfate concentrations.
- **Performance Integrity**: All reported metrics are verified on an isolated 20% test set without leakage.

---

## 17. Limitations
- **Dataset Scope**: The dataset contains 3,276 entries from specific geographical sources; results may not generalize globally.
- **Missing Parameters**: Key biological hazards (e.g., *E. coli*, coliform bacteria, heavy metals like Arsenic or Lead) are not present in this physicochemical dataset.
- **Class Boundary Overlap**: Several potable and non-potable samples display overlapping physicochemical readings, limiting purely statistical accuracy.

---

## 18. Future Scope
Realistic planned enhancements:
1. **IoT Sensor Integration**: Connecting real-time micro-controller probes (ESP32/Arduino) to stream live turbidity and pH readings directly into the API.
2. **Broader Contaminant Tracking**: Incorporating heavy metal and microbiological assay columns into model retraining.
3. **Anomaly Detection**: Implementing isolation forests to flag sensor drift or contaminated instrument readings.
4. **Cloud Deployment**: Containerizing the Streamlit application for automated deployment to AWS / Streamlit Community Cloud.
5. **Multilingual Interface**: Providing localized interfaces for environmental health workers in regional areas.
6. **Regulatory Standard Mapping**: Automated warnings when parameters exceed specific regional guidelines (WHO, BIS, EPA).

---

## 19. Disclaimer
> **IMPORTANT NOTICE:** This software is an academic machine learning research and educational demonstration project. It is **NOT** a certified water testing instrument, clinical diagnostic tool, or medical safety device. Do not use predictions from this application as the sole basis for drinking water consumption or public health determinations. Always verify water potability through accredited municipal or laboratory testing procedures.

---

## 20. Author
- **Project Author**: B.Tech CSE / AIML Major Project Team
- **Institution / Affiliation**: Computer Science & Artificial Intelligence Department
- **License**: [MIT License](LICENSE)
