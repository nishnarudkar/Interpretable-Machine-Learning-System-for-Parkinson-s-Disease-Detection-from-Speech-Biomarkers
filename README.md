<div align="center">

# 🧠 Interpretable ML System for Parkinson's Disease Detection

### From Speech Biomarkers to Clinical Insights

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nishnarudkar/Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers/blob/main/notebooks/Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb)
&nbsp;
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.10-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.67-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6600)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-FF0000)](https://shap.readthedocs.io/)

<br/>

> **A production-grade MLOps pipeline that detects Parkinson's disease from voice recordings using interpretable machine learning, with full experiment tracking, data versioning, and a real-time explainable AI web application.**

</div>

---

## 👥 Authors

<div align="center">

| | Name | GitHub |
|---|---|---|
| 🧑‍💻 | Nishant Narudkar | [@nishnarudkar](https://github.com/nishnarudkar) |
| 🧑‍💻 | Maitreya Pawar | [@Metzo64](https://github.com/Metzo64) |
| 🧑‍💻 | Vatsal Parmar | [@Vatsal211005](https://github.com/Vatsal211005) |
| 🧑‍💻 | Aamir Sarang | [@Aamir-Sarang31](https://github.com/Aamir-Sarang31) |

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Model Results](#-model-results)
- [Explainability (SHAP)](#-explainability-shap)
- [MLOps Stack](#-mlops-stack)
- [Web Application](#-web-application)
- [API Reference](#-api-reference)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Docker Deployment](#-docker-deployment)
- [CI/CD with Jenkins](#-cicd-with-jenkins)
- [Technology Stack](#-technology-stack)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

Parkinson's disease is a progressive neurological disorder whose earliest symptoms often manifest as measurable changes in speech patterns. This project builds a full MLOps system that bridges research and production:

| Capability | Implementation |
|---|---|
| Multi-model comparison | 6 classifiers with RandomizedSearchCV tuning |
| Class imbalance handling | SMOTE inside leakage-free ImbPipeline CV folds |
| Feature selection | SelectFromModel (Random Forest, top 100 features) |
| Experiment tracking | MLflow + DagsHub (remote) |
| Model registry | MLflow Model Registry |
| Explainability | SHAP TreeExplainer (global + per-prediction) |
| Serving | FastAPI + dark-themed interactive UI |
| Data versioning | DVC backed by DagsHub |
| Containerisation | Docker multi-stage build |
| CI/CD | Jenkins 6-stage pipeline |

> **Production model:** XGBoost is selected over higher-scoring models (KNN, SVM) because SHAP `TreeExplainer` provides fast, exact feature attribution — critical for a medical application where clinicians need to understand *why* a prediction was made.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  pd_speech_features.csv  ──►  DVC  ──►  DagsHub Remote         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                      TRAINING PIPELINE                          │
│                                                                 │
│  Feature Selection          Scaling           SMOTE             │
│  SelectFromModel(RF)  ──►  StandardScaler  ──►  ImbPipeline     │
│                                                                 │
│  6 Models × RandomizedSearchCV × StratifiedKFold(5)            │
│  LR │ RF │ SVM │ KNN │ DT │ XGBoost                            │
│                                                                 │
│  Best Model (XGBoost) ──► MLflow Registry ──► DagsHub          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                      EXPLAINABILITY                             │
│  SHAP TreeExplainer ──► feature_importance.png                  │
│  Per-prediction SHAP ──► shap_bar.png + top 10 contributions    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                      SERVING LAYER                              │
│  FastAPI  ──►  /predict  ──►  Preprocessing  ──►  XGBoost      │
│  4-tab UI: Feature Importance │ Learning Curve │ Prediction     │
│            Model Comparison                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `data/pd_speech_features.csv` |
| Size | ~5.3 MB (DVC tracked) |
| Samples | 756 rows |
| Raw features | 755 columns (753 features + `id` + `class`) |
| Selected features | 100 (by Random Forest importance) |
| Target | Binary — `1` = Parkinson's, `0` = Healthy |
| Class distribution | 564 Parkinson's (74.6%) / 192 Healthy (25.4%) |

### Feature Groups

| Group | Description |
|---|---|
| `PPE`, `DFA`, `RPDE` | Nonlinear dynamical complexity measures |
| `numPulses`, `numPeriodsPulses` | Glottal pulse counts |
| `locPctJitter`, `locAbsJitter` | Jitter — frequency variation measures |
| `localShimmer`, `localdbShimmer` | Shimmer — amplitude variation measures |
| `mean_MFCC_*` | Mel-frequency cepstral coefficients |
| `tqwt_*` | Tunable Q-factor Wavelet Transform sub-band features (36 levels) |

---

## 📁 Project Structure

```
.
├── api/
│   └── main.py                      # FastAPI application + all endpoints
├── src/
│   ├── config.py                    # Central path + dataset configuration
│   ├── train.py                     # Training pipeline + MLflow logging
│   ├── explain.py                   # SHAP global feature importance
│   ├── learning_curve.py            # Bias-variance analysis
│   ├── model_selection.py           # Interpretable model selection logic
│   └── mlflow_comparison.py         # MLflow metrics fetcher for UI
├── data/
│   ├── pd_speech_features.csv       # Dataset (DVC tracked)
│   └── pd_speech_features.csv.dvc   # DVC pointer file
├── models/                          # Trained artifacts (gitignored)
│   ├── model.pkl                    # Production XGBoost model
│   ├── scaler.pkl                   # Fitted StandardScaler
│   ├── selector.pkl                 # Fitted SelectFromModel
│   ├── feature_names.pkl            # 100 selected feature names
│   └── column_order.pkl             # Training column order (753 features)
├── artifacts/
│   └── model_metrics.json           # Per-run comparison metrics
├── static/                          # Frontend assets + generated charts
│   ├── feature_importance.png       # SHAP global chart
│   ├── learning_curve.png           # Bias-variance plot
│   ├── feature_medians.json         # Feature defaults for prediction form
│   └── script.js / style.css        # UI assets
├── templates/
│   └── index.html                   # Jinja2 UI template
├── notebooks/
│   └── Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb
├── dvc.yaml                         # DVC pipeline definition
├── Dockerfile                       # Multi-stage container build
├── Jenkinsfile                      # 6-stage CI/CD pipeline
├── requirements.txt                 # Full pinned dependencies
├── requirements-api.txt             # API-only dependencies (Docker)
└── .env.example                     # Credentials template
```

---

## ⚙️ ML Pipeline

### Step 1 — Feature Selection

```python
# SMOTE applied first so RF selector learns minority class features
smote_fs = SMOTE(random_state=42)
X_train_fs, y_train_fs = smote_fs.fit_resample(X_train, y_train)

selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=100)
selector.fit(X_train_fs, y_train_fs)   # 753 → 100 features
```

### Step 2 — Scaling

`StandardScaler` applied after selection (select → scale). Fitted only on training data to prevent leakage.

### Step 3 — SMOTE inside ImbPipeline

```python
pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", SomeClassifier()),
])
RandomizedSearchCV(pipeline, param_dist, cv=StratifiedKFold(5), scoring="f1_macro")
```

SMOTE is applied **inside each CV fold** — validation folds always contain real, unaugmented data.

### Step 4 — Model Training

| Model | Data | Search Iterations | Key Params Tuned |
|---|---|---|---|
| Logistic Regression | Scaled | 20 | C, solver |
| Random Forest | Unscaled | 20 | n_estimators, max_depth, max_features |
| SVM | Scaled | 15 | C, gamma, kernel |
| KNN | Scaled | 15 | n_neighbors, weights, metric |
| Decision Tree | Unscaled | 20 | max_depth, criterion, min_samples |
| XGBoost | Unscaled | 30 | max_depth, gamma, subsample, colsample_bytree |

---

## 📈 Model Results

| Model | Accuracy | Macro F1 | ROC AUC | Selected |
|---|---|---|---|---|
| **XGBoost** | **0.89** | **0.855** | **0.946** | ✅ Production |
| SVM | 0.87 | 0.833 | 0.920 | |
| Random Forest | 0.87 | 0.828 | 0.931 | |
| KNN | 0.83 | 0.804 | 0.947 | |
| Logistic Regression | 0.82 | 0.776 | 0.859 | |
| Decision Tree | 0.84 | 0.766 | 0.747 | |

All metrics are macro-averaged on the held-out test set (152 samples). XGBoost is selected as the production model for interpretability, not just raw performance — SHAP `TreeExplainer` provides fast, exact feature attribution essential for clinical transparency.

### Best XGBoost Configuration

```
max_depth        : 5       colsample_bytree : 0.8
min_child_weight : 1       n_estimators     : 300
gamma            : 0       learning_rate    : 0.05
subsample        : 0.8
```

---

## 🔬 Explainability (SHAP)

### Global Feature Importance

- `TreeExplainer` computes mean absolute SHAP values across 100 sampled rows
- Top 20 most influential speech biomarkers plotted and saved to `static/feature_importance.png`
- Displayed in the Feature Importance tab

### Per-Prediction Explanation

Every `/predict` call returns:
- Top 10 SHAP feature contributions with actual feature names
- A server-generated `shap_bar.png` (dark-themed horizontal bar chart)
- Color coding: 🔴 red = pushes toward Parkinson's, 🟢 green = pushes toward Healthy

```json
{
  "feature_name": "tqwt_kurtosisValue_dec_5",
  "impact": 0.2341
}
```

### Learning Curve

Full `ImbPipeline` (SMOTE → SelectFromModel → StandardScaler → model) refitted per CV fold — no leakage. Plots train vs. validation macro F1 with ±1 std confidence bands.

---

## 🛠 MLOps Stack

### MLflow + DagsHub

```python
dagshub.init(repo_owner="nishnarudkar", repo_name="...", mlflow=True)
mlflow.set_experiment("parkinson_detection")
```

Each run logs: `accuracy`, `macro_f1`, `roc_auc`, `precision`, `recall`, hyperparameters, and the serialised model. XGBoost is registered atomically:

```python
mlflow.sklearn.log_model(
    best_tuned_xgb,
    name="model",
    registered_model_name="parkinson_detection_model",
)
```

### DVC Pipeline

```yaml
stages:
  train:          python src/train.py          → models/*.pkl, artifacts/model_metrics.json
  explain:        python src/explain.py         → static/feature_importance.png
  learning_curve: python src/learning_curve.py  → static/learning_curve.png
```

Run with: `dvc repro`

---

## 🌐 Web Application

Four-tab dark-themed UI served by FastAPI + Jinja2:

| Tab | Content |
|---|---|
| 📊 Feature Importance | Global SHAP chart + top 5 biomarkers ranked list |
| 📈 Learning Curve | Bias-variance plot with confidence bands + legend |
| 🔬 Prediction | 5-input form (top SHAP features) + result card + SHAP explanation |
| 📋 Model Comparison | Live leaderboard from MLflow metrics |

**UI features:** dark theme, loading spinner, animated probability bar, color-coded results, responsive grid layout, feature range hints, tooltip explanations.

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok", "model": "XGBClassifier", "explainer": "TreeExplainer", "model_loaded": true }
```

### `POST /predict`

**Request:**
```json
{ "features": [753 floats in training column order] }
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Parkinson's Detected",
  "probability": 0.923,
  "top_contributions": [
    { "feature_index": 42, "feature_name": "tqwt_kurtosisValue_dec_5", "impact": 0.2341 }
  ],
  "shap_bar_url": "/static/shap_bar.png"
}
```

### `GET /feature-defaults`
Returns top 5 SHAP-ranked features with `median`, `min`, `max` values and all 753 column defaults for the prediction form.

### `GET /model-comparison`
Returns live model metrics from `artifacts/model_metrics.json` (or MLflow fallback), sorted by ROC AUC.

### `GET /top-features`
Returns top 5 globally important features by mean absolute SHAP value.

---

## 🚀 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/nishnarudkar/Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers.git
cd Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
copy .env.example .env
# Edit .env — fill in DAGSHUB_USERNAME and DAGSHUB_TOKEN

# 5. Pull dataset via DVC
dvc pull
```

---

## ▶️ Running the Project

### Option A — Step by step

```bash
python src/train.py          # Train all 6 models, save artifacts
python src/explain.py        # Generate SHAP feature importance chart
python src/learning_curve.py # Generate bias-variance learning curve
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Option B — DVC pipeline (recommended)

```bash
dvc repro
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

### What each step produces

| Script | Outputs |
|---|---|
| `train.py` | `models/*.pkl`, `artifacts/model_metrics.json`, `static/feature_medians.json` |
| `explain.py` | `static/feature_importance.png` |
| `learning_curve.py` | `static/learning_curve.png` |
| `uvicorn` | Live web app at port 8000 |

---

## 🐳 Docker Deployment

The Dockerfile uses a **multi-stage build** — only API dependencies are installed in the runtime image, keeping it lean.

```bash
# Build
docker build -t parkinson-ml .

# Run
docker run -p 8000:8000 parkinson-ml
```

Open **http://localhost:8000**

---

## 🔄 CI/CD with Jenkins

The `Jenkinsfile` defines a **6-stage automated pipeline** triggered on every push:

```
Install Deps → Pull Data (DVC) → Train → Generate Charts → Smoke Test → Build Docker
```

| Stage | Command |
|---|---|
| Install Dependencies | `pip install -r requirements.txt` |
| Pull Data (DVC) | `dvc pull` |
| Train Model | `python src/train.py` |
| Generate Explanations | `python src/explain.py && python src/learning_curve.py` |
| Smoke Test | `curl -f http://localhost:8000/health` |
| Build Docker Image | `docker build -t parkinson-ml .` |

Credentials (`DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`) are injected via Jenkins credentials store — never hardcoded in source.

---

## 🧰 Technology Stack

| Category | Tools |
|---|---|
| ML / Data | scikit-learn, XGBoost, pandas, numpy, imbalanced-learn, scipy |
| Explainability | SHAP (TreeExplainer + KernelExplainer) |
| Experiment Tracking | MLflow 3.x, DagsHub |
| Data Versioning | DVC 3.x |
| Web Framework | FastAPI, Uvicorn, Jinja2 |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Containerisation | Docker (multi-stage) |
| CI/CD | Jenkins |
| Visualisation | matplotlib, seaborn |
| Notebook | Google Colab |

---

## 📡 Automated Monitoring & Data Drift Detection

### What is Data Drift?

Data drift occurs when the statistical distribution of production input data shifts away from the training data distribution. In a medical ML system, this can happen when:
- Patient demographics change over time
- Recording equipment or protocols change
- New speech patterns emerge in the population

### Why It Matters in Healthcare ML

In clinical applications, a drifted model may silently produce incorrect predictions without any obvious error. Unlike software bugs, drift is invisible without active monitoring — making it especially dangerous in a Parkinson's detection context where false negatives could delay diagnosis.

### How Evidently Is Used

[Evidently](https://www.evidentlyai.com/) compares the **baseline distribution** (training data) against **current production inputs** logged by the API.

```
monitoring/
├── baseline_data.csv    # X_train features saved after each training run
├── current_data.csv     # API inputs appended on every /predict call
└── drift_report.html    # Generated HTML report (interactive)
```

Run drift detection manually:
```bash
python monitoring/drift_check.py
```

The script:
1. Loads `baseline_data.csv` as the reference distribution
2. Loads `current_data.csv` (production inputs)
3. Runs `DataDriftPreset` from Evidently
4. Saves an interactive HTML report to `monitoring/drift_report.html`
5. Exits gracefully if current data is missing or insufficient

### How Jenkins Automates It

The `Jenkinsfile` includes a **Drift Detection** stage that runs after every build:

```
Train → Explain → Smoke Test → Build Docker → Drift Detection
```

The drift report is archived as a Jenkins build artifact, making it accessible from the Jenkins UI after every pipeline run.

### MLflow Local Tracking

Every prediction is logged to a local MLflow run (no server required):
- **Param:** `model_type` (e.g., `XGBClassifier`)
- **Metrics:** `prediction` (0 or 1), `probability` (confidence score)

View locally with:
```bash
mlflow ui
```

Open `http://localhost:5000` to browse prediction history.

---

This project is intended for **research and educational purposes only**. It is not a validated medical diagnostic tool. Do not use predictions from this system for clinical decision-making. Always consult a qualified healthcare professional for medical advice.
