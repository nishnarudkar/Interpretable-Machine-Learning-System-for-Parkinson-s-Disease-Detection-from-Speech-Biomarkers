# Interpretable Machine Learning System for Parkinson's Disease Detection from Speech Biomarkers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nishnarudkar/Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers/blob/main/notebooks/Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.10-orange.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.67-purple.svg)](https://dvc.org/)

---

## Authors

| Name | GitHub |
|---|---|
| Nishant Narudkar | [@nishnarudkar](https://github.com/nishnarudkar) |
| Maitreya Pawar | [@Metzo64](https://github.com/Metzo64) |
| Vatsal Parmar | [@Vatsal211005](https://github.com/Vatsal211005) |
| Aamir Sarang | [@Aamir-Sarang31](https://github.com/Aamir-Sarang31) |

---

## Overview

Parkinson's disease is a progressive neurological disorder whose earliest symptoms often manifest as measurable changes in speech patterns. This project delivers a production-grade MLOps system that:

- Trains and compares **6 classifiers** on 753 speech biomarker features extracted from voice recordings
- Handles class imbalance with **SMOTE** applied inside leakage-free `ImbPipeline` CV folds
- Selects the top 100 features using **SelectFromModel** (Random Forest)
- Tunes all models with **RandomizedSearchCV** for efficient hyperparameter search
- Tracks every experiment with **MLflow** synced to **DagsHub**
- Registers the production model in the **MLflow Model Registry**
- Generates global and local explanations using **SHAP** (TreeExplainer)
- Serves predictions through a **FastAPI** web application with a dark-themed interactive UI
- Versions the dataset with **DVC** backed by DagsHub remote storage
- Automates the full pipeline with **Jenkins** CI/CD and **Docker** containerisation

> **Note:** XGBoost is selected as the production model for its balance of performance and interpretability. SHAP TreeExplainer provides fast, exact feature attribution — essential for a medical application.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [ML Pipeline](#ml-pipeline)
4. [Model Results](#model-results)
5. [Explainability](#explainability)
6. [MLOps Stack](#mlops-stack)
7. [Web Application](#web-application)
8. [API Reference](#api-reference)
9. [Installation & Setup](#installation--setup)
10. [Running the Project](#running-the-project)
11. [Docker Deployment](#docker-deployment)
12. [CI/CD with Jenkins](#cicd-with-jenkins)
13. [Technology Stack](#technology-stack)
14. [Disclaimer](#disclaimer)

---

## Dataset

| Property | Value |
|---|---|
| File | `data/pd_speech_features.csv` |
| Size | ~5.3 MB (DVC tracked) |
| Samples | 756 rows |
| Features | 753 speech biomarkers |
| Target | Binary (`1` = Parkinson's, `0` = Healthy) |
| Class ratio | 74.6% Parkinson's / 25.4% Healthy |

### Feature Groups

| Group | Description |
|---|---|
| `PPE`, `DFA`, `RPDE` | Nonlinear dynamical complexity measures |
| `numPulses`, `numPeriodsPulses` | Glottal pulse counts |
| `locPctJitter`, `locAbsJitter`, etc. | Jitter (frequency variation) |
| `localShimmer`, `localdbShimmer`, etc. | Shimmer (amplitude variation) |
| `mean_MFCC_*` | Mel-frequency cepstral coefficients |
| `tqwt_*` | Tunable Q-factor Wavelet Transform sub-band features |

---

## Project Structure

```
.
├── api/
│   └── main.py                     # FastAPI application
├── src/
│   ├── config.py                   # Central path + dataset configuration
│   ├── train.py                    # Model training + MLflow logging
│   ├── explain.py                  # SHAP global feature importance
│   ├── learning_curve.py           # Bias-variance analysis
│   ├── model_selection.py          # Interpretable model selection logic
│   └── mlflow_comparison.py        # MLflow metrics fetcher
├── data/
│   ├── pd_speech_features.csv      # Dataset (DVC tracked)
│   └── pd_speech_features.csv.dvc  # DVC pointer
├── models/                         # Trained artifacts (gitignored)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   ├── feature_names.pkl
│   └── column_order.pkl
├── artifacts/
│   └── model_metrics.json          # Per-run comparison metrics
├── static/                         # Frontend assets + generated charts
├── templates/
│   └── index.html                  # Jinja2 UI template
├── notebooks/
│   └── Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb
├── dvc.yaml                        # DVC pipeline
├── Dockerfile                      # Multi-stage container build
├── Jenkinsfile                     # CI/CD pipeline
├── requirements.txt                # Full dependencies (pinned)
├── requirements-api.txt            # API-only dependencies for Docker
└── .env.example                    # Credentials template
```

---

## ML Pipeline

### 1. Data Loading

Loaded via `src/config.py` — paths resolve relative to project root (works locally, in Docker, and in CI):

```python
X, y = load_dataset()   # 753 features, binary target
```

### 2. Train/Test Split

- 80/20 stratified split (`random_state=42`)
- Train: 604 samples | Test: 152 samples

### 3. Feature Selection

`SelectFromModel(RandomForestClassifier, max_features=100)` — trained on SMOTE-balanced data to learn minority class feature importances, then applied to the original split to avoid leakage.

### 4. Scaling

`StandardScaler` applied after selection (select → scale order). Fitted only on training data.

### 5. Class Imbalance

SMOTE applied **inside each model's `ImbPipeline`** during cross-validation — never on the full training set before CV. This ensures validation folds always contain real, unaugmented data.

### 6. Hyperparameter Tuning

All 6 models tuned with `RandomizedSearchCV` + `StratifiedKFold(5)`:

| Model | Input | Iterations |
|---|---|---|
| Logistic Regression | Scaled | 20 |
| Random Forest | Unscaled | 20 |
| SVM | Scaled | 15 |
| KNN | Scaled | 15 |
| Decision Tree | Unscaled | 20 |
| XGBoost | Unscaled | 30 |

---

## Model Results

| Model | Accuracy | Macro F1 | ROC AUC |
|---|---|---|---|
| **XGBoost** ⭐ | **0.89** | **0.855** | **0.946** |
| SVM | 0.87 | 0.833 | 0.920 |
| Random Forest | 0.87 | 0.828 | 0.931 |
| KNN | 0.83 | 0.804 | 0.947 |
| Logistic Regression | 0.82 | 0.776 | 0.859 |
| Decision Tree | 0.84 | 0.766 | 0.747 |

All metrics are macro-averaged on the held-out test set. XGBoost is selected as the production model for interpretability, not just raw performance.

---

## Explainability

### Global (Feature Importance tab)

SHAP `TreeExplainer` computes mean absolute SHAP values across 100 sampled rows. The top 20 most influential speech biomarkers are plotted and saved to `static/feature_importance.png`.

### Local (per-prediction)

Every prediction returns the top 10 SHAP feature contributions with actual feature names:

```json
{
  "feature_name": "tqwt_kurtosisValue_dec_5",
  "impact": 0.2341
}
```

Positive values push toward Parkinson's; negative values push toward Healthy. A server-generated `shap_bar.png` is also produced per prediction.

---

## MLOps Stack

### MLflow + DagsHub

Credentials loaded from environment variables:

```bash
export DAGSHUB_USERNAME=your_username
export DAGSHUB_TOKEN=your_token
```

Each run logs: params, macro_f1, roc_auc, accuracy, precision, recall, and the serialised model. The production XGBoost model is registered in the MLflow Model Registry as `parkinson_detection_model`.

### DVC Pipeline

```yaml
stages:
  train:       python src/train.py        → models/*.pkl, artifacts/model_metrics.json
  explain:     python src/explain.py      → static/feature_importance.png
  learning_curve: python src/learning_curve.py → static/learning_curve.png
```

Run with: `dvc repro`

---

## Web Application

Four-tab dark-themed UI:

| Tab | Content |
|---|---|
| Feature Importance | Global SHAP chart + top 5 biomarkers list |
| Learning Curve | Bias-variance plot with confidence bands |
| Prediction | 5-input form (top SHAP features) + result + SHAP explanation |
| Model Comparison | Live leaderboard from MLflow metrics |

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "model": "XGBClassifier", "explainer": "TreeExplainer" }
```

### `POST /predict`

**Request:** `{ "features": [753 floats in training column order] }`

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
Returns top 5 SHAP-ranked features with median, min, max values and all 753 column defaults.

### `GET /model-comparison`
Returns live model metrics from `artifacts/model_metrics.json` or MLflow.

### `GET /top-features`
Returns top 5 globally important features by SHAP importance.

---

## Installation & Setup

```bash
# 1. Clone
git clone https://github.com/nishnarudkar/Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers.git
cd Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set credentials
copy .env.example .env         # then fill in DAGSHUB_USERNAME and DAGSHUB_TOKEN

# 5. Pull dataset
dvc pull
```

---

## Running the Project

```bash
# Train all models (writes models/, artifacts/, static/feature_medians.json)
python src/train.py

# Generate SHAP feature importance chart
python src/explain.py

# Generate learning curve chart
python src/learning_curve.py

# Start the web app
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or run the full pipeline in one command:

```bash
dvc repro
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`

---

## Docker Deployment

The Dockerfile uses a multi-stage build — only API dependencies are installed in the runtime image, keeping it lean.

```bash
docker build -t parkinson-ml .
docker run -p 8000:8000 parkinson-ml
```

---

## CI/CD with Jenkins

The `Jenkinsfile` defines a six-stage automated pipeline:

| Stage | Command |
|---|---|
| Install Dependencies | `pip install -r requirements.txt` |
| Pull Data (DVC) | `dvc pull` |
| Train Model | `python src/train.py` |
| Generate Explanations | `python src/explain.py && python src/learning_curve.py` |
| Smoke Test | `curl -f http://localhost:8000/health` |
| Build Docker Image | `docker build -t parkinson-ml .` |

Credentials (`DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`) are injected via Jenkins credentials store — never hardcoded.

---

## Technology Stack

| Category | Tools |
|---|---|
| ML / Data | scikit-learn, XGBoost, pandas, numpy, imbalanced-learn, scipy |
| Explainability | SHAP (TreeExplainer + KernelExplainer) |
| Experiment Tracking | MLflow, DagsHub |
| Data Versioning | DVC |
| Web Framework | FastAPI, Uvicorn, Jinja2 |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Containerisation | Docker (multi-stage) |
| CI/CD | Jenkins |
| Visualisation | matplotlib, seaborn |
| Notebook | Google Colab |

---

## Disclaimer

This project is intended for **research and educational purposes only**. It is not a validated medical diagnostic tool. Do not use predictions from this system for clinical decision-making. Always consult a qualified healthcare professional for medical advice.
