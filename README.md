# Interpretable Machine Learning System for Parkinson's Disease Detection from Speech Biomarkers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nishnarudkar/Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers/blob/main/notebooks/Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb)

An end-to-end MLOps pipeline that detects Parkinson's disease from voice recordings using interpretable machine learning. The system combines rigorous model experimentation, SHAP-based explainability, full experiment tracking via MLflow + DagsHub, data versioning with DVC, and a real-time web application served by FastAPI.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [ML Pipeline](#ml-pipeline)
5. [Model Results](#model-results)
6. [Explainability (SHAP)](#explainability-shap)
7. [MLOps Stack](#mlops-stack)
8. [Web Application](#web-application)
9. [API Reference](#api-reference)
10. [Installation & Setup](#installation--setup)
11. [Running the Project](#running-the-project)
12. [Docker Deployment](#docker-deployment)
13. [CI/CD with Jenkins](#cicd-with-jenkins)
14. [Technology Stack](#technology-stack)
15. [Disclaimer](#disclaimer)

---

## Project Overview

Parkinson's disease is a progressive neurological disorder. One of its earliest and most measurable symptoms is changes in speech patterns. This project builds a full MLOps system that:

- Trains and compares 6 classifiers on 753 speech biomarker features
- Handles class imbalance using **SMOTE** inside leakage-free `ImbPipeline` CV folds
- Selects the top 100 most informative features using **SelectFromModel** (RandomForest)
- Tunes all models with **RandomizedSearchCV** for efficient hyperparameter search
- Tracks every experiment with **MLflow** synced to **DagsHub**
- Registers the best model in the **MLflow Model Registry**
- Generates global and local explanations using **SHAP** (TreeExplainer or KernelExplainer depending on model type)
- Serves predictions and explanations through a **FastAPI** web app with an interactive dark-themed UI
- Versions the dataset with **DVC** backed by DagsHub remote storage
- Automates training and deployment via **Jenkins** and **Docker**

---

## Dataset

**File:** `data/pd_speech_features.csv`
**Size:** ~5.3 MB (DVC tracked)
**Shape:** 756 rows × 755 columns (753 features + 1 target after dropping `id`)

### Feature Groups

The dataset contains a rich set of speech biomarkers extracted from sustained phonation recordings:

| Feature Group | Description |
|---|---|
| `PPE`, `DFA`, `RPDE` | Nonlinear dynamical complexity measures |
| `numPulses`, `numPeriodsPulses` | Glottal pulse counts |
| `meanPeriodPulses`, `stdDevPeriodPulses` | Pulse period statistics |
| `locPctJitter`, `locAbsJitter`, etc. | Jitter (frequency variation) measures |
| `localShimmer`, `localdbShimmer`, etc. | Shimmer (amplitude variation) measures |
| `tqwt_*` | Tunable Q-factor Wavelet Transform sub-band features (energy, entropy, kurtosis across 36 decomposition levels) |

### Class Distribution

| Class | Label | Count |
|---|---|---|
| 1 | Parkinson's | 564 (74.6%) |
| 0 | Healthy | 192 (25.4%) |

The dataset is imbalanced (~3:1 ratio), which is addressed using SMOTE during training.

---

## Project Structure

```
.
├── data/
│   ├── pd_speech_features.csv          # Dataset (DVC tracked)
│   └── pd_speech_features.csv.dvc      # DVC pointer file
├── src/
│   ├── config.py                       # Central path + dataset config
│   ├── train.py                        # Model training + MLflow logging
│   ├── explain.py                      # SHAP global feature importance
│   └── learning_curve.py               # Bias-variance learning curve
├── api/
│   └── main.py                         # FastAPI app (prediction + UI serving)
├── models/
│   ├── model.pkl                       # Best trained model
│   ├── scaler.pkl                      # Fitted StandardScaler
│   ├── selector.pkl                    # Fitted SelectFromModel selector
│   ├── feature_names.pkl               # Names of the 100 selected features
│   └── column_order.pkl                # Training column order for serving alignment
├── static/
│   ├── feature_importance.png          # SHAP bar chart (generated)
│   ├── learning_curve.png              # Learning curve plot (generated)
│   ├── script.js                       # Frontend JavaScript
│   └── style.css                       # Dark-themed CSS
├── templates/
│   └── index.html                      # Jinja2 HTML template
├── notebooks/
│   └── Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb  # Full EDA + experimentation
├── dvc.yaml                            # DVC pipeline definition
├── Dockerfile                          # Multi-stage container image
├── Jenkinsfile                         # CI/CD pipeline
├── requirements.txt                    # Full training + API dependencies (pinned)
├── requirements-api.txt                # API-only dependencies for Docker
├── .env.example                        # Environment variable template
└── .dvc/config                         # DVC remote (DagsHub)
```

---

## ML Pipeline

### 1. Data Loading

Data is loaded via `src/config.py` which resolves paths relative to the project root (works in Docker, CI, and local dev) and validates the CSV structure:

```python
X, y = load_dataset()   # 753 features, binary target
```

### 2. Train/Test Split

- 80/20 stratified split (`random_state=42`)
- Stratification preserves the 74.6% / 25.4% class ratio in both splits
- Train: 604 samples | Test: 152 samples

### 3. Preprocessing

**Feature Selection** — `SelectFromModel(RandomForestClassifier, max_features=100)`
- Trains a Random Forest on SMOTE-balanced data to learn feature importances
- Selects the top 100 features by importance score
- Reduces dimensionality from 753 → 100 features
- Fitted only on training data to prevent leakage

```python
smote_fs = SMOTE(random_state=42)
X_train_fs, y_train_fs = smote_fs.fit_resample(X_train, y_train)

selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=100)
selector.fit(X_train_fs, y_train_fs)
```

**Scaling** — `StandardScaler`
- Applied after feature selection (select → scale order)
- Zero mean, unit variance normalization on the 100 selected features
- Fitted only on training data

### 4. Class Imbalance — SMOTE inside ImbPipeline

SMOTE is applied **inside each model's `ImbPipeline`** during cross-validation to avoid data leakage. It is never applied to the full training set before CV:

```python
pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", SomeClassifier()),
])
RandomizedSearchCV(pipeline, param_dist, cv=StratifiedKFold(5), ...)
```

This ensures the validation fold in each CV split is always real, unaugmented data.

### 5. Model Training — RandomizedSearchCV

All 6 models are tuned with `RandomizedSearchCV` (not GridSearchCV) for efficient hyperparameter search. Each model uses a leakage-free `ImbPipeline` with SMOTE inside:

| Model | Data | Search Iterations |
|---|---|---|
| Logistic Regression | scaled | 20 |
| Random Forest | unscaled | 20 |
| SVM | scaled | 15 |
| KNN | scaled | 15 |
| Decision Tree | unscaled | 20 |
| XGBoost | unscaled | 30 |

All runs are logged to MLflow. The best model by macro F1 is registered in the MLflow Model Registry as `parkinson_detection_model` and saved to `models/`.

---

## Model Results

Results from the notebook experimentation (with SMOTE + RandomizedSearchCV):

| Model | Accuracy | Macro F1 | ROC AUC |
|---|---|---|---|
| **XGBoost (tuned)** | **0.89** | **0.855** | **0.946** |
| SVM | 0.87 | 0.833 | 0.920 |
| Random Forest | 0.87 | 0.828 | 0.931 |
| KNN | 0.83 | 0.804 | 0.947 |
| Logistic Regression | 0.82 | 0.776 | 0.859 |
| Decision Tree | 0.84 | 0.766 | 0.747 |

All metrics are macro-averaged. Selection criterion is macro F1 (not accuracy) to account for class imbalance.

### Best XGBoost Configuration (RandomizedSearchCV)

```
colsample_bytree : 0.8
gamma            : 0
max_depth        : 5
min_child_weight : 1
n_estimators     : 300
subsample        : 0.8
learning_rate    : 0.05
```

### Tuned XGBoost Classification Report

```
              precision  recall  f1-score  support
           0       0.78    0.79      0.78       39
           1       0.93    0.92      0.92      113
    accuracy                         0.89      152
   macro avg       0.85    0.86      0.85      152
weighted avg       0.89    0.89      0.89      152
```

---

## Explainability (SHAP)

The system automatically selects the appropriate SHAP explainer based on the winning model type.

### Explainer Selection

```python
TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
               DecisionTreeClassifier, XGBClassifier)

if isinstance(model, TREE_MODELS):
    explainer = shap.TreeExplainer(model)      # fast, exact
else:
    explainer = shap.KernelExplainer(model.predict_proba, background)  # model-agnostic
```

### Global Feature Importance (`src/explain.py`)

- Samples 50 rows from the dataset for speed
- Computes mean absolute SHAP values across all samples
- Plots the top 20 most influential speech biomarkers
- Saves to `static/feature_importance.png`

### Local Prediction Explanation (`api/main.py`)

For every prediction, the API returns the top 10 feature contributions with actual feature names:

```json
{
  "feature_index": 42,
  "feature_name": "tqwt_kurtosisValue_dec_5",
  "impact": 0.2341
}
```

- Positive SHAP value → pushes prediction toward Parkinson's
- Negative SHAP value → pushes prediction toward Healthy
- Displayed as a color-coded horizontal bar chart in the UI

### Learning Curve (`src/learning_curve.py`)

- Full `ImbPipeline` (SMOTE → SelectFromModel → StandardScaler → model) refitted per fold
- 5-fold stratified CV, macro F1 scoring
- Plots train vs. validation score with ±1 std confidence bands
- Annotates the train/val gap at the largest training size
- Saves to `static/learning_curve.png`

---

## MLOps Stack

### MLflow + DagsHub

All experiments are tracked remotely via DagsHub. Credentials are loaded from environment variables:

```bash
export DAGSHUB_USERNAME=your_username
export DAGSHUB_TOKEN=your_token
```

Each run logs:
- **Params:** model name, hyperparameters, num_features
- **Metrics:** accuracy, macro_precision, macro_recall, macro_f1, roc_auc (all macro-averaged)
- **Artifacts:** serialized model (sklearn flavor)

### DVC Pipeline (`dvc.yaml`)

```yaml
stages:
  train:
    cmd: python src/train.py
    deps: [src/train.py, data/pd_speech_features.csv]
    outs: [models/model.pkl, models/scaler.pkl, models/selector.pkl,
           models/feature_names.pkl, models/column_order.pkl]

  explain:
    cmd: python src/explain.py
    deps: [src/explain.py, models/model.pkl, models/scaler.pkl, models/selector.pkl]
    outs: [static/feature_importance.png]

  learning_curve:
    cmd: python src/learning_curve.py
    deps: [src/learning_curve.py, models/model.pkl, data/pd_speech_features.csv]
    outs: [static/learning_curve.png]
```

Run the full pipeline with:

```bash
dvc repro
```

---

## Web Application

The FastAPI app serves a dark-themed, three-tab interactive UI.

### Tabs

**Feature Importance** — Global SHAP bar chart of the top 20 speech biomarkers

**Learning Curve** — Bias-variance analysis with confidence bands

**Prediction** — Enter 753 comma-separated feature values to get:
- Prediction label (Parkinson's / Healthy)
- Confidence probability bar
- Top 10 SHAP feature contributions with actual feature names

### UI Features

- Dark theme with purple accent colors
- Loading spinner during API calls
- Input validation with user-friendly error messages
- Animated probability bar
- Responsive layout (mobile-friendly)
- Chart.js for dynamic SHAP visualization

---

## API Reference

### `GET /`

Returns the main HTML interface.

### `GET /health`

Returns API health status.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`

Runs inference on a feature vector and returns a SHAP explanation.

**Request body:**

```json
{
  "features": [0.85247, 0.71826, 0.57227, 240, 239, 0.008064, ...]
}
```

Exactly 753 numeric values required (excluding `id` and `class`). Pydantic enforces this at the schema level — wrong length returns `422 Unprocessable Entity`.

**Response:**

```json
{
  "prediction": 1,
  "label": "Parkinson's Detected",
  "probability": 0.923,
  "top_contributions": [
    { "feature_index": 42, "feature_name": "tqwt_kurtosisValue_dec_5", "impact": 0.2341 },
    { "feature_index": 15, "feature_name": "PPE", "impact": -0.1823 }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | int | `1` = Parkinson's, `0` = Healthy |
| `label` | string | Human-readable prediction label |
| `probability` | float | Probability of Parkinson's (class 1) |
| `top_contributions` | list | Top 10 SHAP feature impacts |
| `feature_name` | string | Actual biomarker name |
| `impact` | float | SHAP value (positive = toward Parkinson's) |

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip
- Docker (optional)
- Jenkins (optional)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/nishnarudkar/Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers.git
cd Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set credentials**

```bash
cp .env.example .env
# Edit .env and fill in DAGSHUB_USERNAME and DAGSHUB_TOKEN
```

**5. Pull the dataset via DVC**

```bash
dvc pull
```

---

## Running the Project

### Option A — Run each step manually

```bash
python src/train.py          # trains all 6 models, saves best to models/
python src/explain.py        # generates static/feature_importance.png
python src/learning_curve.py # generates static/learning_curve.png
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

### Option B — Run via DVC pipeline

```bash
dvc repro
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Option C — Open the notebook

`notebooks/Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb` contains the full exploratory workflow:

- EDA and class distribution analysis
- SMOTE-based oversampling with leakage-free ImbPipeline
- Feature selection with `SelectFromModel` (Random Forest)
- Training and comparing 6 models: LR, RF, SVM, XGBoost, KNN, Decision Tree
- RandomizedSearchCV tuning with StratifiedKFold
- Confusion matrix, ROC curve, and learning curve visualization
- SHAP analysis

---

## Docker Deployment

The Dockerfile uses a multi-stage build — only API dependencies are installed in the runtime image.

```bash
docker build -t parkinson-ml .
docker run -p 8000:8000 parkinson-ml
```

The app will be available at `http://localhost:8000`.

---

## CI/CD with Jenkins

The `Jenkinsfile` defines a five-stage pipeline:

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
| Containerization | Docker (multi-stage) |
| CI/CD | Jenkins |
| Visualization | matplotlib, seaborn |
| Notebook | Google Colab |

---

## Disclaimer

This project is intended for **research and educational purposes only**. It is not a validated medical diagnostic tool. Do not use predictions from this system for clinical decision-making. Always consult a qualified healthcare professional for medical advice.
