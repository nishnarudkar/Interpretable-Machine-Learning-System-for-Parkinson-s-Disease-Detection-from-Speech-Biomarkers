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

- Trains and compares multiple classifiers on 753 speech biomarker features
- Handles class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique)
- Selects the top 100 most informative features using **SelectKBest** (f_classif)
- Tracks every experiment with **MLflow** synced to **DagsHub**
- Registers the best model in the **MLflow Model Registry**
- Generates global and local explanations using **SHAP TreeExplainer**
- Serves predictions and explanations through a **FastAPI** web app with an interactive dark-themed UI
- Versions the dataset with **DVC** backed by DagsHub remote storage
- Automates training and deployment via **Jenkins** and **Docker**

---

## Dataset

**File:** `data/pd_speech_features.csv`
**Size:** ~5.3 MB (DVC tracked)
**Shape:** 756 rows × 755 columns (754 features + 1 target after dropping `id`)

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
│   ├── train.py                        # Model training + MLflow logging
│   ├── explain.py                      # SHAP global feature importance
│   └── learning_curve.py               # Bias-variance learning curve
├── api/
│   └── main.py                         # FastAPI app (prediction + UI serving)
├── models/
│   ├── model.pkl                       # Best trained model
│   ├── scaler.pkl                      # Fitted StandardScaler
│   └── selector.pkl                    # Fitted SelectKBest selector
├── static/
│   ├── feature_importance.png          # SHAP bar chart (generated)
│   ├── learning_curve.png              # Learning curve plot (generated)
│   ├── script.js                       # Frontend JavaScript
│   └── style.css                       # Dark-themed CSS
├── templates/
│   └── index.html                      # Jinja2 HTML template
├── notebooks/
│   └── Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb  # Full EDA + experimentation
├── mlruns/                             # MLflow local experiment store
├── dvc.yaml                            # DVC pipeline definition
├── Dockerfile                          # Container image definition
├── Jenkinsfile                         # CI/CD pipeline
├── requirements.txt                    # Python dependencies
└── .dvc/config                         # DVC remote (DagsHub)
```

---

## ML Pipeline

### 1. Data Loading

```python
df = pd.read_csv("data/pd_speech_features.csv", header=1)
df = df.drop("id", axis=1)
X = df.drop("class", axis=1)   # 753 features
y = df["class"]                 # binary target
```

### 2. Train/Test Split

- 80/20 stratified split (`random_state=42`)
- Stratification preserves the 74.6% / 25.4% class ratio in both splits
- Train: 604 samples | Test: 152 samples

### 3. Preprocessing

**Feature Selection** — `SelectKBest(f_classif, k=100)`
- Selects the top 100 features by ANOVA F-statistic
- Reduces dimensionality from 753 → 100 features
- Fitted only on training data to prevent leakage

**Scaling** — `StandardScaler`
- Zero mean, unit variance normalization
- Required for distance-based models (LR, SVM, KNN)
- Fitted only on training data

### 4. Class Imbalance — SMOTE

SMOTE is applied **inside the training fold only** to avoid data leakage:

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sel, y_train)
```

This synthetically oversamples the minority class (Healthy) to balance the training distribution.

### 5. Model Training

**Baseline models** trained and logged to MLflow:

| Model | Notes |
|---|---|
| Logistic Regression | `max_iter=500`, trained on scaled + SMOTE data |
| Random Forest | `n_estimators=200`, trained on unscaled + SMOTE data |

**XGBoost hyperparameter tuning** via grid search over 12 combinations:

```python
depth_values    = [3, 5, 7]
learning_rates  = [0.05, 0.1]
estimators      = [100, 200]
```

Each combination is logged as a separate MLflow run. The best model across all runs is automatically selected by accuracy, registered in the MLflow Model Registry as `parkinson_detection_model`, and saved locally to `models/`.

---

## Model Results

Results from the notebook experimentation (with SMOTE + GridSearchCV):

| Model | Accuracy | Macro F1 | ROC AUC |
|---|---|---|---|
| **XGBoost (tuned)** | **0.89** | **0.855** | **0.946** |
| SVM | 0.87 | 0.833 | 0.920 |
| Random Forest | 0.87 | 0.828 | 0.931 |
| KNN | 0.83 | 0.804 | 0.947 |
| Logistic Regression | 0.82 | 0.776 | 0.859 |
| Decision Tree | 0.84 | 0.766 | 0.747 |

### Best XGBoost Configuration (GridSearchCV)

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

The system uses **SHAP TreeExplainer** for both global and local model interpretability.

### Global Feature Importance (`src/explain.py`)

- Samples 50 rows from the dataset for speed
- Computes mean absolute SHAP values across all samples
- Plots the top 20 most influential speech biomarkers
- Saves to `static/feature_importance.png`

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_selected)
importance = np.abs(shap_values).mean(axis=0)
```

### Local Prediction Explanation (`api/main.py`)

For every prediction, the API returns the top 10 feature contributions:

```python
shap_values = explainer.shap_values(arr_selected)[0]
top_indices = np.argsort(np.abs(shap_values))[-10:][::-1]
```

- Positive SHAP value → pushes prediction toward Parkinson's
- Negative SHAP value → pushes prediction toward Healthy
- Displayed as a color-coded horizontal bar chart in the UI

### Learning Curve (`src/learning_curve.py`)

- 5-fold cross-validation across increasing training set sizes
- Plots train vs. validation accuracy to diagnose bias/variance
- Saves to `static/learning_curve.png`

---

## MLOps Stack

### MLflow + DagsHub

All experiments are tracked remotely via DagsHub:

```python
dagshub.init(
    repo_owner="nishnarudkar",
    repo_name="Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers",
    mlflow=True
)
mlflow.set_experiment("parkinson_detection")
```

Each run logs:
- **Params:** model name, hyperparameters, num_features
- **Metrics:** accuracy, precision, recall, f1_score
- **Artifacts:** serialized model (sklearn flavor)

The best model is registered in the MLflow Model Registry:

```python
mlflow.register_model(f"runs:/{best_run_id}/model", "parkinson_detection_model")
```

### DVC Pipeline (`dvc.yaml`)

```yaml
stages:
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/pd_speech_features.csv
    outs:
      - models/model.pkl

  explain:
    cmd: python src/explain.py
    deps:
      - src/explain.py
      - models/model.pkl
    outs:
      - static/feature_importance.png
```

Run the full pipeline with:

```bash
dvc repro
```

DVC tracks inputs/outputs and only re-runs stages whose dependencies have changed.

### DVC Remote Storage

Dataset is versioned and stored on DagsHub:

```
https://dagshub.com/nishnarudkar/Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers.dvc
```

---

## Web Application

The FastAPI app serves a dark-themed, three-tab interactive UI.

### Tabs

**Feature Importance**
- Displays the global SHAP bar chart (`feature_importance.png`)
- Shows the top 20 most influential speech biomarkers

**Learning Curve**
- Displays the bias-variance learning curve (`learning_curve.png`)
- Helps visualize model generalization behavior

**Prediction**
- Accepts a comma-separated feature vector
- Returns prediction label, confidence probability bar, and a SHAP explanation chart
- Color-coded result: red for Parkinson's Detected, green for Healthy
- Interactive horizontal bar chart showing top 10 feature contributions

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

---

### `POST /predict`

Runs inference on a feature vector and returns a SHAP explanation.

**Request body:**

```json
{
  "features": [0.85247, 0.71826, 0.57227, 240, 239, 0.008064, ...]
}
```

The vector must contain the same number of features as the original dataset (753 values, excluding `id` and `class`).

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.923,
  "top_contributions": [
    { "feature_index": 42, "impact": 0.2341 },
    { "feature_index": 15, "impact": -0.1823 },
    ...
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | int | `1` = Parkinson's, `0` = Healthy |
| `probability` | float | Probability of Parkinson's (class 1) |
| `top_contributions` | list | Top 10 SHAP feature impacts |
| `feature_index` | int | Index of the selected feature |
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

**4. Pull the dataset via DVC**

```bash
dvc pull
```

---

## Running the Project

### Option A — Run each step manually

**Train the model:**

```bash
python src/train.py
```

This trains Logistic Regression, Random Forest, and XGBoost (12 hyperparameter combos), logs everything to MLflow/DagsHub, and saves the best model to `models/`.

**Generate SHAP feature importance:**

```bash
python src/explain.py
```

Saves `static/feature_importance.png`.

**Generate learning curve:**

```bash
python src/learning_curve.py
```

Saves `static/learning_curve.png`.

**Start the web app:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

### Option B — Run via DVC pipeline

```bash
dvc repro
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

DVC handles dependency tracking and only re-runs stages that are out of date.

---

### Option C — Open the notebook

The Colab notebook `notebooks/Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb` contains the full exploratory workflow including:

- EDA and class distribution analysis
- SMOTE-based oversampling
- Feature selection with `SelectFromModel` (Random Forest)
- Training and comparing 6 models: Logistic Regression, Random Forest, SVM, XGBoost, KNN, Decision Tree
- GridSearchCV tuning of XGBoost with a leakage-free `ImbPipeline`
- Confusion matrix and ROC curve visualization
- SHAP analysis

---

## Docker Deployment

**Build the image:**

```bash
docker build -t parkinson-ml .
```

**Run the container:**

```bash
docker run -p 8000:8000 parkinson-ml
```

The app will be available at `http://localhost:8000`.

The `Dockerfile` uses Python 3.10, copies the full project, installs dependencies, and starts Uvicorn on port 8000.

---

## CI/CD with Jenkins

The `Jenkinsfile` defines a three-stage pipeline:

```
Install Dependencies → Train Model → Build Docker Image
```

| Stage | Command |
|---|---|
| Install Dependencies | `pip install -r requirements.txt` |
| Train Model | `python src/train.py` |
| Build Docker Image | `docker build -t parkinson-ml .` |

Point a Jenkins job at this repository and configure it to use the `Jenkinsfile` for automated retraining and image builds on every push.

---

## Technology Stack

| Category | Tools |
|---|---|
| ML / Data | scikit-learn, XGBoost, pandas, numpy, imbalanced-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow, DagsHub |
| Data Versioning | DVC |
| Web Framework | FastAPI, Uvicorn, Jinja2 |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Containerization | Docker |
| CI/CD | Jenkins |
| Visualization | matplotlib, seaborn |
| Notebook | Google Colab |

---

## Disclaimer

This project is intended for **research and educational purposes only**. It is not a validated medical diagnostic tool. Do not use predictions from this system for clinical decision-making. Always consult a qualified healthcare professional for medical advice.
