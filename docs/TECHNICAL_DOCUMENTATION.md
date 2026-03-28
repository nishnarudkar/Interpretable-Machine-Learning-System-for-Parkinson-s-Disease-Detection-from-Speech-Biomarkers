# Technical Documentation

## Interpretable ML System for Parkinson's Disease Detection

**Version:** 1.0.0 | **Python:** 3.10+ | **Last Updated:** March 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Reference](#2-module-reference)
   - [src/config.py](#21-srcconfigpy)
   - [src/train.py](#22-srctrainpy)
   - [src/explain.py](#23-srcexplainpy)
   - [src/learning_curve.py](#24-srclearning_curvepy)
   - [src/model_selection.py](#25-srcmodel_selectionpy)
   - [src/mlflow_comparison.py](#26-srcmlflow_comparisonpy)
   - [api/main.py](#27-apimainpy)
3. [Data Flow](#3-data-flow)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Model Training Details](#5-model-training-details)
6. [API Endpoints](#6-api-endpoints)
7. [Artifact Reference](#7-artifact-reference)
8. [DVC Pipeline](#8-dvc-pipeline)
9. [Docker Build](#9-docker-build)
10. [CI/CD Pipeline (Jenkins)](#10-cicd-pipeline-jenkins)
11. [Environment Variables](#11-environment-variables)
12. [Error Handling](#12-error-handling)

---

## 1. System Overview

The system is structured as a three-layer MLOps pipeline:

```
Layer 1 — Training     src/train.py → models/ + artifacts/
Layer 2 — Analysis     src/explain.py + src/learning_curve.py → static/
Layer 3 — Serving      api/main.py → HTTP endpoints
```

All layers share configuration through `src/config.py`, which resolves all file paths relative to the project root using `pathlib.Path`. This ensures scripts run correctly from any working directory — locally, inside Docker, or in CI.

---

## 2. Module Reference

### 2.1 `src/config.py`

Central configuration module. All other modules import paths and constants from here.

#### Constants

| Constant | Type | Value | Description |
|---|---|---|---|
| `ROOT` | `Path` | `Path(__file__).parent.parent` | Project root directory |
| `DATA_FILE` | `Path` | `ROOT/data/pd_speech_features.csv` | Dataset path |
| `CSV_HEADER_ROW` | `int` | `1` | Row index used as column names (UCI dataset has 2-row header) |
| `TARGET_COLUMN` | `str` | `"class"` | Binary target column name |
| `DROP_COLUMNS` | `list` | `["id"]` | Columns dropped before training |
| `EXPECTED_RAW_FEATURES` | `int` | `753` | Expected feature count after dropping id/class |
| `MLFLOW_TRACKING_URI` | `str` | DagsHub URI (env override) | MLflow remote tracking server |
| `MLFLOW_EXPERIMENT_NAME` | `str` | `"parkinson_detection"` | MLflow experiment name |
| `MODEL_PATH` | `Path` | `ROOT/models/model.pkl` | Production model artifact |
| `SCALER_PATH` | `Path` | `ROOT/models/scaler.pkl` | Fitted StandardScaler |
| `SELECTOR_PATH` | `Path` | `ROOT/models/selector.pkl` | Fitted SelectFromModel |
| `FEATURE_NAMES_PATH` | `Path` | `ROOT/models/feature_names.pkl` | 100 selected feature names |
| `MODEL_METRICS_PATH` | `Path` | `ROOT/artifacts/model_metrics.json` | Per-run comparison metrics |
| `FEATURE_CONFIG_PATH` | `Path` | `ROOT/artifacts/feature_config.json` | Top features + default values |
| `FEATURE_IMPORTANCE_PNG` | `Path` | `ROOT/static/feature_importance.png` | SHAP global chart |
| `LEARNING_CURVE_PNG` | `Path` | `ROOT/static/learning_curve.png` | Bias-variance plot |

#### Functions

**`load_dataset() → tuple[pd.DataFrame, pd.Series]`**

Loads and validates the Parkinson's speech dataset.

- Reads CSV with `header=1` (row 1 contains actual column names)
- Drops `id` column
- Validates `class` column exists (guards against wrong `header` value)
- Validates feature count equals `EXPECTED_RAW_FEATURES` (753)
- Raises `FileNotFoundError` if CSV missing (run `dvc pull`)
- Raises `ValueError` if column count or target column is wrong

---

### 2.2 `src/train.py`

Full training pipeline. Trains 6 models, logs to MLflow, saves production artifacts.

#### Execution Flow

```
1. Initialize DagsHub + MLflow
2. load_dataset() → X (753 features), y (binary)
3. train_test_split (80/20, stratified, random_state=42)
4. Feature selection (SMOTE-balanced RF selector, max_features=100)
5. Scaling (StandardScaler on selected features)
6. Train 6 models with RandomizedSearchCV + ImbPipeline
7. Log each run to MLflow
8. Persist model_metrics.json
9. Save production artifacts (XGBoost always selected)
10. Generate feature_medians.json
```

#### Feature Selection Detail

```python
smote_fs = SMOTE(random_state=42)
X_train_fs, y_train_fs = smote_fs.fit_resample(X_train, y_train)
# SMOTE applied to help RF selector learn minority class features

selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=100)
selector.fit(X_train_fs, y_train_fs)
# Selector fitted on balanced data, transforms original splits (no leakage)

X_train_sel = selector.transform(X_train)   # 604 × 100
X_test_sel  = selector.transform(X_test)    # 152 × 100
```

#### Scaling Order

```
select → scale   (NOT scale → select)
```

The scaler is fitted on `X_train_sel` (100 features), not on the full 753-feature space.

#### Model Training Pattern

All 6 models follow the same leakage-free pattern:

```python
pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),   # applied inside each CV fold
    ("model", SomeClassifier()),
])
search = RandomizedSearchCV(
    pipeline, param_dist,
    scoring="f1_macro",                  # primary metric
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42,
)
search.fit(X_train_sel, y_train)         # raw (unsmoted) input
```

#### `compute_metrics(y_true, y_pred, y_prob) → dict`

Returns all metrics macro-averaged:

```python
{
    "accuracy":  float,   # overall accuracy
    "precision": float,   # macro-averaged precision
    "recall":    float,   # macro-averaged recall
    "macro_f1":  float,   # macro-averaged F1 (primary selection metric)
    "roc_auc":   float,   # ROC AUC score
}
```

#### Production Model Selection

XGBoost is **always** saved as the production model regardless of leaderboard rank:

```python
production_model = best_tuned_xgb
```

Rationale: SHAP `TreeExplainer` provides fast, exact feature attribution for tree models. KNN/SVM require slow `KernelExplainer` approximations — unacceptable for a medical application requiring clinical transparency.

#### Outputs Written

| File | Description |
|---|---|
| `models/model.pkl` | Production XGBoost model |
| `models/scaler.pkl` | Fitted StandardScaler (100 features) |
| `models/selector.pkl` | Fitted SelectFromModel |
| `models/feature_names.pkl` | List of 100 selected feature names |
| `models/column_order.pkl` | List of 753 column names in training order |
| `artifacts/model_metrics.json` | All 6 model metrics with selection flags |
| `artifacts/feature_config.json` | Top 5 features + all 753 column means |
| `static/feature_medians.json` | All 753 column medians for prediction form |

---

### 2.3 `src/explain.py`

Generates the global SHAP feature importance chart.

#### Execution Flow

```
1. Load model, scaler, selector from models/
2. load_dataset() → X
3. Sample 50 rows (for speed)
4. Preprocess: selector.transform → scaler.transform
5. Choose explainer (TreeExplainer for tree models, KernelExplainer otherwise)
6. Compute SHAP values → extract_shap_for_class1()
7. Compute mean |SHAP| per feature
8. Plot top 20 features → static/feature_importance.png
```

#### `extract_shap_for_class1(raw) → np.ndarray`

Normalises SHAP output to shape `(n_samples, n_features)` for class 1, handling all known output formats:

| Input shape | Source | Handling |
|---|---|---|
| `list` of arrays | sklearn RF, old SHAP | `raw[1]` → class-1 |
| `(n, f, c)` ndarray | some XGBoost configs | `[:, :, 1]` |
| `(n, f)` ndarray | XGBoost binary, new SHAP | use as-is |
| `(f,)` ndarray | single-sample shortcut | `newaxis` → `(1, f)` |

#### Preprocessing Assertion

```python
assert X_selected.shape[1] == scaler.n_features_in_
```

Fails immediately with a clear message if selector and scaler were saved from different training runs.

---

### 2.4 `src/learning_curve.py`

Generates the bias-variance learning curve with no data leakage.

#### Key Design Decision

The full preprocessing pipeline is wrapped in `ImbPipeline` and passed to `sklearn.learning_curve`. This means SMOTE, feature selection, and scaling are **all refitted inside each CV fold** — the validation fold never influences preprocessing.

```python
pipeline = ImbPipeline([
    ("smote",    SMOTE(random_state=42)),
    ("selector", SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=100)),
    ("scaler",   StandardScaler()),
    ("model",    model),
])
```

#### Output

- `static/learning_curve.png` — train vs. validation macro F1 with ±1 std bands
- Gap annotation at the largest training size
- Console warning if gap > 0.15 (potential overfitting)

---

### 2.5 `src/model_selection.py`

Applies selection flags to model comparison rows.

#### `apply_selection_flags(rows: list) → None`

Mutates rows in-place. Sets `selected: True` on the best model using this priority:

1. Best **interpretable** model by composite score: `0.6 × roc_auc + 0.4 × macro_f1`
2. If no interpretable model exists, fall back to overall best `macro_f1`

**Interpretable models:** `XGBoost`, `XGBoost_tuned`, `Random Forest`, `RandomForest`, `Decision Tree`

These are preferred because SHAP `TreeExplainer` works natively with them.

---

### 2.6 `src/mlflow_comparison.py`

#### `fetch_model_comparison_from_mlflow() → list[dict]`

Returns model comparison data using a two-strategy approach:

**Strategy 1 (fast):** Read `artifacts/model_metrics.json` written by `train.py`. No network call.

**Strategy 2 (fallback):** Query MLflow tracking server directly. Keeps the most recent run per model name. Requires `macro_f1` and `roc_auc` metrics to be present.

Returns list of dicts: `{ model, accuracy, macro_f1, roc_auc, selected }`

---

### 2.7 `api/main.py`

FastAPI application. Loads all artifacts at startup and serves 6 endpoints.

#### Startup Validation

Three assertions run at import time:

```python
assert len(column_order) == EXPECTED_RAW_FEATURES      # 753
assert selector.n_features_in_ == EXPECTED_RAW_FEATURES # 753
assert scaler.n_features_in_ == len(feature_names)      # 100
```

If any fails, the app raises `RuntimeError` before accepting requests.

#### `build_explainer(model, selector, scaler) → shap.Explainer`

- Tree models → `shap.TreeExplainer(model)` (fast, exact)
- Other models → `shap.KernelExplainer` with k-means background (10 clusters from 50 samples)
- Fallback → zero-vector background if sampling fails

#### `extract_shap_for_class1(raw) → np.ndarray`

Same logic as `src/explain.py` but returns 1-D array (single prediction).

#### `FeatureInput` (Pydantic model)

```python
features: Annotated[List[float], Field(min_length=753, max_length=753)]
```

Enforced at schema level — wrong length returns `422 Unprocessable Entity` before any ML code runs.

---

## 3. Data Flow

```
pd_speech_features.csv (756 × 755)
    │
    ▼ drop "id"
(756 × 754)
    │
    ▼ drop "class"
X: (756 × 753)    y: (756,)
    │
    ▼ train_test_split(stratify=y, test_size=0.2)
X_train: (604 × 753)    X_test: (152 × 753)
    │
    ▼ SMOTE on X_train (for selector only)
X_train_fs: (~1128 × 753)
    │
    ▼ SelectFromModel(RF, max_features=100).fit(X_train_fs)
    │   .transform(X_train) → X_train_sel: (604 × 100)
    │   .transform(X_test)  → X_test_sel:  (152 × 100)
    │
    ▼ StandardScaler.fit(X_train_sel)
    │   .transform(X_train_sel) → X_train_sel_scaled: (604 × 100)
    │   .transform(X_test_sel)  → X_test_sel_scaled:  (152 × 100)
    │
    ▼ ImbPipeline([SMOTE, model]).fit(X_train_sel, y_train)
      (SMOTE applied inside each CV fold — no leakage)
```

---

## 4. Preprocessing Pipeline

### Training vs. Serving

| Step | Training (`train.py`) | Serving (`api/main.py`) |
|---|---|---|
| Input | 753 raw features | 753 raw features (column-aligned) |
| Step 1 | `selector.transform(X_train)` | `selector.transform(arr)` |
| Step 2 | `scaler.transform(X_train_sel)` | `scaler.transform(arr_selected)` |
| Step 3 | `model.predict(X_train_sel_scaled)` | `model.predict(arr_scaled)` |

The serving pipeline exactly mirrors training. Column alignment is enforced by:

```python
arr = pd.DataFrame([data.features], columns=column_order).values
```

This prevents silent wrong-feature selection if input columns arrive in a different order.

---

## 5. Model Training Details

### Hyperparameter Search Spaces

**Logistic Regression** (scaled input, 20 iterations)
```
C:      Uniform(0.01, 10.01)
solver: ["lbfgs", "saga"]
```

**Random Forest** (unscaled input, 20 iterations)
```
n_estimators:      randint(100, 400)
max_depth:         [None, 10, 20, 30]
max_features:      ["sqrt", "log2"]
min_samples_split: randint(2, 10)
```

**SVM** (scaled input, 15 iterations)
```
C:      Uniform(0.1, 10.1)
gamma:  ["scale", "auto"]
kernel: ["rbf", "poly"]
```

**KNN** (scaled input, 15 iterations)
```
n_neighbors: randint(3, 20)
weights:     ["uniform", "distance"]
metric:      ["euclidean", "manhattan"]
```

**Decision Tree** (unscaled input, 20 iterations)
```
max_depth:         [None, 5, 10, 15, 20]
min_samples_split: randint(2, 20)
min_samples_leaf:  randint(1, 10)
criterion:         ["gini", "entropy"]
```

**XGBoost** (unscaled input, 30 iterations)
```
max_depth:        randint(3, 8)
min_child_weight: randint(1, 6)
gamma:            Uniform(0, 0.5)
subsample:        Uniform(0.7, 1.0)
colsample_bytree: Uniform(0.7, 1.0)
n_estimators:     randint(100, 400)
learning_rate:    0.05 (fixed)
```

### MLflow Logging

Each run logs:
- **Params:** model name, hyperparameters, `num_features=100`
- **Metrics:** `accuracy`, `precision`, `recall`, `macro_f1`, `roc_auc`
- **Artifact:** serialised model (sklearn flavor, `name="model"`)

XGBoost additionally registers in the Model Registry:
```python
mlflow.sklearn.log_model(
    best_tuned_xgb,
    name="model",
    registered_model_name="parkinson_detection_model",
)
```

---

## 6. API Endpoints

### `GET /`
Returns the main HTML interface (Jinja2 template).

---

### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model": "XGBClassifier",
  "explainer": "TreeExplainer",
  "model_loaded": true
}
```

---

### `POST /predict`

**Request body:**
```json
{ "features": [753 floats in training column order] }
```

Pydantic enforces exactly 753 values. Wrong count → `422 Unprocessable Entity`.

**Response:**
```json
{
  "prediction": 1,
  "label": "Parkinson's Detected",
  "probability": 0.923,
  "top_contributions": [
    {
      "feature_index": 42,
      "feature_name": "tqwt_kurtosisValue_dec_5",
      "impact": 0.2341
    }
  ],
  "shap_bar_url": "/static/shap_bar.png"
}
```

Side effect: generates `static/shap_bar.png` (dark-themed horizontal bar chart).

---

### `GET /feature-defaults`

Returns top 5 SHAP-ranked features with dataset statistics for the prediction form.

**Response:**
```json
{
  "top5": [
    {
      "name": "maxIntensity",
      "label": "Max Intensity (maxIntensity)",
      "tooltip": "Maximum vocal intensity...",
      "median": 78.5814,
      "min": 44.1335,
      "max": 86.3162
    }
  ],
  "columns": ["gender", "PPE", ...],
  "medians": { "gender": 1.0, "PPE": 0.5234, ... }
}
```

Top 5 computed by SHAP importance (100-sample background). Falls back to `feature_names[:5]` if SHAP fails.

---

### `GET /model-comparison`

Returns model leaderboard from `artifacts/model_metrics.json` (or MLflow fallback).

**Response:**
```json
{
  "models": [
    { "model": "XGBoost", "accuracy": 0.89, "macro_f1": 0.855, "roc_auc": 0.946, "selected": true },
    { "model": "SVM",     "accuracy": 0.87, "macro_f1": 0.833, "roc_auc": 0.920, "selected": false }
  ]
}
```

---

### `GET /top-features`

Returns top 5 globally important features by mean absolute SHAP value.

**Response:**
```json
{
  "top_features": [
    { "rank": 1, "name": "tqwt_kurtosisValue_dec_5", "importance": 0.1823 }
  ]
}
```

---

## 7. Artifact Reference

### `models/` directory

| File | Created by | Used by | Description |
|---|---|---|---|
| `model.pkl` | `train.py` | `api/main.py`, `explain.py`, `learning_curve.py` | Production XGBoost model |
| `scaler.pkl` | `train.py` | `api/main.py`, `explain.py` | StandardScaler fitted on 100 selected features |
| `selector.pkl` | `train.py` | `api/main.py`, `explain.py` | SelectFromModel fitted on SMOTE-balanced data |
| `feature_names.pkl` | `train.py` | `api/main.py`, `explain.py` | List of 100 selected feature names |
| `column_order.pkl` | `train.py` | `api/main.py` | List of 753 column names in training order |

### `artifacts/` directory

| File | Created by | Used by | Description |
|---|---|---|---|
| `model_metrics.json` | `train.py` | `api/main.py` → `/model-comparison` | All 6 model metrics with selection flags |
| `feature_config.json` | `train.py` | (legacy) | Top 5 features + all column means |

### `static/` directory

| File | Created by | Used by | Description |
|---|---|---|---|
| `feature_importance.png` | `explain.py` | UI Feature Importance tab | Global SHAP bar chart |
| `learning_curve.png` | `learning_curve.py` | UI Learning Curve tab | Bias-variance plot |
| `shap_bar.png` | `api/main.py` `/predict` | UI Prediction tab | Per-prediction SHAP chart |
| `feature_medians.json` | `train.py` | `api/main.py` → `/feature-defaults` | All 753 column medians |

---

## 8. DVC Pipeline

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

Run with `dvc repro`. DVC tracks file hashes and only re-runs stages whose dependencies have changed.

---

## 9. Docker Build

Multi-stage build. Stage 1 installs dependencies; Stage 2 copies only what the API needs.

**Runtime image contains:**
- `api/main.py`
- `src/config.py`, `src/__init__.py`, `src/mlflow_comparison.py`, `src/model_selection.py`
- `models/` (all 5 pkl files)
- `static/` (charts + feature_medians.json)
- `templates/index.html`

**Not included:** `src/train.py`, `src/explain.py`, `src/learning_curve.py`, `data/`, `notebooks/`, `mlruns/`

```bash
docker build -t parkinson-api .
docker run -p 8000:8000 parkinson-api
```

---

## 10. CI/CD Pipeline (Jenkins)

Six-stage pipeline. Uses Windows `bat` commands (configured for Windows agents).

| Stage | Command | Purpose |
|---|---|---|
| Install Dependencies | `pip install -r requirements.txt` | Install all packages |
| Pull Data (DVC) | `dvc pull data/pd_speech_features.csv.dvc` | Fetch dataset from DagsHub |
| Train Model | `python src/train.py` | Full training run |
| Generate Explanations | `python src/explain.py && python src/learning_curve.py` | Generate charts |
| Smoke Test | `curl -f http://localhost:8000/health` | Verify API starts |
| Build Docker Image | `docker build -t parkinson-api .` | Create container |

Credentials injected via Jenkins credentials store (`dagshub-username`, `dagshub-token`). Never hardcoded.

---

## 11. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DAGSHUB_USERNAME` | Yes (training) | `"nishnarudkar"` | DagsHub account username |
| `DAGSHUB_TOKEN` | Yes (training) | `""` | DagsHub personal access token |
| `MLFLOW_TRACKING_URI` | No | DagsHub URI | Override MLflow tracking server |
| `MLFLOW_EXPERIMENT_NAME` | No | `"parkinson_detection"` | MLflow experiment name |

Set via `.env` file (gitignored). Copy `.env.example` to get started.

---

## 12. Error Handling

### API startup errors

| Condition | Error | Resolution |
|---|---|---|
| `models/*.pkl` missing | `RuntimeError: Model artifact not found` | Run `python src/train.py` |
| Artifact shape mismatch | `RuntimeError: Artifact consistency check failed` | Re-run `python src/train.py` (artifacts from different runs) |

### API request errors

| Condition | HTTP Status | Detail |
|---|---|---|
| Wrong feature count | `422` | `"List should have at most/at least 753 items"` |
| Non-numeric values | `422` | Pydantic type validation error |
| ML pipeline failure | `500` | `"Prediction failed: <error message>"` |
| `/feature-defaults` missing JSON | `404` | `"Run feature_medians generation first"` |
| `/model-comparison` no data | `404` | `"No model runs found in MLflow"` |
| MLflow unreachable | `503` | `"Could not load metrics from MLflow"` |

### Training errors

| Condition | Error | Resolution |
|---|---|---|
| Dataset missing | `FileNotFoundError` | Run `dvc pull` |
| Wrong CSV structure | `ValueError: Column 'class' not found` | Check `CSV_HEADER_ROW` in `config.py` |
| Wrong feature count | `ValueError: Expected 753 features` | Check `DROP_COLUMNS` in `config.py` |
| DagsHub auth failure | MLflow connection error | Set `DAGSHUB_TOKEN` env var |
