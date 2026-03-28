import os
import json
import warnings
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.config` resolves from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    load_dataset, MODELS_DIR,
    MODEL_PATH, SCALER_PATH, SELECTOR_PATH, FEATURE_NAMES_PATH,
    ARTIFACTS_DIR, MODEL_METRICS_PATH, FEATURE_CONFIG_PATH,
)
from src.model_selection import apply_selection_flags

COLUMN_ORDER_PATH = MODELS_DIR / "column_order.pkl"

# Suppress MLflow pickle serialisation warning — expected for sklearn models
warnings.filterwarnings(
    "ignore",
    message="Saving scikit-learn models in the pickle",
    category=UserWarning,
)

# ── Credentials from environment variables (never hardcode secrets) ───────────
DAGSHUB_TRACKING_URI = (
    "https://dagshub.com/nishnarudkar/"
    "Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers.mlflow"
)
os.environ["MLFLOW_TRACKING_URI"]      = DAGSHUB_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "nishnarudkar")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
os.environ["DAGSHUB_USER_TOKEN"]       = os.getenv("DAGSHUB_TOKEN", "")
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import dagshub

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from scipy.stats import randint, uniform


# --------------------------------
# Initialize DagsHub + MLflow
# --------------------------------
dagshub.init(
    repo_owner="nishnarudkar",
    repo_name="Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers",
    mlflow=True
)

mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
mlflow.set_experiment("parkinson_detection")


# --------------------------------
# Load dataset
# --------------------------------
X, y = load_dataset()
print(f"Dataset shape: {X.shape}, Class distribution:\n{y.value_counts()}")


# --------------------------------
# Train/Test Split (stratified)
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# --------------------------------
# Feature Selection
# Use SMOTE first so the RF selector learns minority class features,
# then select on the original (non-SMOTE) train split to avoid leakage.
# --------------------------------
smote_fs = SMOTE(random_state=42)
X_train_fs, y_train_fs = smote_fs.fit_resample(X_train, y_train)

rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf_selector, max_features=100)
selector.fit(X_train_fs, y_train_fs)

X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)
print(f"Selected features: {X_train_sel.shape[1]}")

# Save selected feature names for the API
selected_feature_names = list(X.columns[selector.get_support()])


# --------------------------------
# Scaling (after selection — select → scale order)
# --------------------------------
scaler = StandardScaler()
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled  = scaler.transform(X_test_sel)


# --------------------------------
# Helper: compute all metrics (all macro-averaged for consistency)
# --------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall":    recall_score(y_true, y_pred, average="macro"),
        "macro_f1":  f1_score(y_true, y_pred, average="macro"),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


best_f1     = 0
best_model  = None
best_run_id = None


# --------------------------------
# Logistic Regression — RandomizedSearchCV inside ImbPipeline (scaled, no pre-SMOTE)
# --------------------------------
print("\n--- Tuning Logistic Regression ---")
lr_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("lr",    LogisticRegression(random_state=42, max_iter=5000)),
])
lr_param_dist = {
    "lr__C":      uniform(0.01, 10),
    "lr__solver": ["lbfgs", "saga"],
    # penalty removed — deprecated in sklearn 1.8; regularisation controlled by C alone
}
lr_search = RandomizedSearchCV(
    lr_pipeline, lr_param_dist, n_iter=20,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
lr_search.fit(X_train_sel_scaled, y_train)   # raw (unsmoted) train — SMOTE inside pipeline
best_lr_pipeline = lr_search.best_estimator_

y_pred_lr = best_lr_pipeline.predict(X_test_sel_scaled)
y_prob_lr = best_lr_pipeline.predict_proba(X_test_sel_scaled)[:, 1]
lr_metrics = compute_metrics(y_test, y_pred_lr, y_prob_lr)

with mlflow.start_run(run_name="LogisticRegression") as run:
    params = {k.replace("lr__", ""): v for k, v in lr_search.best_params_.items()}
    params.update({"model": "LogisticRegression", "num_features": 100})
    mlflow.log_params(params)
    mlflow.log_metrics(lr_metrics)
    mlflow.sklearn.log_model(best_lr_pipeline.named_steps["lr"], name="model")
    lr_run_id = run.info.run_id

print(f"\nLogisticRegression: {lr_metrics}")
print(classification_report(y_test, y_pred_lr))

if lr_metrics["macro_f1"] > best_f1:
    best_f1     = lr_metrics["macro_f1"]
    best_model  = best_lr_pipeline.named_steps["lr"]
    best_run_id = lr_run_id


# --------------------------------
# Random Forest — RandomizedSearchCV inside ImbPipeline (unscaled)
# --------------------------------
print("\n--- Tuning Random Forest ---")
rf_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf",    RandomForestClassifier(random_state=42)),
])
rf_param_dist = {
    "rf__n_estimators":      randint(100, 400),
    "rf__max_depth":         [None, 10, 20, 30],
    "rf__max_features":      ["sqrt", "log2"],
    "rf__min_samples_split": randint(2, 10),
}
rf_search = RandomizedSearchCV(
    rf_pipeline, rf_param_dist, n_iter=20,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
rf_search.fit(X_train_sel, y_train)
best_rf_pipeline = rf_search.best_estimator_

y_pred_rf = best_rf_pipeline.predict(X_test_sel)
y_prob_rf = best_rf_pipeline.predict_proba(X_test_sel)[:, 1]
rf_metrics = compute_metrics(y_test, y_pred_rf, y_prob_rf)

with mlflow.start_run(run_name="RandomForest") as run:
    params = {k.replace("rf__", ""): v for k, v in rf_search.best_params_.items()}
    params.update({"model": "RandomForest", "num_features": 100})
    mlflow.log_params(params)
    mlflow.log_metrics(rf_metrics)
    mlflow.sklearn.log_model(best_rf_pipeline.named_steps["rf"], name="model")
    rf_run_id = run.info.run_id

print(f"\nRandomForest: {rf_metrics}")
print(classification_report(y_test, y_pred_rf))

if rf_metrics["macro_f1"] > best_f1:
    best_f1     = rf_metrics["macro_f1"]
    best_model  = best_rf_pipeline.named_steps["rf"]
    best_run_id = rf_run_id


# --------------------------------
# SVM — RandomizedSearchCV inside ImbPipeline (scaled)
# --------------------------------
print("\n--- Tuning SVM ---")
svm_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("svm",   SVC(probability=True, random_state=42)),
])
svm_param_dist = {
    "svm__C":      uniform(0.1, 10),
    "svm__gamma":  ["scale", "auto"],
    "svm__kernel": ["rbf", "poly"],
}
svm_search = RandomizedSearchCV(
    svm_pipeline, svm_param_dist, n_iter=15,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
svm_search.fit(X_train_sel_scaled, y_train)
best_svm_pipeline = svm_search.best_estimator_

y_pred_svm = best_svm_pipeline.predict(X_test_sel_scaled)
y_prob_svm = best_svm_pipeline.predict_proba(X_test_sel_scaled)[:, 1]
svm_metrics = compute_metrics(y_test, y_pred_svm, y_prob_svm)

with mlflow.start_run(run_name="SVM") as run:
    params = {k.replace("svm__", ""): v for k, v in svm_search.best_params_.items()}
    params.update({"model": "SVM", "num_features": 100})
    mlflow.log_params(params)
    mlflow.log_metrics(svm_metrics)
    mlflow.sklearn.log_model(best_svm_pipeline.named_steps["svm"], name="model")
    svm_run_id = run.info.run_id

print(f"\nSVM: {svm_metrics}")
print(classification_report(y_test, y_pred_svm))

if svm_metrics["macro_f1"] > best_f1:
    best_f1     = svm_metrics["macro_f1"]
    best_model  = best_svm_pipeline.named_steps["svm"]
    best_run_id = svm_run_id


# --------------------------------
# KNN — RandomizedSearchCV inside ImbPipeline (scaled)
# --------------------------------
print("\n--- Tuning KNN ---")
knn_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("knn",   KNeighborsClassifier()),
])
knn_param_dist = {
    "knn__n_neighbors": randint(3, 20),
    "knn__weights":     ["uniform", "distance"],
    "knn__metric":      ["euclidean", "manhattan"],
}
knn_search = RandomizedSearchCV(
    knn_pipeline, knn_param_dist, n_iter=15,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
knn_search.fit(X_train_sel_scaled, y_train)
best_knn_pipeline = knn_search.best_estimator_

y_pred_knn = best_knn_pipeline.predict(X_test_sel_scaled)
y_prob_knn = best_knn_pipeline.predict_proba(X_test_sel_scaled)[:, 1]
knn_metrics = compute_metrics(y_test, y_pred_knn, y_prob_knn)

with mlflow.start_run(run_name="KNN") as run:
    params = {k.replace("knn__", ""): v for k, v in knn_search.best_params_.items()}
    params.update({"model": "KNN", "num_features": 100})
    mlflow.log_params(params)
    mlflow.log_metrics(knn_metrics)
    mlflow.sklearn.log_model(best_knn_pipeline.named_steps["knn"], name="model")
    knn_run_id = run.info.run_id

print(f"\nKNN: {knn_metrics}")
print(classification_report(y_test, y_pred_knn))

if knn_metrics["macro_f1"] > best_f1:
    best_f1     = knn_metrics["macro_f1"]
    best_model  = best_knn_pipeline.named_steps["knn"]
    best_run_id = knn_run_id


# --------------------------------
# Decision Tree — RandomizedSearchCV inside ImbPipeline (unscaled)
# --------------------------------
print("\n--- Tuning Decision Tree ---")
dt_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("dt",    DecisionTreeClassifier(random_state=42)),
])
dt_param_dist = {
    "dt__max_depth":         [None, 5, 10, 15, 20],
    "dt__min_samples_split": randint(2, 20),
    "dt__min_samples_leaf":  randint(1, 10),
    "dt__criterion":         ["gini", "entropy"],
}
dt_search = RandomizedSearchCV(
    dt_pipeline, dt_param_dist, n_iter=20,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
dt_search.fit(X_train_sel, y_train)
best_dt_pipeline = dt_search.best_estimator_

y_pred_dt = best_dt_pipeline.predict(X_test_sel)
y_prob_dt = best_dt_pipeline.predict_proba(X_test_sel)[:, 1]
dt_metrics = compute_metrics(y_test, y_pred_dt, y_prob_dt)

with mlflow.start_run(run_name="DecisionTree") as run:
    params = {k.replace("dt__", ""): v for k, v in dt_search.best_params_.items()}
    params.update({"model": "DecisionTree", "num_features": 100})
    mlflow.log_params(params)
    mlflow.log_metrics(dt_metrics)
    mlflow.sklearn.log_model(best_dt_pipeline.named_steps["dt"], name="model")
    dt_run_id = run.info.run_id

print(f"\nDecisionTree: {dt_metrics}")
print(classification_report(y_test, y_pred_dt))

if dt_metrics["macro_f1"] > best_f1:
    best_f1     = dt_metrics["macro_f1"]
    best_model  = best_dt_pipeline.named_steps["dt"]
    best_run_id = dt_run_id


# --------------------------------
# XGBoost — RandomizedSearchCV with leakage-free ImbPipeline
# (replaced GridSearchCV to reduce 1080 → ~216 fits)
# --------------------------------
print("\n--- Tuning XGBoost (RandomizedSearchCV) ---")

xgb_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("xgb",   XGBClassifier(learning_rate=0.05, eval_metric="logloss", random_state=42)),
])

xgb_param_dist = {
    "xgb__max_depth":        randint(3, 8),
    "xgb__min_child_weight": randint(1, 6),
    "xgb__gamma":            uniform(0, 0.5),
    "xgb__subsample":        uniform(0.7, 0.3),
    "xgb__colsample_bytree": uniform(0.7, 0.3),
    "xgb__n_estimators":     randint(100, 400),
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline,
    xgb_param_dist,
    n_iter=30,
    scoring="f1_macro",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

xgb_search.fit(X_train_sel, y_train)

best_tuned_pipeline = xgb_search.best_estimator_
best_tuned_xgb      = best_tuned_pipeline.named_steps["xgb"]

y_pred_tuned = best_tuned_pipeline.predict(X_test_sel)
y_prob_tuned = best_tuned_pipeline.predict_proba(X_test_sel)[:, 1]

xgb_metrics = compute_metrics(y_test, y_pred_tuned, y_prob_tuned)

print(f"\nBest XGBoost params: {xgb_search.best_params_}")
for k, v in xgb_metrics.items():
    print(f"  {k}: {v:.4f}")
print(classification_report(y_test, y_pred_tuned))

with mlflow.start_run(run_name="XGBoost_tuned") as run:
    best_params = {k.replace("xgb__", ""): v for k, v in xgb_search.best_params_.items()}
    best_params["model"] = "XGBoost_tuned"
    best_params["num_features"] = 100
    mlflow.log_params(best_params)
    mlflow.log_metrics(xgb_metrics)
    # registered_model_name registers the model in the MLflow Model Registry
    # in a single atomic call — the correct approach for MLflow 3.x
    mlflow.sklearn.log_model(
        best_tuned_xgb,
        name="model",
        registered_model_name="parkinson_detection_model",
    )
    xgb_run_id = run.info.run_id
    print(f"Logged and registered XGBoost as 'parkinson_detection_model' (run {xgb_run_id})")

if xgb_metrics["macro_f1"] > best_f1:
    best_f1     = xgb_metrics["macro_f1"]
    best_model  = best_tuned_xgb
    best_run_id = xgb_run_id


# --------------------------------
# Persist model comparison metrics (held-out test set)
# --------------------------------
def _metrics_row(display_name: str, metrics: dict) -> dict:
    return {
        "model":    display_name,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "roc_auc":  float(metrics["roc_auc"]),
        "selected": False,
    }


model_metrics_rows = [
    _metrics_row("Logistic Regression", lr_metrics),
    _metrics_row("Random Forest",       rf_metrics),
    _metrics_row("SVM",                 svm_metrics),
    _metrics_row("KNN",                 knn_metrics),
    _metrics_row("Decision Tree",       dt_metrics),
    _metrics_row("XGBoost",             xgb_metrics),
]

apply_selection_flags(model_metrics_rows)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(model_metrics_rows, f, indent=2)
print(f"\nSaved model comparison metrics: {MODEL_METRICS_PATH}")


# --------------------------------
# Save artifacts for API
# --------------------------------
# XGBoost is always used as the production model regardless of leaderboard rank.
# Rationale: this is a medical application where interpretability is critical.
# XGBoost supports fast, exact SHAP (TreeExplainer) which provides meaningful
# biomarker explanations. KNN/SVM may score marginally higher on some metrics
# but offer no native feature attribution — only slow KernelExplainer approximations.
# All models are still logged to MLflow for full comparison.
production_model = best_tuned_xgb
print(f"\nLeaderboard winner:  {type(best_model).__name__} (macro F1: {best_f1:.4f})")
print(f"Production model:    XGBoost_tuned (macro F1: {xgb_metrics['macro_f1']:.4f})")
print(f"Reason: XGBoost selected for interpretability (SHAP TreeExplainer) in medical context.")

# Top 5 features by production XGBoost feature_importances_; default_values = train-split means (raw)
imp = np.asarray(production_model.feature_importances_)
top5_idx = np.argsort(imp)[::-1][:5]
top_features = [selected_feature_names[i] for i in top5_idx]
feature_config = {
    "top_features": top_features,
    # Train-split means (raw) for every column — used by API to fill missing features
    "default_values": {col: float(X_train[col].mean()) for col in X.columns},
}
with open(FEATURE_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, indent=2)
print(f"Saved feature config: {FEATURE_CONFIG_PATH}")

os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(production_model,       MODEL_PATH)
joblib.dump(scaler,                 SCALER_PATH)
joblib.dump(selector,               SELECTOR_PATH)
joblib.dump(selected_feature_names, FEATURE_NAMES_PATH)
joblib.dump(list(X.columns),        COLUMN_ORDER_PATH)

# ── Generate feature_medians.json for the API prediction form ─────────────────
import json as _json
from src.config import STATIC_DIR as _STATIC_DIR

_STATIC_DIR.mkdir(parents=True, exist_ok=True)
_medians = X.median().to_dict()
_feature_medians = {
    "columns": list(X.columns),
    "medians": {col: float(_medians[col]) for col in X.columns},
}
_medians_path = _STATIC_DIR / "feature_medians.json"
with open(_medians_path, "w", encoding="utf-8") as _f:
    _json.dump(_feature_medians, _f, indent=2)
print(f"Saved feature medians: {_medians_path}")
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nProduction model saved: {type(production_model).__name__}")
print(f"Saved feature names: {len(selected_feature_names)} features")
print(f"Saved column order:  {len(list(X.columns))} columns")
