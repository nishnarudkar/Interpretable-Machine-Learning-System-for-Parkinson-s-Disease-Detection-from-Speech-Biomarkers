import os

# ── Credentials from environment variables (never hardcode secrets) ───────────
DAGSHUB_TRACKING_URI = (
    "https://dagshub.com/nishnarudkar/"
    "Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers.mlflow"
)
os.environ["MLFLOW_TRACKING_URI"]      = DAGSHUB_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "nishnarudkar")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
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
df = pd.read_csv("data/pd_speech_features.csv", header=1)
df = df.drop("id", axis=1)

X = df.drop("class", axis=1)
y = df["class"]

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
# SMOTE on selected features for training
# --------------------------------
smote = SMOTE(random_state=42)
X_train_sel_smote,    y_train_smote = smote.fit_resample(X_train_sel,        y_train)
X_train_scaled_smote, _             = smote.fit_resample(X_train_sel_scaled, y_train)


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


# --------------------------------
# Helper: log a run to MLflow
# --------------------------------
def log_run(run_name, model_obj, params, X_tr, y_tr, X_te, y_te):
    with mlflow.start_run(run_name=run_name) as run:
        model_obj.fit(X_tr, y_tr)

        y_pred = model_obj.predict(X_te)
        y_prob = model_obj.predict_proba(X_te)[:, 1]

        metrics = compute_metrics(y_te, y_pred, y_prob)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model_obj, artifact_path="model")

        print(f"\n{run_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(classification_report(y_te, y_pred))

        return run.info.run_id, metrics["macro_f1"], model_obj


best_f1     = 0
best_model  = None
best_run_id = None


# --------------------------------
# Logistic Regression — RandomizedSearchCV (scaled + SMOTE)
# --------------------------------
print("\n--- Tuning Logistic Regression ---")
lr_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("lr",    LogisticRegression(random_state=42, max_iter=2000)),
])
lr_param_dist = {
    "lr__C":      uniform(0.01, 10),
    "lr__solver": ["lbfgs", "saga"],
    "lr__penalty":["l2"],
}
lr_search = RandomizedSearchCV(
    lr_pipeline, lr_param_dist, n_iter=20,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
lr_search.fit(X_train_sel_scaled, y_train)
best_lr = lr_search.best_estimator_.named_steps["lr"]

run_id, f1, mdl = log_run(
    run_name="LogisticRegression",
    model_obj=best_lr,
    params={**{k.replace("lr__", ""): v for k, v in lr_search.best_params_.items()},
            "model": "LogisticRegression", "num_features": 100},
    X_tr=X_train_scaled_smote, y_tr=y_train_smote,
    X_te=X_test_sel_scaled,    y_te=y_test,
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# Random Forest — RandomizedSearchCV (unscaled + SMOTE)
# --------------------------------
print("\n--- Tuning Random Forest ---")
rf_param_dist = {
    "n_estimators": randint(100, 400),
    "max_depth":    [None, 10, 20, 30],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": randint(2, 10),
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_dist, n_iter=20,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
rf_search.fit(X_train_sel_smote, y_train_smote)
best_rf = rf_search.best_estimator_

run_id, f1, mdl = log_run(
    run_name="RandomForest",
    model_obj=best_rf,
    params={**rf_search.best_params_, "model": "RandomForest", "num_features": 100},
    X_tr=X_train_sel_smote, y_tr=y_train_smote,
    X_te=X_test_sel,        y_te=y_test,
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# SVM — RandomizedSearchCV (scaled + SMOTE)
# --------------------------------
print("\n--- Tuning SVM ---")
svm_param_dist = {
    "C":     uniform(0.1, 10),
    "gamma": ["scale", "auto"],
    "kernel":["rbf", "poly"],
}
svm_search = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_dist, n_iter=15,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
svm_search.fit(X_train_scaled_smote, y_train_smote)
best_svm = svm_search.best_estimator_

run_id, f1, mdl = log_run(
    run_name="SVM",
    model_obj=best_svm,
    params={**svm_search.best_params_, "model": "SVM", "num_features": 100},
    X_tr=X_train_scaled_smote, y_tr=y_train_smote,
    X_te=X_test_sel_scaled,    y_te=y_test,
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# KNN — RandomizedSearchCV (scaled + SMOTE)
# --------------------------------
print("\n--- Tuning KNN ---")
knn_param_dist = {
    "n_neighbors": randint(3, 20),
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}
knn_search = RandomizedSearchCV(
    KNeighborsClassifier(),
    knn_param_dist, n_iter=15,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
knn_search.fit(X_train_scaled_smote, y_train_smote)
best_knn = knn_search.best_estimator_

run_id, f1, mdl = log_run(
    run_name="KNN",
    model_obj=best_knn,
    params={**knn_search.best_params_, "model": "KNN", "num_features": 100},
    X_tr=X_train_scaled_smote, y_tr=y_train_smote,
    X_te=X_test_sel_scaled,    y_te=y_test,
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# Decision Tree — RandomizedSearchCV (unscaled + SMOTE)
# --------------------------------
print("\n--- Tuning Decision Tree ---")
dt_param_dist = {
    "max_depth":        [None, 5, 10, 15, 20],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf":  randint(1, 10),
    "criterion":        ["gini", "entropy"],
}
dt_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_dist, n_iter=20,
    scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42),
    n_jobs=-1, random_state=42, verbose=0
)
dt_search.fit(X_train_sel_smote, y_train_smote)
best_dt = dt_search.best_estimator_

run_id, f1, mdl = log_run(
    run_name="DecisionTree",
    model_obj=best_dt,
    params={**dt_search.best_params_, "model": "DecisionTree", "num_features": 100},
    X_tr=X_train_sel_smote, y_tr=y_train_smote,
    X_te=X_test_sel,        y_te=y_test,
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


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
    mlflow.sklearn.log_model(best_tuned_xgb, artifact_path="model")
    xgb_run_id = run.info.run_id

    if xgb_metrics["macro_f1"] > best_f1:
        mlflow.register_model(
            model_uri=f"runs:/{xgb_run_id}/model",
            name="parkinson_detection_model",
        )

if xgb_metrics["macro_f1"] > best_f1:
    best_f1     = xgb_metrics["macro_f1"]
    best_model  = best_tuned_xgb
    best_run_id = xgb_run_id


# --------------------------------
# Register best baseline model if XGBoost didn't win
# --------------------------------
def try_register(run_id, name="parkinson_detection_model"):
    try:
        mlflow.register_model(f"runs:/{run_id}/model", name)
        print(f"Registered model from run {run_id}")
    except Exception as e:
        print(f"Model registration skipped: {e}")

if best_run_id != xgb_run_id:
    try_register(best_run_id)


# --------------------------------
# Save artifacts for API
# --------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model,             "models/model.pkl")
joblib.dump(scaler,                 "models/scaler.pkl")
joblib.dump(selector,               "models/selector.pkl")
joblib.dump(selected_feature_names, "models/feature_names.pkl")

print(f"\nBest model saved — Macro F1: {best_f1:.4f}")
print(f"Saved feature names: {len(selected_feature_names)} features")
