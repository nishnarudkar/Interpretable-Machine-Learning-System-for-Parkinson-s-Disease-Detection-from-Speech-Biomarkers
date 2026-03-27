import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import dagshub
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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


# --------------------------------
# Initialize DagsHub + MLflow
# --------------------------------
dagshub.init(
    repo_owner="nishnarudkar",
    repo_name="Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers",
    mlflow=True
)

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


# --------------------------------
# Scaling (after selection)
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
# Helper: log a run to MLflow
# --------------------------------
def log_run(run_name, model_name, params, X_tr, y_tr, X_te, use_scaled=False):
    X_test_eval = X_test_sel_scaled if use_scaled else X_test_sel
    with mlflow.start_run(run_name=run_name) as run:
        model_obj = model_name
        model_obj.fit(X_tr, y_tr)

        pred = model_obj.predict(X_te)
        prob = model_obj.predict_proba(X_te)[:, 1]

        acc       = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall    = recall_score(y_test, pred)
        f1        = f1_score(y_test, pred, average="macro")
        roc_auc   = roc_auc_score(y_test, prob)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("macro_f1",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)
        mlflow.sklearn.log_model(model_obj, artifact_path="model")

        print(f"{run_name}: acc={acc:.4f}  macro_f1={f1:.4f}  roc_auc={roc_auc:.4f}")
        print(classification_report(y_test, pred))

        return run.info.run_id, f1, model_obj


best_f1      = 0
best_model   = None
best_run_id  = None


# --------------------------------
# Logistic Regression (scaled + SMOTE)
# --------------------------------
run_id, f1, mdl = log_run(
    run_name="LogisticRegression",
    model_name=LogisticRegression(max_iter=1000, random_state=42),
    params={"model": "LogisticRegression", "max_iter": 1000, "num_features": 100},
    X_tr=X_train_scaled_smote, y_tr=y_train_smote,
    X_te=X_test_sel_scaled,    use_scaled=True
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# Random Forest (unscaled + SMOTE)
# --------------------------------
run_id, f1, mdl = log_run(
    run_name="RandomForest",
    model_name=RandomForestClassifier(n_estimators=300, random_state=42),
    params={"model": "RandomForest", "n_estimators": 300, "num_features": 100},
    X_tr=X_train_sel_smote, y_tr=y_train_smote,
    X_te=X_test_sel,        use_scaled=False
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# SVM (scaled + SMOTE)
# --------------------------------
run_id, f1, mdl = log_run(
    run_name="SVM",
    model_name=SVC(probability=True, random_state=42),
    params={"model": "SVM", "kernel": "rbf", "num_features": 100},
    X_tr=X_train_scaled_smote, y_tr=y_train_smote,
    X_te=X_test_sel_scaled,    use_scaled=True
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# KNN (scaled + SMOTE)
# --------------------------------
run_id, f1, mdl = log_run(
    run_name="KNN",
    model_name=KNeighborsClassifier(n_neighbors=5),
    params={"model": "KNN", "n_neighbors": 5, "num_features": 100},
    X_tr=X_train_scaled_smote, y_tr=y_train_smote,
    X_te=X_test_sel_scaled,    use_scaled=True
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# Decision Tree (unscaled + SMOTE)
# --------------------------------
run_id, f1, mdl = log_run(
    run_name="DecisionTree",
    model_name=DecisionTreeClassifier(random_state=42),
    params={"model": "DecisionTree", "num_features": 100},
    X_tr=X_train_sel_smote, y_tr=y_train_smote,
    X_te=X_test_sel,        use_scaled=False
)
if f1 > best_f1:
    best_f1, best_model, best_run_id = f1, mdl, run_id


# --------------------------------
# XGBoost — GridSearchCV with leakage-free ImbPipeline
# --------------------------------
print("\n--- Tuning XGBoost (GridSearchCV) ---")

xgb_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("xgb",   XGBClassifier(learning_rate=0.05, eval_metric="logloss", random_state=42))
])

param_grid = {
    "xgb__max_depth":         [3, 4, 5],
    "xgb__min_child_weight":  [1, 3, 5],
    "xgb__gamma":             [0, 0.1, 0.5],
    "xgb__subsample":         [0.8, 1.0],
    "xgb__colsample_bytree":  [0.8, 1.0],
    "xgb__n_estimators":      [200, 300],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_sel, y_train)

best_tuned_pipeline = grid_search.best_estimator_
best_tuned_xgb      = best_tuned_pipeline.named_steps["xgb"]

y_pred_tuned = best_tuned_pipeline.predict(X_test_sel)
y_prob_tuned = best_tuned_pipeline.predict_proba(X_test_sel)[:, 1]

acc_xgb     = accuracy_score(y_test, y_pred_tuned)
f1_xgb      = f1_score(y_test, y_pred_tuned, average="macro")
roc_auc_xgb = roc_auc_score(y_test, y_prob_tuned)

print(f"\nBest XGBoost params: {grid_search.best_params_}")
print(f"Tuned XGBoost: acc={acc_xgb:.4f}  macro_f1={f1_xgb:.4f}  roc_auc={roc_auc_xgb:.4f}")
print(classification_report(y_test, y_pred_tuned))

with mlflow.start_run(run_name="XGBoost_tuned") as run:
    best_params = {k.replace("xgb__", ""): v for k, v in grid_search.best_params_.items()}
    best_params["model"] = "XGBoost_tuned"
    best_params["num_features"] = 100
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy",  acc_xgb)
    mlflow.log_metric("precision", precision_score(y_test, y_pred_tuned))
    mlflow.log_metric("recall",    recall_score(y_test, y_pred_tuned))
    mlflow.log_metric("macro_f1",  f1_xgb)
    mlflow.log_metric("roc_auc",   roc_auc_xgb)
    mlflow.sklearn.log_model(best_tuned_xgb, artifact_path="model")
    xgb_run_id = run.info.run_id

    # Register inside the active run so the artifact URI is guaranteed to resolve
    if f1_xgb > best_f1:
        mlflow.register_model(
            model_uri=f"runs:/{xgb_run_id}/model",
            name="parkinson_detection_model"
        )

if f1_xgb > best_f1:
    best_f1    = f1_xgb
    best_model = best_tuned_xgb
    best_run_id = xgb_run_id


# --------------------------------
# Register best model (baseline winner so far)
# Done after all runs so we have the correct best_run_id
# --------------------------------
def try_register(run_id, name="parkinson_detection_model"):
    try:
        mlflow.register_model(f"runs:/{run_id}/model", name)
        print(f"Registered model from run {run_id}")
    except Exception as e:
        print(f"Model registration skipped: {e}")


# --------------------------------
# Save artifacts for API
# --------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
joblib.dump(selector,   "models/selector.pkl")

# Register the overall best (XGBoost already registered inside its run if it won)
# For baseline winners, register here
if best_run_id != xgb_run_id:
    try_register(best_run_id)

print(f"\nBest model saved — Macro F1: {best_f1:.4f}")
