import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    load_dataset,
    MODEL_PATH, SCALER_PATH, SELECTOR_PATH,
    FEATURE_IMPORTANCE_PNG, STATIC_DIR,
)

print("Loading model and preprocessing artifacts...")
model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
selector = joblib.load(SELECTOR_PATH)

print("Loading dataset...")
X, _ = load_dataset()

# Sample for speed
X_sample = X.sample(50, random_state=42)

print("Applying preprocessing (select → scale, matching train pipeline)...")

# Step 1: select from raw features → 100 selected features
X_selected = selector.transform(X_sample)
assert X_selected.shape[1] == scaler.n_features_in_, (
    f"Shape mismatch: selector output {X_selected.shape[1]} features, "
    f"but scaler expects {scaler.n_features_in_}. "
    "Ensure selector and scaler were saved from the same training run."
)

# Step 2: scale the selected features
X_scaled = scaler.transform(X_selected)

# Recover feature names for the selected columns
selected_features = X.columns[selector.get_support()]

# ── Choose the right SHAP explainer based on model type ──────────────────────
TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
               DecisionTreeClassifier, XGBClassifier)

if isinstance(model, TREE_MODELS):
    print("Creating SHAP TreeExplainer...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
else:
    print(f"Model is {type(model).__name__} — using KernelExplainer (slower)...")
    background  = shap.kmeans(X_scaled, 10)
    explainer   = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_scaled, nsamples=100)
# ─────────────────────────────────────────────────────────────────────────────

print("Calculating SHAP values...")

# Normalise to a single 2D array (n_samples, n_features)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

importance = np.abs(shap_values).mean(axis=0)

import pandas as pd
feature_importance = pd.DataFrame({
    "feature":    selected_features,
    "importance": importance,
}).sort_values("importance", ascending=False)

top_features = feature_importance.head(20)

print("Generating SHAP bar chart...")
plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP value|")
plt.title(f"Top 20 Speech Biomarkers for Parkinson Detection\n({type(model).__name__})")
plt.tight_layout()

os.makedirs(STATIC_DIR, exist_ok=True)
plt.savefig(FEATURE_IMPORTANCE_PNG, bbox_inches="tight")
plt.close()

print(f"Saved → {FEATURE_IMPORTANCE_PNG}")
