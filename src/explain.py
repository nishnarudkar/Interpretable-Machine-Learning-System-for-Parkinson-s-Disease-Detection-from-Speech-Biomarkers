import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    load_dataset,
    MODEL_PATH, SCALER_PATH, SELECTOR_PATH,
    FEATURE_IMPORTANCE_PNG, STATIC_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
               DecisionTreeClassifier, XGBClassifier)


def extract_shap_for_class1(raw: object) -> np.ndarray:
    """
    Robustly extract a 2-D SHAP array (n_samples, n_features) for class 1,
    regardless of SHAP version or model type output format.
    """
    if isinstance(raw, list):
        arr = np.array(raw[1])
    else:
        arr = np.array(raw)

    if arr.ndim == 3:
        arr = arr[:, :, 1]   # (n_samples, n_features, n_classes) → class-1

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # single sample → (1, n_features)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape after normalisation: {arr.shape}")

    return arr   # (n_samples, n_features)


print("Loading model and preprocessing artifacts...")
model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
selector = joblib.load(SELECTOR_PATH)

print("Loading dataset...")
X, _ = load_dataset()

X_sample = X.sample(50, random_state=42)

print("Applying preprocessing (select → scale)...")
X_selected = selector.transform(X_sample)
assert X_selected.shape[1] == scaler.n_features_in_, (
    f"Shape mismatch: selector output {X_selected.shape[1]} features, "
    f"but scaler expects {scaler.n_features_in_}. "
    "Ensure selector and scaler were saved from the same training run."
)
X_scaled = scaler.transform(X_selected)

selected_features = X.columns[selector.get_support()]

# ── Choose explainer ──────────────────────────────────────────────────────────
if isinstance(model, TREE_MODELS):
    print("Creating SHAP TreeExplainer...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
else:
    print(f"Model is {type(model).__name__} — using KernelExplainer (slower)...")
    n_bg       = min(50, len(X_scaled))
    n_clusters = min(10, n_bg)
    try:
        background = shap.kmeans(X_scaled, n_clusters)
    except Exception as e:
        logger.warning(f"kmeans background failed ({e}), using zero vector")
        background = np.zeros((1, X_scaled.shape[1]))
    explainer   = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_scaled, nsamples=100)
# ─────────────────────────────────────────────────────────────────────────────

print("Calculating SHAP values...")
shap_2d    = extract_shap_for_class1(shap_values)   # (n_samples, n_features)
importance = np.abs(shap_2d).mean(axis=0)           # (n_features,)

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
