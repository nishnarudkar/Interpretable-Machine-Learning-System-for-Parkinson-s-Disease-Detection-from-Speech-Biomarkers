import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

print("Loading model and preprocessing artifacts...")

model    = joblib.load("models/model.pkl")
scaler   = joblib.load("models/scaler.pkl")
selector = joblib.load("models/selector.pkl")

print("Loading dataset...")

df = pd.read_csv("data/pd_speech_features.csv", header=1)
X  = df.drop(["id", "class"], axis=1)

# Sample for speed
X_sample = X.sample(50, random_state=42)

print("Applying preprocessing (select → scale, matching train pipeline)...")

X_selected = selector.transform(X_sample)
X_scaled   = scaler.transform(X_selected)

# Recover feature names for the selected columns
selected_features = X.columns[selector.get_support()]

# ── Choose the right SHAP explainer based on model type ──────────────────────
TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
               DecisionTreeClassifier, XGBClassifier)

if isinstance(model, TREE_MODELS):
    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
else:
    print(f"Model is {type(model).__name__} — using KernelExplainer (slower)...")
    # KernelExplainer needs a background dataset; use the mean of training data
    background = shap.kmeans(X_scaled, 10)
    explainer  = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_scaled, nsamples=100)
# ─────────────────────────────────────────────────────────────────────────────

print("Calculating SHAP values...")

# Normalise to a single 2D array (n_samples, n_features)
if isinstance(shap_values, list):
    shap_values = shap_values[1]          # list → class-1 slice
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]    # 3D → class-1 slice

importance = np.abs(shap_values).mean(axis=0)

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

os.makedirs("static", exist_ok=True)
plt.savefig("static/feature_importance.png", bbox_inches="tight")
plt.close()

print("Saved → static/feature_importance.png")
