import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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

# Step 1: feature selection on raw data (matches train.py order)
X_selected = selector.transform(X_sample)

# Step 2: scale the selected features
X_scaled = scaler.transform(X_selected)

# Recover feature names for the selected columns
selected_features = X.columns[selector.get_support()]

print("Creating SHAP TreeExplainer...")
explainer  = shap.TreeExplainer(model)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_scaled)

# Normalise to a single 2D array (n_samples, n_features)
# - RandomForest / list output  → take class-1 slice
# - XGBoost 3D (n,f,c)          → take class-1 slice along last axis
# - Already 2D                  → use as-is
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# Now guaranteed 2D → mean absolute SHAP per feature
importance = np.abs(shap_values).mean(axis=0)  # shape: (n_features,)

feature_importance = pd.DataFrame({
    "feature":    selected_features,
    "importance": importance
}).sort_values("importance", ascending=False)

top_features = feature_importance.head(20)

print("Generating SHAP bar chart...")

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP value|")
plt.title("Top 20 Speech Biomarkers for Parkinson Detection")
plt.tight_layout()

os.makedirs("static", exist_ok=True)
plt.savefig("static/feature_importance.png", bbox_inches="tight")
plt.close()

print("Saved → static/feature_importance.png")
