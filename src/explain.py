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

# For binary classifiers that return a list (e.g. RandomForest), take class-1 values
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Mean absolute SHAP across samples
importance = np.abs(shap_values).mean(axis=0)

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
