from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from typing import List
import joblib
import numpy as np
import shap
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

app = FastAPI(title="Parkinson Detection API", version="1.0.0")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
               DecisionTreeClassifier, XGBClassifier)

# ── Load artifacts at startup ─────────────────────────────────────────────────
try:
    model          = joblib.load("models/model.pkl")
    scaler         = joblib.load("models/scaler.pkl")
    selector       = joblib.load("models/selector.pkl")
    feature_names  = joblib.load("models/feature_names.pkl")
    EXPECTED_FEATURES = 753

    if isinstance(model, TREE_MODELS):
        explainer = shap.TreeExplainer(model)
    else:
        # KernelExplainer: needs a small background dataset
        # Load a sample of the training data for the background
        import pandas as pd
        df = pd.read_csv("data/pd_speech_features.csv", header=1)
        X_bg = df.drop(["id", "class"], axis=1).sample(50, random_state=42)
        X_bg_sel    = selector.transform(X_bg)
        X_bg_scaled = scaler.transform(X_bg_sel)
        background  = shap.kmeans(X_bg_scaled, 10)
        explainer   = shap.KernelExplainer(model.predict_proba, background)

    print(f"Loaded model: {type(model).__name__}, explainer: {type(explainer).__name__}")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artifact not found: {e}. Run `python src/train.py` first."
    )
# ─────────────────────────────────────────────────────────────────────────────


class FeatureInput(BaseModel):
    features: List[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features, got {len(v)}"
            )
        return v


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(data: FeatureInput):
    arr = np.array(data.features).reshape(1, -1)

    # Preprocessing: select → scale  (matches train.py order)
    arr_selected = selector.transform(arr)
    arr_scaled   = scaler.transform(arr_selected)

    # Prediction
    prediction = int(model.predict(arr_scaled)[0])
    prob       = float(model.predict_proba(arr_scaled)[0][1])

    # SHAP explanation
    shap_vals = explainer.shap_values(arr_scaled)

    # Normalise SHAP output across all explainer/model types
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]          # list → class-1 (RF, KernelExplainer)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]    # 3D → class-1 slice (some XGBoost configs)

    shap_vals = shap_vals[0]              # shape: (n_selected_features,)

    # Top 10 contributing features with actual names
    top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
    explanation = [
        {
            "feature_index": int(i),
            "feature_name":  feature_names[i],
            "impact":        float(shap_vals[i]),
        }
        for i in top_indices
    ]

    return {
        "prediction":       prediction,
        "label":            "Parkinson's Detected" if prediction == 1 else "Healthy",
        "probability":      prob,
        "top_contributions": explanation,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
