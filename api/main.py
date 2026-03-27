from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Annotated, List
import joblib
import numpy as np
import shap
import sys
from pathlib import Path

# Resolve project root so config imports work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    load_dataset,
    MODEL_PATH, SCALER_PATH, SELECTOR_PATH, FEATURE_NAMES_PATH,
    TEMPLATES_DIR, STATIC_DIR, EXPECTED_RAW_FEATURES,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

app = FastAPI(title="Parkinson Detection API", version="1.0.0")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
               DecisionTreeClassifier, XGBClassifier)

# ── Load artifacts at startup ─────────────────────────────────────────────────
try:
    model          = joblib.load(MODEL_PATH)
    scaler         = joblib.load(SCALER_PATH)
    selector       = joblib.load(SELECTOR_PATH)
    feature_names  = joblib.load(FEATURE_NAMES_PATH)

    if isinstance(model, TREE_MODELS):
        explainer = shap.TreeExplainer(model)
    else:
        X_bg, _     = load_dataset()
        X_bg        = X_bg.sample(50, random_state=42)
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
    """
    Strictly typed input for the /predict endpoint.
    Pydantic enforces:
      - exactly EXPECTED_RAW_FEATURES float values (no more, no less)
      - each value must be a valid float (rejects strings, nulls, etc.)
    This validation fires before any ML code runs, returning a clean
    422 Unprocessable Entity with a descriptive message on bad input.
    """
    features: Annotated[
        List[float],
        Field(
            min_length=EXPECTED_RAW_FEATURES,
            max_length=EXPECTED_RAW_FEATURES,
            description=f"Exactly {EXPECTED_RAW_FEATURES} numeric speech feature values",
        ),
    ]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(data: FeatureInput):
    try:
        arr = np.array(data.features, dtype=np.float64).reshape(1, -1)

        # Preprocessing: select → scale  (matches train.py order)
        arr_selected = selector.transform(arr)          # 753 → 100 features
        arr_scaled   = scaler.transform(arr_selected)   # scale the 100 selected features

        # Prediction
        prediction = int(model.predict(arr_scaled)[0])
        prob       = float(model.predict_proba(arr_scaled)[0][1])

        # SHAP explanation
        shap_vals = explainer.shap_values(arr_scaled)

        # Normalise SHAP output across all explainer/model types
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        shap_vals = shap_vals[0]

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
            "prediction":        prediction,
            "label":             "Parkinson's Detected" if prediction == 1 else "Healthy",
            "probability":       prob,
            "top_contributions": explanation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
