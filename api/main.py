from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Annotated, List
import joblib
import numpy as np
import pandas as pd
import shap
import sys
import logging
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

# Resolve project root so config imports work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    load_dataset,
    MODEL_PATH, SCALER_PATH, SELECTOR_PATH, FEATURE_NAMES_PATH,
    TEMPLATES_DIR, STATIC_DIR, EXPECTED_RAW_FEATURES, MODELS_DIR,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

COLUMN_ORDER_PATH = MODELS_DIR / "column_order.pkl"
TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
               DecisionTreeClassifier, XGBClassifier)

app = FastAPI(title="Parkinson Detection API", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── SHAP normalisation ────────────────────────────────────────────────────────
def extract_shap_for_class1(raw: object) -> np.ndarray:
    """
    Robustly extract a 1-D SHAP array for the positive class (class 1)
    regardless of how the installed SHAP version returns values.

    Handles all known output shapes:
      - list of arrays  → [class0_arr, class1_arr]  (sklearn RF, old SHAP)
      - 3-D ndarray     → (n_samples, n_features, n_classes)  (some XGBoost)
      - 2-D ndarray     → (n_samples, n_features)  (XGBoost binary, new SHAP)
      - 1-D ndarray     → (n_features,)  (single-sample shortcut)
    """
    if isinstance(raw, list):
        # list output: take class-1 slice, then squeeze to 1-D
        arr = np.array(raw[1])
    else:
        arr = np.array(raw)

    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) → class-1 slice
        arr = arr[:, :, 1]

    # Now arr is (n_samples, n_features) or (n_features,)
    if arr.ndim == 2:
        arr = arr[0]   # single prediction → (n_features,)

    if arr.ndim != 1:
        raise ValueError(
            f"Unexpected SHAP output shape after normalisation: {arr.shape}. "
            "Please check your SHAP version."
        )
    return arr


# ── Explainer factory ─────────────────────────────────────────────────────────
def build_explainer(model, selector, scaler):
    """
    Build the appropriate SHAP explainer.
    For non-tree models, falls back to KernelExplainer with a safe
    background dataset — handles small datasets and sampling failures.
    """
    if isinstance(model, TREE_MODELS):
        logger.info("Using SHAP TreeExplainer")
        return shap.TreeExplainer(model)

    logger.info(f"Model is {type(model).__name__} — building KernelExplainer background")
    try:
        X_all, _ = load_dataset()
        n_bg = min(50, len(X_all))          # safe even on tiny datasets
        X_bg = X_all.sample(n_bg, random_state=42)
        X_bg_sel    = selector.transform(X_bg)
        X_bg_scaled = scaler.transform(X_bg_sel)

        n_clusters = min(10, n_bg)          # kmeans needs k ≤ n_samples
        background  = shap.kmeans(X_bg_scaled, n_clusters)
        logger.info(f"KernelExplainer background: {n_bg} samples → {n_clusters} clusters")
    except Exception as e:
        # Last-resort fallback: use the feature-wise mean as a single background point
        logger.warning(
            f"Background sampling failed ({e}). "
            "Falling back to zero-vector background for KernelExplainer."
        )
        n_features  = scaler.n_features_in_
        background  = np.zeros((1, n_features))

    return shap.KernelExplainer(model.predict_proba, background)


# ── Load artifacts at startup ─────────────────────────────────────────────────
try:
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    selector      = joblib.load(SELECTOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    column_order  = joblib.load(COLUMN_ORDER_PATH)

    # Validate the preprocessing chain is internally consistent
    assert len(column_order) == EXPECTED_RAW_FEATURES, (
        f"column_order has {len(column_order)} entries, expected {EXPECTED_RAW_FEATURES}"
    )
    assert selector.n_features_in_ == EXPECTED_RAW_FEATURES, (
        f"selector expects {selector.n_features_in_} features, expected {EXPECTED_RAW_FEATURES}"
    )
    assert scaler.n_features_in_ == len(feature_names), (
        f"scaler expects {scaler.n_features_in_} features but feature_names has {len(feature_names)}"
    )

    explainer = build_explainer(model, selector, scaler)
    logger.info(f"Loaded: {type(model).__name__} + {type(explainer).__name__}")

except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artifact not found: {e}. Run `python src/train.py` first."
    )
except AssertionError as e:
    raise RuntimeError(f"Artifact consistency check failed: {e}")
# ─────────────────────────────────────────────────────────────────────────────


class FeatureInput(BaseModel):
    """
    Strictly typed input — Pydantic enforces exactly EXPECTED_RAW_FEATURES
    float values at the schema level before any ML code runs.
    Wrong length → 422 Unprocessable Entity.
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
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
def predict(data: FeatureInput):
    try:
        # Align to training column order to prevent silent feature mismatch
        arr = pd.DataFrame([data.features], columns=column_order).values

        # Preprocessing: select → scale  (matches train.py order)
        arr_selected = selector.transform(arr)        # 753 → 100 features
        arr_scaled   = scaler.transform(arr_selected) # standardise

        # Prediction
        prediction = int(model.predict(arr_scaled)[0])
        prob       = float(model.predict_proba(arr_scaled)[0][1])

        # SHAP — robust extraction regardless of SHAP version / model type
        raw_shap  = explainer.shap_values(arr_scaled)
        shap_vals = extract_shap_for_class1(raw_shap)   # guaranteed 1-D

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
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        type(model).__name__,
        "explainer":    type(explainer).__name__,
        "model_loaded": model is not None,
    }
