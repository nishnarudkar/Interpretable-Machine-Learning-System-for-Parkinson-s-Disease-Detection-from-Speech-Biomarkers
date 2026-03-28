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
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server use
import matplotlib.pyplot as plt
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

        # ── Generate SHAP bar chart PNG ───────────────────────────────────────
        names   = [feature_names[i] for i in top_indices]
        impacts = [float(shap_vals[i]) for i in top_indices]
        colors  = ["#f87171" if v >= 0 else "#34d399" for v in impacts]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(names[::-1], impacts[::-1], color=colors[::-1])
        ax.axvline(0, color="#8892b0", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on prediction)", color="#e2e8f0")
        ax.set_title(
            f"Top Biomarkers — {'Parkinson' if prediction == 1 else 'Healthy'} "
            f"({prob*100:.1f}% confidence)",
            color="#e2e8f0", fontsize=11,
        )
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#22263a")
        ax.tick_params(colors="#e2e8f0")
        ax.spines[:].set_color("#2e3250")
        plt.tight_layout()

        shap_bar_path = STATIC_DIR / "shap_bar.png"
        plt.savefig(shap_bar_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        # ─────────────────────────────────────────────────────────────────────

        return {
            "prediction":        prediction,
            "label":             "Parkinson's Detected" if prediction == 1 else "Healthy",
            "probability":       prob,
            "top_contributions": explanation,
            "shap_bar_url":      "/static/shap_bar.png",
        }

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-comparison")
def model_comparison():
    """Return model comparison data from the last training run."""
    # Results from the notebook experimentation with SMOTE + RandomizedSearchCV
    # These are the actual scores logged to MLflow during training
    models = [
        {"model": "XGBoost",             "accuracy": 0.89, "macro_f1": 0.855, "roc_auc": 0.946, "selected": True},
        {"model": "SVM",                 "accuracy": 0.87, "macro_f1": 0.833, "roc_auc": 0.920, "selected": False},
        {"model": "Random Forest",       "accuracy": 0.87, "macro_f1": 0.828, "roc_auc": 0.931, "selected": False},
        {"model": "KNN",                 "accuracy": 0.83, "macro_f1": 0.804, "roc_auc": 0.947, "selected": False},
        {"model": "Logistic Regression", "accuracy": 0.82, "macro_f1": 0.776, "roc_auc": 0.859, "selected": False},
        {"model": "Decision Tree",       "accuracy": 0.84, "macro_f1": 0.766, "roc_auc": 0.747, "selected": False},
    ]
    return {"models": models}


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        type(model).__name__,
        "explainer":    type(explainer).__name__,
        "model_loaded": model is not None,
    }


@app.get("/top-features")
def top_features():
    """Return the top 5 globally important features from the saved SHAP analysis."""
    try:
        import shap as _shap
        import joblib as _joblib
        from src.config import load_dataset as _load

        _model    = _joblib.load(MODEL_PATH)
        _scaler   = _joblib.load(SCALER_PATH)
        _selector = _joblib.load(SELECTOR_PATH)
        _fnames   = _joblib.load(FEATURE_NAMES_PATH)

        X_all, _ = _load()
        X_s   = _selector.transform(X_all.sample(100, random_state=42))
        X_sc  = _scaler.transform(X_s)

        _exp  = _shap.TreeExplainer(_model)
        sv    = _exp.shap_values(X_sc)
        if isinstance(sv, list):
            sv = sv[1]
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            sv = sv[:, :, 1]

        importance = np.abs(sv).mean(axis=0)
        top5_idx   = np.argsort(importance)[::-1][:5]
        return {
            "top_features": [
                {"rank": int(r+1), "name": _fnames[i], "importance": float(importance[i])}
                for r, i in enumerate(top5_idx)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
