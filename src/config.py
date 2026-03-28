"""
Central configuration — all file paths resolved relative to the project root.
This ensures scripts work correctly regardless of where they are invoked from
(local dev, Docker container, CI/CD agent).
"""
import os
from pathlib import Path

# ── MLflow (API reads experiment metrics; training also uses these) ───────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/nishnarudkar/"
    "Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers.mlflow",
)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "parkinson_detection")

# Project root = parent of this file's directory (src/)
ROOT = Path(__file__).resolve().parent.parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR  = ROOT / "data"
DATA_FILE = DATA_DIR / "pd_speech_features.csv"

# The UCI dataset has a two-row header:
#   row 0 → subject metadata (not column names)
#   row 1 → actual feature names
# header=1 tells pandas to use row 1 as the header.
# If the file is ever re-saved without this offset, loading will silently
# produce wrong column names — the assertion in load_dataset() guards against this.
CSV_HEADER_ROW = 1
TARGET_COLUMN  = "class"
DROP_COLUMNS   = ["id"]

# Expected number of raw feature columns after dropping id/class
EXPECTED_RAW_FEATURES = 753

# ── Training outputs (metrics, reports) ───────────────────────────────────────
ARTIFACTS_DIR       = ROOT / "artifacts"
MODEL_METRICS_PATH  = ARTIFACTS_DIR / "model_metrics.json"
FEATURE_CONFIG_PATH = ARTIFACTS_DIR / "feature_config.json"

# ── Model artifacts ───────────────────────────────────────────────────────────
MODELS_DIR         = ROOT / "models"
MODEL_PATH         = MODELS_DIR / "model.pkl"
SCALER_PATH        = MODELS_DIR / "scaler.pkl"
SELECTOR_PATH      = MODELS_DIR / "selector.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

# ── Static outputs ────────────────────────────────────────────────────────────
STATIC_DIR             = ROOT / "static"
FEATURE_IMPORTANCE_PNG = STATIC_DIR / "feature_importance.png"
LEARNING_CURVE_PNG     = STATIC_DIR / "learning_curve.png"

# ── Templates ─────────────────────────────────────────────────────────────────
TEMPLATES_DIR = ROOT / "templates"


def load_dataset():
    """Load and validate the Parkinson's speech dataset."""
    import pandas as pd

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Run `dvc pull` to fetch it."
        )

    df = pd.read_csv(DATA_FILE, header=CSV_HEADER_ROW)

    # Guard: verify the expected columns exist after loading
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{TARGET_COLUMN}' not found after loading with header={CSV_HEADER_ROW}. "
            "The CSV structure may have changed. Check CSV_HEADER_ROW in config.py."
        )

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(col, axis=1)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Guard: verify feature count matches expectation
    if X.shape[1] != EXPECTED_RAW_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_RAW_FEATURES} features after preprocessing, "
            f"got {X.shape[1]}. Check CSV_HEADER_ROW or DROP_COLUMNS in config.py."
        )

    return X, y
