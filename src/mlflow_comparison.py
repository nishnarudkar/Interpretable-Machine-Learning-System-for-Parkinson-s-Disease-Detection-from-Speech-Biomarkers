"""
Fetch model comparison metrics from the saved artifacts/model_metrics.json
(written by train.py after every training run).
Falls back to querying MLflow directly if the file is missing.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_METRICS_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.model_selection import apply_selection_flags


def fetch_model_comparison_from_mlflow() -> list:
    """
    Returns a list of dicts:
      { model, accuracy, macro_f1, roc_auc, selected }

    Strategy:
    1. Read artifacts/model_metrics.json (fast, no network call)
    2. If missing, query MLflow tracking server for the latest run per model name
    """
    # ── Strategy 1: local JSON written by train.py ────────────────────────────
    if MODEL_METRICS_PATH.exists():
        with open(MODEL_METRICS_PATH, encoding="utf-8") as f:
            rows = json.load(f)
        # Re-apply selection flags in case the file was written by an older version
        apply_selection_flags(rows)
        return rows

    # ── Strategy 2: query MLflow ──────────────────────────────────────────────
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if exp is None:
            return []

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="",
            order_by=["start_time DESC"],
            max_results=200,
        )

        # Keep the most recent run per model name
        seen: dict = {}
        for run in runs:
            name = run.data.params.get("model", "")
            if not name or name in seen:
                continue
            m = run.data.metrics
            if not all(k in m for k in ("accuracy", "macro_f1", "roc_auc")):
                continue
            seen[name] = {
                "model":    name,
                "accuracy": round(float(m["accuracy"]),  4),
                "macro_f1": round(float(m["macro_f1"]),  4),
                "roc_auc":  round(float(m["roc_auc"]),   4),
                "selected": False,
            }

        rows = sorted(seen.values(), key=lambda r: r["roc_auc"], reverse=True)
        apply_selection_flags(rows)
        return rows

    except Exception as exc:
        raise RuntimeError(f"MLflow query failed: {exc}") from exc
