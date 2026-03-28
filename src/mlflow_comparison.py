"""
Latest per-model metrics from MLflow (matches run_name values in train.py).
"""
from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.model_selection import apply_selection_flags

logger = logging.getLogger("uvicorn.error")

# mlflow.start_run(run_name=...) → tag mlflow.runName
RUN_NAME_TO_DISPLAY: dict[str, str] = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest":       "Random Forest",
    "SVM":                "SVM",
    "KNN":                "KNN",
    "DecisionTree":       "Decision Tree",
    "XGBoost_tuned":      "XGBoost",
}

REQUIRED_METRICS = ("accuracy", "macro_f1", "roc_auc")


def fetch_model_comparison_from_mlflow() -> list[dict[str, Any]]:
    """
    For each known model, take the latest run (by start time), read test metrics,
    return rows sorted by roc_auc descending.

    Selection (selected=True): see ``model_selection.apply_selection_flags`` — interpretable
    models first (combined score with tie-break on ROC AUC); otherwise penalized score.

    Training logs: accuracy, macro_f1, roc_auc (see train.compute_metrics).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise FileNotFoundError(
            f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found."
        )

    runs = client.search_runs(
        experiment_ids=[str(exp.experiment_id)],
        order_by=["start_time DESC"],
        max_results=500,
    )

    latest_by_run_name: dict[str, Any] = {}
    for run in runs:
        tags = getattr(run.data, "tags", None) or {}
        raw = tags.get("mlflow.runName") or getattr(run.info, "run_name", None) or ""
        raw = str(raw).strip()
        if raw not in RUN_NAME_TO_DISPLAY:
            continue
        if raw not in latest_by_run_name:
            latest_by_run_name[raw] = run

    rows: list[dict[str, Any]] = []
    for run_name, run in latest_by_run_name.items():
        metrics = getattr(run.data, "metrics", None) or {}
        missing = [k for k in REQUIRED_METRICS if k not in metrics]
        if missing:
            logger.warning("Skipping run %s: missing metrics %s", run_name, missing)
            continue
        rows.append(
            {
                "model":    RUN_NAME_TO_DISPLAY[run_name],
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "roc_auc":  float(metrics["roc_auc"]),
                "selected": False,
            }
        )

    rows.sort(key=lambda r: r["roc_auc"], reverse=True)
    apply_selection_flags(rows)
    return rows
