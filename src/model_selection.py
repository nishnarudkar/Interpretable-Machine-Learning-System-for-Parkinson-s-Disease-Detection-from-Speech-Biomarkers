"""
Shared logic for marking the "selected" model in comparison tables.

Prefers interpretable models (XGBoost, Random Forest, Decision Tree), ranking by
combined score 0.6 * roc_auc + 0.4 * macro_f1. Other models use the same score
with a -0.02 penalty and are only chosen if no interpretable model is present.
"""
from __future__ import annotations

from typing import Any

INTERPRETABLE_MODELS: frozenset[str] = frozenset(
    {"XGBoost", "Random Forest", "Decision Tree"}
)
NON_INTERPRETABLE_PENALTY = 0.02


def combined_score(roc_auc: float, macro_f1: float) -> float:
    return 0.6 * roc_auc + 0.4 * macro_f1


def _is_interpretable(display_name: str) -> bool:
    return display_name in INTERPRETABLE_MODELS


def _ranked_score(row: dict[str, Any]) -> float:
    s = combined_score(row["roc_auc"], row["macro_f1"])
    if not _is_interpretable(row["model"]):
        s -= NON_INTERPRETABLE_PENALTY
    return s


def select_best_model_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    interpretable = [r for r in rows if _is_interpretable(r["model"])]
    if interpretable:
        return max(
            interpretable,
            key=lambda r: (
                combined_score(r["roc_auc"], r["macro_f1"]),
                r["roc_auc"],
            ),
        )
    return max(
        rows,
        key=lambda r: (_ranked_score(r), r["roc_auc"]),
    )


def apply_selection_flags(rows: list[dict[str, Any]]) -> None:
    """Set selected True on the chosen row only (mutates rows in place)."""
    chosen = select_best_model_row(rows)
    if chosen is None:
        return
    name = chosen["model"]
    for r in rows:
        r["selected"] = r["model"] == name
