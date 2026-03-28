"""
Model selection logic: marks the best interpretable model as selected.
Interpretable models (XGBoost, Random Forest, Decision Tree) are preferred
because SHAP TreeExplainer works natively with them — critical for a medical app.
"""

INTERPRETABLE_MODELS = {"XGBoost", "Random Forest", "Decision Tree"}


def apply_selection_flags(rows: list) -> None:
    """
    Mutates each row dict in-place, setting 'selected': True on the best model.

    Selection priority:
    1. Best interpretable model by composite score (0.6 * roc_auc + 0.4 * macro_f1)
    2. If no interpretable model exists, fall back to overall best macro_f1
    """
    if not rows:
        return

    def composite(row):
        return 0.6 * float(row.get("roc_auc", 0)) + 0.4 * float(row.get("macro_f1", 0))

    interpretable = [r for r in rows if r["model"] in INTERPRETABLE_MODELS]
    candidates    = interpretable if interpretable else rows

    best = max(candidates, key=composite)

    for row in rows:
        row["selected"] = (row is best)
