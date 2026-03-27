import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os

from src.config import (
    load_dataset,
    MODEL_PATH, LEARNING_CURVE_PNG, STATIC_DIR,
)

print("Loading model and dataset...")
model  = joblib.load(MODEL_PATH)
X, y   = load_dataset()

# ── Full pipeline — preprocessing refitted per CV fold (no leakage) ───────────
pipeline = ImbPipeline([
    ("smote",    SMOTE(random_state=42)),
    ("selector", SelectFromModel(
                     RandomForestClassifier(n_estimators=100, random_state=42),
                     max_features=100,
                 )),
    ("scaler",   StandardScaler()),
    ("model",    model),
])

print("Computing learning curve (5-fold stratified CV)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y,
    cv=cv,
    scoring="f1_macro",
    train_sizes=np.linspace(0.2, 1.0, 8),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
test_mean  = test_scores.mean(axis=1)
test_std   = test_scores.std(axis=1)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_mean, label="Train",      color="#6c63ff", linewidth=2)
ax.plot(train_sizes, test_mean,  label="Validation", color="#34d399", linewidth=2)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.15, color="#6c63ff")
ax.fill_between(train_sizes, test_mean - test_std,   test_mean + test_std,
                alpha=0.15, color="#34d399")

gap = train_mean[-1] - test_mean[-1]
ax.annotate(
    f"Gap: {gap:.3f}",
    xy=(train_sizes[-1], (train_mean[-1] + test_mean[-1]) / 2),
    xytext=(-80, 0), textcoords="offset points",
    fontsize=9, color="#f87171",
    arrowprops=dict(arrowstyle="->", color="#f87171"),
)

ax.set_xlabel("Training set size")
ax.set_ylabel("Macro F1 Score")
ax.set_title("Learning Curve — Bias/Variance Analysis\n"
             "(preprocessing refitted per fold — no leakage)")
ax.legend()
ax.set_ylim(0, 1.05)
fig.tight_layout()

os.makedirs(STATIC_DIR, exist_ok=True)
plt.savefig(LEARNING_CURVE_PNG, dpi=150)
plt.close()

print(f"Saved → {LEARNING_CURVE_PNG}")
print(f"Train F1: {train_mean[-1]:.4f} | Val F1: {test_mean[-1]:.4f} | Gap: {gap:.4f}")
if gap > 0.15:
    print("WARNING: Large train/val gap — model may be overfitting.")
else:
    print("Gap is within acceptable range for a tree ensemble.")
