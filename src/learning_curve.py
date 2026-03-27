import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("Loading model and dataset...")

model = joblib.load("models/model.pkl")

df = pd.read_csv("data/pd_speech_features.csv", header=1)
X  = df.drop(["id", "class"], axis=1)
y  = df["class"]

# ---------------------------------------------------------------
# Wrap the full pipeline so learning_curve refits preprocessing
# on each CV fold — avoids leakage from pre-fitted selector/scaler
# ---------------------------------------------------------------
pipeline = ImbPipeline([
    ("smote",    SMOTE(random_state=42)),
    ("selector", SelectFromModel(
                     RandomForestClassifier(n_estimators=100, random_state=42),
                     max_features=100
                 )),
    ("scaler",   StandardScaler()),
    ("model",    model),
])

print("Computing learning curve (5-fold stratified CV)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, test_scores = learning_curve(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
    train_sizes=np.linspace(0.2, 1.0, 8),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
test_mean  = test_scores.mean(axis=1)
test_std   = test_scores.std(axis=1)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(train_sizes, train_mean, label="Train",      color="#6c63ff", linewidth=2)
ax.plot(train_sizes, test_mean,  label="Validation", color="#34d399", linewidth=2)
ax.fill_between(train_sizes,
                train_mean - train_std, train_mean + train_std,
                alpha=0.15, color="#6c63ff")
ax.fill_between(train_sizes,
                test_mean - test_std,  test_mean + test_std,
                alpha=0.15, color="#34d399")

# Annotate the gap at the largest training size
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

plt.savefig("static/learning_curve.png", dpi=150)
plt.close()

final_gap = train_mean[-1] - test_mean[-1]
print(f"Saved → static/learning_curve.png")
print(f"Train F1: {train_mean[-1]:.4f} | Val F1: {test_mean[-1]:.4f} | Gap: {final_gap:.4f}")

if final_gap > 0.15:
    print("WARNING: Large train/val gap — model may be overfitting.")
    print("Consider: more regularization, lower max_depth, or more training data.")
else:
    print("Gap is within acceptable range for a tree ensemble.")
