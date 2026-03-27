import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold

print("Loading model and preprocessing artifacts...")

model    = joblib.load("models/model.pkl")
scaler   = joblib.load("models/scaler.pkl")
selector = joblib.load("models/selector.pkl")

print("Loading dataset...")

df = pd.read_csv("data/pd_speech_features.csv", header=1)
X  = df.drop(["id", "class"], axis=1)
y  = df["class"]

# Apply the same preprocessing as train.py: select → scale
X_selected = selector.transform(X)
X_scaled   = scaler.transform(X_selected)

print("Computing learning curve (5-fold CV)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, test_scores = learning_curve(
    model,
    X_scaled,
    y,
    cv=cv,
    scoring="f1_macro",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
test_mean  = test_scores.mean(axis=1)
test_std   = test_scores.std(axis=1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_mean, label="Train",      color="#6c63ff", linewidth=2)
plt.plot(train_sizes, test_mean,  label="Validation", color="#34d399", linewidth=2)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#6c63ff")
plt.fill_between(train_sizes, test_mean  - test_std,  test_mean  + test_std,  alpha=0.15, color="#34d399")

plt.xlabel("Training set size")
plt.ylabel("Macro F1 Score")
plt.title("Learning Curve — Bias/Variance Analysis")
plt.legend()
plt.tight_layout()

plt.savefig("static/learning_curve.png", dpi=150)
plt.close()

print("Saved → static/learning_curve.png")
