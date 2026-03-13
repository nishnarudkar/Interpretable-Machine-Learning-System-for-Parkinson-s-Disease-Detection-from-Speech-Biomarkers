from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

df = pd.read_csv("data/pd_speech_features.csv", header=1)

X = df.drop(["id","class"], axis=1)
y = df["class"]

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label="Train")
plt.plot(train_sizes, test_mean, label="Validation")

plt.legend()

plt.savefig("static/learning_curve.png")