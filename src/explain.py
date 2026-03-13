import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/model.pkl")

df = pd.read_csv("data/pd_speech_features.csv", header=1)

X = df.drop(["id","class"], axis=1)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, show=False)

plt.savefig("static/feature_importance.png")