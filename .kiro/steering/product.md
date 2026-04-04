# Product

An interpretable MLOps system that detects Parkinson's disease from speech biomarkers. It trains and compares 6 classifiers, selects XGBoost as the production model (chosen for SHAP TreeExplainer compatibility, not just raw performance), and serves predictions via a FastAPI web app with per-prediction SHAP explanations.

## Core Capabilities

- Multi-model training with RandomizedSearchCV + StratifiedKFold CV
- SMOTE applied inside CV folds to prevent data leakage
- Feature selection: 753 raw speech features → 100 via SelectFromModel (Random Forest)
- Global and per-prediction SHAP explanations (TreeExplainer)
- Dark-themed 4-tab UI: Feature Importance, Learning Curve, Prediction, Model Comparison
- Data drift monitoring via Evidently (baseline vs. production inputs)
- Full MLOps stack: MLflow experiment tracking, DVC data versioning, Docker, Jenkins CI/CD

## Target Users

Researchers and clinicians exploring ML-assisted Parkinson's screening. This is a research/educational tool — not a validated clinical diagnostic system.

## Production Model

XGBoost (accuracy: 0.89, macro F1: 0.855, ROC AUC: 0.946). Selected over higher-scoring KNN/SVM because SHAP TreeExplainer provides fast, exact feature attribution essential for clinical transparency.
