# Tech Stack

## Language & Runtime
- Python 3.10+

## ML / Data
- scikit-learn 1.8 — pipelines, feature selection, cross-validation
- XGBoost 3.2 — production model
- imbalanced-learn 0.14 — ImbPipeline + SMOTE
- SHAP 0.51 — TreeExplainer (global + per-prediction)
- pandas 2.3, numpy 2.4, scipy 1.17
- matplotlib 3.10, seaborn 0.13

## MLOps
- MLflow 3.10 — experiment tracking, model registry (remote: DagsHub)
- DVC 3.67 — data versioning (remote: DagsHub)
- DagsHub 0.6.9 — remote storage for MLflow + DVC
- Evidently 0.7 — data drift detection

## API / Serving
- FastAPI 0.135 + Uvicorn 0.42
- Pydantic 2.12 — request validation
- Jinja2 3.1 — HTML templating
- joblib 1.5 — artifact serialization

## Frontend
- Vanilla HTML/CSS/JS + Chart.js (dark theme)

## Infrastructure
- Docker (multi-stage build — runtime uses `requirements-api.txt` only)
- Jenkins — 7-stage CI/CD pipeline (Windows agent, uses `bat` steps)
- python-dotenv — `.env` for credentials

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Pull dataset
dvc pull

# Run full training pipeline (recommended)
dvc repro

# Run steps individually
python src/train.py          # train all 6 models, save artifacts
python src/explain.py        # generate SHAP feature importance chart
python src/learning_curve.py # generate bias-variance plot

# Serve the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Drift detection
python monitoring/drift_check.py

# Docker
docker build -t parkinson-ml .
docker run -p 8000:8000 parkinson-ml

# View MLflow UI (local)
mlflow ui   # opens http://localhost:5000
```

## Environment Variables
Credentials are loaded from `.env` (see `.env.example`):
- `DAGSHUB_USERNAME`
- `DAGSHUB_TOKEN`
- `MLFLOW_TRACKING_URI` (defaults to DagsHub remote)
- `MLFLOW_EXPERIMENT_NAME` (defaults to `parkinson_detection`)

In Jenkins, credentials are injected via the credentials store — never hardcoded.
