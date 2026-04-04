# Project Structure

```
.
├── api/
│   └── main.py              # FastAPI app — all endpoints, artifact loading, SHAP inference
├── src/
│   ├── config.py            # Single source of truth for all file paths + load_dataset()
│   ├── train.py             # Full training pipeline: feature selection → scaling → 6 models → MLflow
│   ├── explain.py           # Global SHAP feature importance → static/feature_importance.png
│   ├── learning_curve.py    # Bias-variance learning curve → static/learning_curve.png
│   ├── model_selection.py   # Interpretability-aware model selection logic
│   └── mlflow_comparison.py # Fetches per-model metrics from MLflow for the UI
├── data/
│   └── pd_speech_features.csv      # Dataset (DVC tracked, 756 rows × 755 cols)
├── models/                  # Trained artifacts (gitignored, produced by train.py)
│   ├── model.pkl            # Production XGBoost model
│   ├── scaler.pkl           # Fitted StandardScaler (100 features)
│   ├── selector.pkl         # Fitted SelectFromModel (753 → 100 features)
│   ├── feature_names.pkl    # List of 100 selected feature names
│   └── column_order.pkl     # Training column order (753 features, used by API)
├── artifacts/
│   ├── model_metrics.json   # Per-model comparison metrics (accuracy, f1, roc_auc)
│   └── feature_config.json  # Feature configuration metadata
├── static/                  # Frontend assets + generated charts (served by FastAPI)
│   ├── feature_importance.png
│   ├── learning_curve.png
│   ├── shap_bar.png         # Per-prediction SHAP chart (overwritten on each /predict)
│   ├── feature_medians.json # Feature defaults for the prediction form
│   ├── feature_insights.json
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html           # Jinja2 template for the 4-tab dark UI
├── monitoring/
│   ├── baseline_data.csv    # X_train selected features saved after training
│   ├── current_data.csv     # API inputs appended on every /predict call
│   └── drift_check.py       # Evidently drift report generator
├── notebooks/
│   └── Parkinsons_Detection_MLOPS_Project_SMOTE.ipynb
├── dvc.yaml                 # DVC pipeline: train → explain → learning_curve
├── Dockerfile               # Multi-stage build (runtime uses requirements-api.txt)
├── Jenkinsfile              # 7-stage CI/CD pipeline
├── requirements.txt         # Full pinned deps (training + API)
├── requirements-api.txt     # Minimal deps for Docker runtime image
└── .env / .env.example      # Credentials (never commit .env)
```

## Key Conventions

- **All paths go through `src/config.py`** — never hardcode paths in other modules. `ROOT` is resolved relative to `config.py`'s location so scripts work from any working directory.
- **Preprocessing order is select → scale** — `selector.transform()` first, then `scaler.transform()`. This order must be consistent between `train.py` and `api/main.py`.
- **SMOTE inside CV folds only** — use `ImbPipeline` so validation folds always contain real, unaugmented data.
- **DVC pipeline is the canonical way to run training** — `dvc repro` handles dependency tracking. Run individual scripts only for development/debugging.
- **Two requirements files** — `requirements.txt` for full dev/training environment; `requirements-api.txt` for the lean Docker runtime (no training deps).
- **Monitoring baseline** — `monitoring/baseline_data.csv` holds unscaled selected features (post-selector, pre-scaler) from training. The API logs the same representation to `current_data.csv` for apples-to-apples drift comparison.
- **MLflow tracking** — remote tracking URI points to DagsHub; local prediction tracking writes to `mlruns/` using `mlflow.set_tracking_uri("mlruns")`.
- **Jenkins runs on Windows** — `Jenkinsfile` uses `bat` steps, not `sh`.
