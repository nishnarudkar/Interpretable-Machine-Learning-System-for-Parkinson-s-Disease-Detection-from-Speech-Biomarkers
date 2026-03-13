# Interpretable Machine Learning System for Parkinson's Disease Detection from Speech Biomarkers

An end-to-end MLOps pipeline for detecting Parkinson's disease using speech biomarkers with explainable AI (XAI) capabilities. This system combines machine learning, model interpretability (SHAP), experiment tracking (MLflow), data versioning (DVC), and automated deployment (Docker + Jenkins).

## 🎯 Project Overview

This project implements an interpretable machine learning system that:
- Detects Parkinson's disease from speech feature data
- Provides model explanations using SHAP (SHapley Additive exPlanations)
- Tracks experiments and model versions with MLflow and DagsHub
- Versions datasets using DVC (Data Version Control)
- Deploys as a web application with FastAPI
- Automates CI/CD with Jenkins
- Containerizes the application with Docker

## 🏗️ Architecture

```
├── data/                          # Dataset storage (DVC tracked)
├── src/                           # Training and analysis scripts
│   ├── train.py                   # Model training with hyperparameter tuning
│   ├── explain.py                 # SHAP-based feature importance
│   └── learning_curve.py          # Bias-variance analysis
├── api/                           # FastAPI application
│   └── main.py                    # REST API with prediction endpoint
├── models/                        # Trained model artifacts
├── static/                        # Frontend assets and visualizations
├── templates/                     # HTML templates
├── mlruns/                        # MLflow experiment tracking
├── dvc.yaml                       # DVC pipeline definition
├── Dockerfile                     # Container configuration
├── Jenkinsfile                    # CI/CD pipeline
└── requirements.txt               # Python dependencies
```

## 🚀 Features

### 1. Machine Learning Pipeline
- **Data Preprocessing**: StandardScaler normalization and SelectKBest feature selection (top 100 features)
- **Model Training**: Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Model Registry**: Best model automatically registered in MLflow

### 2. Explainable AI (XAI)
- **SHAP Values**: TreeExplainer for model-agnostic interpretability
- **Feature Importance**: Global feature importance visualization
- **Local Explanations**: Per-prediction SHAP contributions for top 10 features
- **Learning Curves**: Bias-variance tradeoff analysis

### 3. MLOps Infrastructure
- **Experiment Tracking**: MLflow integration with DagsHub
- **Data Versioning**: DVC for dataset management
- **Pipeline Automation**: DVC pipelines for reproducible workflows
- **Model Versioning**: Automatic model registration and artifact storage

### 4. Web Application
- **Interactive UI**: Three-tab interface (Feature Importance, Learning Curve, Prediction)
- **Real-time Predictions**: REST API endpoint with probability scores
- **Visual Explanations**: Dynamic SHAP charts using Chart.js
- **Responsive Design**: Clean, user-friendly interface

### 5. Deployment
- **Containerization**: Docker image for consistent deployment
- **CI/CD**: Jenkins pipeline for automated testing and deployment
- **Production Ready**: FastAPI with Uvicorn ASGI server

## 📋 Prerequisites

- Python 3.10+
- pip
- Docker (optional, for containerized deployment)
- Jenkins (optional, for CI/CD)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Pull dataset with DVC**
```bash
dvc pull
```

## 🎓 Usage

### Training the Model

Run the complete training pipeline with hyperparameter tuning:

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Train baseline models (Logistic Regression, Random Forest)
- Perform XGBoost hyperparameter tuning (depth: 3/5/7, learning_rate: 0.05/0.1, n_estimators: 100/200)
- Log all experiments to MLflow
- Save the best model to `models/` directory
- Register the best model in MLflow Model Registry

### Generate Explanations

Create SHAP-based feature importance visualizations:

```bash
python src/explain.py
```

Generates `static/feature_importance.png` showing the top 20 most influential speech biomarkers.

### Generate Learning Curves

Analyze model bias-variance tradeoff:

```bash
python src/learning_curve.py
```

Creates `static/learning_curve.png` with training and validation performance curves.

### Run DVC Pipeline

Execute the complete pipeline (train + explain):

```bash
dvc repro
```

### Start the Web Application

Launch the FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Access the application at `http://localhost:8000`

### API Endpoints

**GET /** - Web interface

**POST /predict** - Prediction endpoint
```json
{
  "features": [0.85, 0.71, 0.57, 240, 239, ...]
}
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.87,
  "top_contributions": [
    {"feature_index": 42, "impact": 0.23},
    {"feature_index": 15, "impact": -0.18},
    ...
  ]
}
```

## 🐳 Docker Deployment

Build the Docker image:

```bash
docker build -t parkinson-ml .
```

Run the container:

```bash
docker run -p 8000:8000 parkinson-ml
```

## 🔄 CI/CD with Jenkins

The `Jenkinsfile` defines a pipeline with three stages:

1. **Install Dependencies**: Install Python packages
2. **Train Model**: Execute training script
3. **Build Docker Image**: Create containerized application

Configure Jenkins to use this Jenkinsfile for automated deployments.

## 📊 Dataset

The project uses the **Parkinson's Disease Speech Dataset** (`pd_speech_features.csv`) containing:
- Speech biomarker features extracted from voice recordings
- Binary classification target (Parkinson's vs. Healthy)
- Tracked with DVC and stored on DagsHub

Dataset size: ~5.3 MB (5,308,926 bytes)

## 🛠️ Technology Stack

- **ML Framework**: scikit-learn, XGBoost
- **Explainability**: SHAP
- **Experiment Tracking**: MLflow, DagsHub
- **Data Versioning**: DVC
- **Web Framework**: FastAPI
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Containerization**: Docker
- **CI/CD**: Jenkins
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 📈 Model Performance

The system trains and compares multiple models:
- Logistic Regression (baseline)
- Random Forest (200 estimators)
- XGBoost (with grid search over 12 hyperparameter combinations)

Best model is automatically selected based on test accuracy and registered in MLflow.

## 🔍 Interpretability Features

### Global Interpretability
- **Feature Importance Chart**: Top 20 speech biomarkers ranked by SHAP values
- **Learning Curves**: Model performance vs. training set size

### Local Interpretability
- **Per-Prediction Explanations**: SHAP values for individual predictions
- **Feature Contributions**: Top 10 features driving each prediction
- **Interactive Visualization**: Dynamic bar charts in the web interface

## 📁 Key Files

- `src/train.py`: Model training with MLflow tracking
- `src/explain.py`: SHAP-based feature importance generation
- `src/learning_curve.py`: Bias-variance analysis
- `api/main.py`: FastAPI application with prediction endpoint
- `dvc.yaml`: DVC pipeline configuration
- `Dockerfile`: Container definition
- `Jenkinsfile`: CI/CD pipeline
- `requirements.txt`: Python dependencies

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

- **nishnarudkar** - [DagsHub Profile](https://dagshub.com/nishnarudkar)

## 🙏 Acknowledgments

- Dataset: Parkinson's Disease Speech Biomarkers
- MLflow and DagsHub for experiment tracking
- SHAP library for model interpretability
- FastAPI for the web framework

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Note**: This is a research/educational project. For medical applications, consult with healthcare professionals and ensure proper validation and regulatory compliance.
