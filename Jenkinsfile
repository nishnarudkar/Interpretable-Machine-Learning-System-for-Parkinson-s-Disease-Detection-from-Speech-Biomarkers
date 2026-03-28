pipeline {
  agent any

  environment {
    DAGSHUB_USERNAME = credentials('dagshub-username')
    DAGSHUB_TOKEN    = credentials('dagshub-token')
    PYTHONIOENCODING = 'utf-8'
  }

  stages {

    stage('Install Dependencies') {
      steps {
        bat 'pip install -r requirements.txt'
      }
    }

    stage('Pull Data (DVC)') {
      steps {
        bat '''
          dvc remote modify origin --local auth basic
          dvc remote modify origin --local user %DAGSHUB_USERNAME%
          dvc remote modify origin --local password %DAGSHUB_TOKEN%
          dvc pull data/pd_speech_features.csv.dvc --force
        '''
      }
    }

    stage('Train Model') {
      steps {
        bat 'python src/train.py'
      }
    }

    stage('Generate Explanations') {
      steps {
        bat 'python src/explain.py'
        bat 'python src/learning_curve.py'
      }
    }

    stage('Smoke Test') {
      steps {
        bat '''
          start /b uvicorn api.main:app --host 0.0.0.0 --port 8000
          timeout /t 5 /nobreak
          curl -f http://localhost:8000/health
          taskkill /F /IM uvicorn.exe || exit 0
        '''
      }
    }

    stage('Build Docker Image') {
      steps {
        bat 'docker build -t parkinson-api .'
      }
    }

  }

  post {
    always {
      bat 'taskkill /F /IM uvicorn.exe || exit 0'
    }
  }
}
