pipeline {
  agent any

  environment {
    DAGSHUB_USERNAME = credentials('dagshub-username')
    DAGSHUB_TOKEN    = credentials('dagshub-token')
  }

  stages {

    stage('Install Dependencies') {
      steps {
        sh 'pip install -r requirements.txt'
      }
    }

    stage('Pull Data (DVC)') {
      steps {
        sh '''
          dvc remote modify origin --local auth basic
          dvc remote modify origin --local user $DAGSHUB_USERNAME
          dvc remote modify origin --local password $DAGSHUB_TOKEN
          dvc pull
        '''
      }
    }

    stage('Train Model') {
      steps {
        sh 'python src/train.py'
      }
    }

    stage('Generate Explanations') {
      steps {
        sh 'python src/explain.py'
        sh 'python src/learning_curve.py'
      }
    }

    stage('Smoke Test') {
      steps {
        sh '''
          uvicorn api.main:app --host 0.0.0.0 --port 8000 &
          sleep 5
          curl -f http://localhost:8000/health
          kill %1
        '''
      }
    }

    stage('Build Docker Image') {
      steps {
        sh 'docker build -t parkinson-ml .'
      }
    }

  }

  post {
    always {
      sh 'pkill -f uvicorn || true'
    }
  }
}
