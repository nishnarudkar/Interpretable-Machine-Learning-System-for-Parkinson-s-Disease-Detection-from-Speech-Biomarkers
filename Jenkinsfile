pipeline {
  stages {

    stage('Install Dependencies') {
      steps {
        sh 'pip install -r requirements.txt'
      }
    }

    stage('Train Model') {
      steps {
        sh 'python src/train.py'
      }
    }

    stage('Build Docker Image') {
      steps {
        sh 'docker build -t parkinson-ml .'
      }
    }

  }
}