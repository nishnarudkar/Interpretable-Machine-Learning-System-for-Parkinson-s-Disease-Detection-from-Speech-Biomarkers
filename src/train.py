import pandas as pd
import mlflow
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/pd_speech_features.csv",header=1)

df = df.drop("id",axis=1)

X = df.drop("class",axis=1)
y = df["class"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectKBest(f_classif,k=100)

X_train = selector.fit_transform(X_train,y_train)
X_test = selector.transform(X_test)

model = XGBClassifier()

with mlflow.start_run():

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    mlflow.log_metric("accuracy",acc)

    mlflow.sklearn.log_model(model,"model")

joblib.dump(model,"models/model.pkl")
joblib.dump(scaler,"models/scaler.pkl")
joblib.dump(selector,"models/selector.pkl")