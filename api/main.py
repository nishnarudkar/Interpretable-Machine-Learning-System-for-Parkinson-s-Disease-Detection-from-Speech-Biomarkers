from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import joblib
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
selector = joblib.load("models/selector.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(features: list):

    arr = np.array(features).reshape(1,-1)

    arr = scaler.transform(arr)
    arr = selector.transform(arr)

    pred = model.predict(arr)

    return {"prediction": int(pred[0])}