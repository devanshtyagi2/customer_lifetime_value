import json
import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import CustomerFeatures, ChurnResponse, CLVResponse

app = FastAPI(
    title="Churn + CLV API",
    version="1.0.0",
    description="Predict customer churn probability and risk-adjusted CLV"
)

MODEL_PATH = "models/churn_xgb.pkl"
FEATURES_PATH = "models/features.json"
THRESHOLD_PATH = "models/threshold.json"

# Load artifacts once at startup
model = joblib.load(MODEL_PATH)
FEATURES = json.load(open(FEATURES_PATH))
THRESHOLD = json.load(open(THRESHOLD_PATH))["threshold"]


@app.get("/")
def home():
    return {"status": "ok", "message": "Churn + CLV API running"}


@app.post("/predict_churn", response_model=ChurnResponse)
def predict_churn(payload: CustomerFeatures):
    row = pd.DataFrame([payload.model_dump()])
    X = row[FEATURES]

    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= THRESHOLD)

    return {
        "churn_probability": prob,
        "churn_prediction": pred,
        "threshold": THRESHOLD
    }


@app.post("/predict_clv", response_model=CLVResponse)
def predict_clv(payload: CustomerFeatures, raw_clv: float):
    row = pd.DataFrame([payload.model_dump()])
    X = row[FEATURES]

    prob = float(model.predict_proba(X)[:, 1][0])
    final_clv = float(raw_clv * (1 - prob))

    return {
        "raw_clv": raw_clv,
        "churn_probability": prob,
        "final_clv": final_clv
    }
