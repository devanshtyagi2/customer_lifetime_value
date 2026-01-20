import json
import joblib
import pandas as pd
from src.utils.config import Paths

def predict_one(sample: dict) -> dict:
    model = joblib.load(Paths.CHURN_MODEL)
    features = json.load(open(Paths.FEATURES_JSON))
    threshold = json.load(open(Paths.THRESHOLD_JSON))["threshold"]

    X = pd.DataFrame([sample])[features]
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= threshold)
    return {"churn_probability": prob, "churn_prediction": pred, "threshold": threshold}

if __name__ == "__main__":
    sample = {
        "avg_order_value": 350,
        "invoice_count": 6,
        "total_quantity": 40,
        "tenure_days": 180
    }
    print(predict_one(sample))
