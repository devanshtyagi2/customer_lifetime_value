import os, json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier


DATA_PATH = "data/processed/customer_churn_ready.csv"   # your churn-ready dataset
MODEL_DIR = "models"

FEATURES = ["avg_order_value", "invoice_count", "total_quantity", "tenure_days"]
TARGET = "churn_flag"

FINAL_THRESHOLD = 0.30


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_experiment("Churn_Prediction_CLV")

    with mlflow.start_run(run_name="xgboost_churn_tuned"):
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        # default threshold 0.50
        y_pred_05 = (y_prob >= 0.50).astype(int)

        # tuned threshold 0.30
        y_pred_tuned = (y_prob >= FINAL_THRESHOLD).astype(int)

        # metrics
        roc = roc_auc_score(y_test, y_prob)

        acc_05 = accuracy_score(y_test, y_pred_05)
        rec_05 = recall_score(y_test, y_pred_05)
        prec_05 = precision_score(y_test, y_pred_05)

        acc_t = accuracy_score(y_test, y_pred_tuned)
        rec_t = recall_score(y_test, y_pred_tuned)
        prec_t = precision_score(y_test, y_pred_tuned)

        # log params
        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("threshold_tuned", FINAL_THRESHOLD)

        # log metrics
        # log metrics (MLflow-safe names)
        mlflow.log_metric("roc_auc", roc)

        mlflow.log_metric("accuracy_0_50", acc_05)
        mlflow.log_metric("recall_0_50", rec_05)
        mlflow.log_metric("precision_0_50", prec_05)

        mlflow.log_metric("accuracy_tuned", acc_t)
        mlflow.log_metric("recall_tuned", rec_t)
        mlflow.log_metric("precision_tuned", prec_t)


        # save artifacts locally
        joblib.dump(model, os.path.join(MODEL_DIR, "churn_xgb.pkl"))
        with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
            json.dump(FEATURES, f)
        with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
            json.dump({"threshold": FINAL_THRESHOLD}, f)

        # log model artifact to mlflow
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("✅ MLflow run complete")
    print("✅ Saved model -> models/churn_xgb.pkl")


if __name__ == "__main__":
    main()
