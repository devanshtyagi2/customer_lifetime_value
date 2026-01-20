import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.utils.config import Paths, ModelConfig


def main():
    os.makedirs(Paths.MODEL_DIR, exist_ok=True)

    df = pd.read_csv(Paths.MODEL_INPUT_CSV)
    X = df[list(ModelConfig.FEATURES)]
    y = df[ModelConfig.TARGET]

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
        y_pred_05 = (y_prob >= 0.50).astype(int)
        y_pred_tuned = (y_prob >= ModelConfig.THRESHOLD).astype(int)

        # log params
        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("features", ",".join(ModelConfig.FEATURES))
        mlflow.log_param("threshold_tuned", ModelConfig.THRESHOLD)

        # log metrics
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

        mlflow.log_metric("accuracy_0_50", accuracy_score(y_test, y_pred_05))
        mlflow.log_metric("recall_0_50", recall_score(y_test, y_pred_05))
        mlflow.log_metric("precision_0_50", precision_score(y_test, y_pred_05))

        mlflow.log_metric("accuracy_tuned", accuracy_score(y_test, y_pred_tuned))
        mlflow.log_metric("recall_tuned", recall_score(y_test, y_pred_tuned))
        mlflow.log_metric("precision_tuned", precision_score(y_test, y_pred_tuned))

        # save model artifact locally
        joblib.dump(model, Paths.CHURN_MODEL_PKL)

        with open(Paths.THRESHOLD_JSON, "w") as f:
            json.dump({"threshold": ModelConfig.THRESHOLD}, f)

        # log model to MLflow (optional but useful)
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("✅ Model saved:", Paths.CHURN_MODEL_PKL)
    print("✅ Threshold saved:", Paths.THRESHOLD_JSON)
    print("✅ MLflow run logged")


if __name__ == "__main__":
    main()
