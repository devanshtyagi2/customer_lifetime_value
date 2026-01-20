from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    CHURN_READY_CSV: str = "data/processed/customer_churn_ready.csv"
    MODEL_INPUT_CSV: str = "data/processed/model_input.csv"
    FINAL_OUTPUT_CSV: str = "data/processed/final_output.csv"

    MODEL_DIR: str = "models"
    CHURN_MODEL_PKL: str = "models/churn_xgb.pkl"
    FEATURES_JSON: str = "models/features.json"
    THRESHOLD_JSON: str = "models/threshold.json"


@dataclass(frozen=True)
class ModelConfig:
    FEATURES: tuple = ("avg_order_value", "invoice_count", "total_quantity", "tenure_days")
    TARGET: str = "churn_flag"
    THRESHOLD: float = 0.30
