import os
import json
import pandas as pd

from src.utils.config import Paths, ModelConfig


def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
    required = ["CustomerID", ModelConfig.TARGET, *ModelConfig.FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    out = df[required].copy()

    # numeric conversions
    for c in ModelConfig.FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out[ModelConfig.TARGET] = pd.to_numeric(out[ModelConfig.TARGET], errors="coerce").fillna(0).astype(int)

    return out


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(Paths.MODEL_DIR, exist_ok=True)

    df = pd.read_csv(Paths.CHURN_READY_CSV)
    model_input = build_model_input(df)

    model_input.to_csv(Paths.MODEL_INPUT_CSV, index=False)

    # also store feature list for API/inference
    with open(Paths.FEATURES_JSON, "w") as f:
        json.dump(list(ModelConfig.FEATURES), f)

    print("✅ Saved:", Paths.MODEL_INPUT_CSV)
    print("✅ Saved:", Paths.FEATURES_JSON)


if __name__ == "__main__":
    main()
