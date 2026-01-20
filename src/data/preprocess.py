import pandas as pd
from src.utils.config import Paths

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # basic cleaning
    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].astype(float)

    for col in ["avg_order_value", "total_revenue", "raw_clv", "risk_adjusted_clv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["invoice_count", "total_quantity", "tenure_days", "recency_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

if __name__ == "__main__":
    df = pd.read_csv(Paths.CHURN_READY_CSV)
    df = preprocess(df)
    df.to_csv(Paths.CHURN_READY_CSV, index=False)
    print("✅ Preprocessed and saved:", Paths.CHURN_READY_CSV)
