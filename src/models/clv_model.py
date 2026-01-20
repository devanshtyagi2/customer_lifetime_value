import json
import joblib
import pandas as pd

from src.utils.config import Paths


def segment_clv(final_clv: pd.Series) -> pd.Series:
    # Stable bins to avoid qcut issues (many zeros)
    maxv = float(final_clv.max()) if len(final_clv) else 0.0
    top = max(maxv, 1000.0)
    return pd.cut(
        final_clv,
        bins=[-1, 100, 1000, top],
        labels=["Low Value", "Mid Value", "High Value"]
    )


def main():
    df = pd.read_csv(Paths.CHURN_READY_CSV)

    model = joblib.load(Paths.CHURN_MODEL_PKL)
    features = json.load(open(Paths.FEATURES_JSON))

    # churn probability
    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    df["churn_probability"] = model.predict_proba(X)[:, 1]

    # ✅ raw_clv proxy from existing column
    if "total_revenue" not in df.columns:
        raise ValueError("total_revenue column missing. Cannot compute raw_clv.")

    df["raw_clv"] = pd.to_numeric(df["total_revenue"], errors="coerce").fillna(0)

    # risk-adjusted CLV
    df["final_clv"] = df["raw_clv"] * (1 - df["churn_probability"])

    # segments
    df["clv_segment"] = segment_clv(df["final_clv"])

    df.to_csv(Paths.FINAL_OUTPUT_CSV, index=False)
    print("✅ Saved:", Paths.FINAL_OUTPUT_CSV)


if __name__ == "__main__":
    main()
