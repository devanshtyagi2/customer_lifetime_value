import pandas as pd
from src.utils.config import Paths

def load_churn_ready() -> pd.DataFrame:
    return pd.read_csv(Paths.CHURN_READY_CSV)

if __name__ == "__main__":
    df = load_churn_ready()
    print(df.shape)
    print(df.columns)
    print(df.head(3))
