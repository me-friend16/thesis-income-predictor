import pandas as pd, numpy as np
from pathlib import Path

IN  = Path("data/interim/feature_store.parquet")
OUT = Path("data/interim/feature_store_fair.parquet")

def add_attrs(df):
    np.random.seed(42)
    n = len(df)
    df["gender"]   = np.random.choice(["M", "F"], size=n)
    df["age_group"] = pd.cut(np.random.randint(18, 70, n),
                             bins=[0, 30, 45, 100],
                             labels=["young", "mid", "senior"])
    return df

if __name__ == "__main__":
    df = pd.read_parquet(IN)
    df = add_attrs(df)
    df.to_parquet(OUT, index=False)
    print("Added protected attrs ->", OUT)
