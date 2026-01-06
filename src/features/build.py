"""Create calendar + lag features."""
import pandas as pd
import numpy as np
from pathlib import Path

INPATH  = Path("data/interim/nrb_macro_clean.parquet")
OUTPATH = Path("data/interim/feature_store.parquet")
LAGS    = [1, 2, 3, 6, 12]

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]      = df.date.dt.year
    df["month"]     = df.date.dt.month
    df["quarter"]   = df.date.dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    return df

def add_lags(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    for lag in LAGS:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df

def build():
    df = pd.read_parquet(INPATH)
    df = add_calendar(df)
    df = add_lags(df, "cpi")
    df = df.dropna()
    df.to_parquet(OUTPATH, index=False)
    print("Feature-store shape:", df.shape, "->", OUTPATH)

if __name__ == "__main__":
    build()
