"""Basic cleaning: dtypes, dates, duplicates."""
import pandas as pd
from pathlib import Path

INPATH  = Path("data/raw/nrb_macro.csv")
OUTPATH = Path("data/interim/nrb_macro_clean.parquet")

def clean():
    df = pd.read_csv(INPATH)
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.lower()
    # assume date column named 'date'
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.to_parquet(OUTPATH, index=False)
    print("Cleaned shape:", df.shape, "->", OUTPATH)

if __name__ == "__main__":
    clean()
