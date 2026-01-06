"""Download & store raw files."""
import pandas as pd
from pathlib import Path
import requests, zipfile, io

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(exist_ok=True)

def fetch_and_save(url: str, fname: str):
    print("Downloading", fname)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    (RAW_DIR / fname).write_bytes(r.content)

if __name__ == "__main__":
    # example: Nepal Rastra Bank macro CSV
    fetch_and_save(
        "https://www.nrb.org.np/export/download?id=135", "nrb_macro.csv"
    )
    print("Ingest done ->", list(RAW_DIR.iter()))
