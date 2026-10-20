"""Phase 2 collection: combine all raw series into one table, data/raw/merged_raw.csv.

Gaps are left empty on purpose; filling them is the cleaning step (clean_data.py).
"""
import logging

import pandas as pd

from config import ALL_COLS, LOG_DIR, RAW_DIR, START_DATE

logging.basicConfig(filename=LOG_DIR / "collection.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def load(name):
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: run fetch_metals.py and fetch_external.py first")
    series = pd.read_csv(path, index_col="date", parse_dates=True)[name]
    repeated = series.index.duplicated()
    if repeated.any():
        logging.warning(f"{name}: {repeated.sum()} repeated dates, keeping the first")
    return series[~repeated]


def main():
    merged = pd.concat([load(name) for name in ALL_COLS], axis=1, join="outer", sort=True)
    merged = merged[merged.index >= START_DATE]
    merged.index.name = "date"
    merged.to_csv(RAW_DIR / "merged_raw.csv")

    msg = (f"merged_raw.csv: {merged.shape[0]} rows x {merged.shape[1]} columns, "
           f"{merged.index.min():%Y-%m-%d} to {merged.index.max():%Y-%m-%d}")
    missing = merged.isna().sum().to_dict()
    logging.info(f"{msg}, missing per column: {missing}")
    print(msg)
    print("Missing values per column:")
    print(merged.isna().sum().to_string())


if __name__ == "__main__":
    main()