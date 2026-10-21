"""Data cleaning pipeline for ProcurePulse.

Cleans raw merged commodity prices and external signals by removing duplicates
and resolving missing values with no lookahead bias.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from config import ALL_COLS, LOG_DIR, METAL_COLS, RAW_DIR

# Set up file logger for cleaning audit trail
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "cleaning.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def load(path: Path) -> pd.DataFrame:
    """Load the raw merged CSV file and parse date index."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    # Parse date column as datetime index for time-series operations
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    # Sort index chronologically to guarantee no temporal ordering issues
    df = df.sort_index()

    msg = f"Loaded {path.name}: {len(df)} rows, {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
    logging.info(msg)
    print(msg)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and remove duplicate dates, keeping the last observation per team requirement."""
    # Find boolean mask of duplicated dates
    dup_mask = df.index.duplicated(keep="last")
    dup_count = int(dup_mask.sum())

    before_rows = len(df)
    # Filter out duplicate rows keeping the last record
    cleaned_df = df[~dup_mask]
    after_rows = len(cleaned_df)

    log_msg = f"Duplicates check: {dup_count} duplicate dates found. Rows: {before_rows} -> {after_rows}"
    logging.info(log_msg)
    print(f"remove_duplicates: {dup_count} duplicates removed (rows {before_rows} -> {after_rows})")

    return cleaned_df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values sequentially without lookahead bias.
    
    Order of execution:
      1. Forward-fill CPI (monthly release published on the 1st, carried forward).
      2. Drop non-trading days where all metals are NaN (weekends, holidays).
      3. Interpolate short gaps (<= 3 days) using time-weighted linear interpolation.
      4. Drop remaining NaNs (series boundaries or gaps exceeding 3 days).
    """
    df = df.copy()

    # Step 1: Forward-fill CPI
    # CPI is published monthly on the 1st; we propagate it forward to trading days
    r1_before, m1_before = len(df), df.isna().sum().to_dict()
    df["cpi"] = df["cpi"].ffill()
    r1_after, m1_after = len(df), df.isna().sum().to_dict()
    msg_step1 = f"Step 1 Forward-fill CPI: rows {r1_before} -> {r1_after}, missing per column {m1_before} -> {m1_after}"
    logging.info(msg_step1)
    print(f"Step 1 (Forward-fill CPI): cpi missing {m1_before['cpi']} -> {m1_after['cpi']}")

    # Step 2: Drop non-trading days where all metal prices are NaN
    # If gold, silver, copper, and platinum are all missing, the exchanges were closed
    r2_before, m2_before = len(df), df.isna().sum().to_dict()
    df = df.dropna(subset=METAL_COLS, how="all")
    r2_after, m2_after = len(df), df.isna().sum().to_dict()
    msg_step2 = f"Step 2 Drop non-trading days (all metals NaN): rows {r2_before} -> {r2_after}, missing per column {m2_before} -> {m2_after}"
    logging.info(msg_step2)
    print(f"Step 2 (Drop non-trading days): rows {r2_before} -> {r2_after}")

    # Step 3: Time-based interpolation for short internal gaps
    # limit_area='inside' ensures we never extrapolate beyond observed values
    r3_before, m3_before = len(df), df.isna().sum().to_dict()
    df = df.interpolate(method="time", limit=3, limit_area="inside")
    r3_after, m3_after = len(df), df.isna().sum().to_dict()
    msg_step3 = f"Step 3 Time-based interpolation (limit=3): rows {r3_before} -> {r3_after}, missing per column {m3_before} -> {m3_after}"
    logging.info(msg_step3)
    print(f"Step 3 (Interpolate short gaps): total missing {sum(m3_before.values())} -> {sum(m3_after.values())}")

    # Step 4: Drop any remaining NaNs
    # Removes leading NaNs before series start or gaps wider than 3 days
    r4_before, m4_before = len(df), df.isna().sum().to_dict()
    df = df.dropna()
    r4_after, m4_after = len(df), df.isna().sum().to_dict()
    msg_step4 = f"Step 4 Drop remaining NaNs: rows {r4_before} -> {r4_after}, missing per column {m4_before} -> {m4_after}"
    logging.info(msg_step4)
    print(f"Step 4 (Drop remaining NaNs): rows {r4_before} -> {r4_after} (removed {r4_before - r4_after} rows)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Clean raw merged commodity data (first half).")
    # Allow user or automated scripts to provide a custom input CSV path
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "merged_raw.csv",
        help="Path to input raw CSV file (default: data/raw/merged_raw.csv)"
    )
    args = parser.parse_args()

    # Pipeline execution
    df = load(args.input)
    df = remove_duplicates(df)
    df = handle_missing(df)

    # Verification checks: ensure clean contract compliance
    assert df.isna().sum().sum() == 0, f"Error: Dataset still contains {df.isna().sum().sum()} NaNs!"
    assert not df.index.duplicated().any(), "Error: Dataset still contains duplicate dates!"

    summary = (
        f"Cleaning successful: {len(df)} clean rows x {len(df.columns)} columns, "
        f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}, 0 NaNs remaining."
    )
    logging.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
