"""Feature engineering pipeline for ProcurePulse.

Generates autoregressive lags, rolling statistics, daily returns, cyclical calendar
encodings, and 5-day forecasting targets strictly without lookahead bias.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALL_COLS, LOG_DIR, PRICE_COLS, PROCESSED_DIR

# Set up logging matching clean_data.py
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "cleaning.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct explanatory features and forward forecasting targets from cleaned data.

    Every input feature relies strictly on past or contemporaneous data.
    Targets project 5 trading days into the future.
    """
    df = df.copy()

    # 1. Daily percentage and point returns
    # Compute relative % changes for price columns (gold, silver, copper, platinum, dxy, wti)
    for col in PRICE_COLS:
        df[f"ret_{col}"] = df[col].pct_change()

    # Treasury yield is quoted in percent, so difference is a change in percentage points
    df["ret_us10y"] = df["us10y"].diff()

    # Continuously compounded log return for gold (additive over time, normal distribution properties)
    df["logret_gold"] = np.log(df["gold"]).diff()

    # 2. Gold autoregressive lags
    # 1-day lag (yesterday's close)
    df["gold_lag1"] = df["gold"].shift(1)
    # 5-day lag (1 trading week ago)
    df["gold_lag5"] = df["gold"].shift(5)
    # 20-day lag (1 trading month / ~4 trading weeks ago)
    df["gold_lag20"] = df["gold"].shift(20)

    # 3. Rolling moving averages and volatility
    # Short-term 7-day moving average
    df["gold_ma7"] = df["gold"].rolling(window=7).mean()
    # Medium-term 30-day moving average baseline
    df["gold_ma30"] = df["gold"].rolling(window=30).mean()
    # 30-day rolling annualized-scale return standard deviation (volatility proxy)
    df["gold_vol30"] = df["ret_gold"].rolling(window=30).std()

    # 4. Cyclical calendar encodings
    # Assert all observations are regular trading weekdays (Monday=0 to Friday=4)
    assert (df.index.dayofweek <= 4).all(), "Found non-weekday dates (Saturday or Sunday) in trading index!"

    # Month encoding (period 12): maps Dec (12) and Jan (1) as smooth continuous neighbors
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Day-of-week encoding (period 5 for 5 trading days): smooth representation across the work week
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)

    # 5. Future forecasting targets (strictly forward-looking labels, never input features)
    # Target 1: Gold price 5 trading days (~1 calendar week) ahead
    df["gold_5d_ahead"] = df["gold"].shift(-5)
    # Target 2: Binary direction indicator (1 if future price > current price, else 0)
    df["direction_5d"] = (df["gold_5d_ahead"] > df["gold"]).astype(int)

    # 6. Outlier column
    # Existing 'is_outlier' column is preserved untouched

    # 7. Boundary NaN handling
    before_drop = len(df)
    # Drop rows containing NaNs introduced by rolling windows/lags (head) and future targets (tail)
    df = df.dropna()
    after_drop = len(df)
    dropped_count = before_drop - after_drop

    # Assert that no missing values remain in the target or any generated feature
    assert not df["gold_5d_ahead"].isna().any(), "Target gold_5d_ahead still contains NaN after dropna!"
    assert df.isna().sum().sum() == 0, f"Features dataframe still contains {df.isna().sum().sum()} NaNs!"

    msg_drop = (
        f"features: dropped {dropped_count} rows ({before_drop} -> {after_drop}): "
        f"30 leading rows dropped due to 30-day rolling windows/lags, "
        f"5 trailing rows dropped because they have no 5-day-ahead price target."
    )
    logging.info(msg_drop)
    print(msg_drop)

    return df


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for ProcurePulse.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "cleaned.csv",
        help="Path to input cleaned CSV file (default: data/processed/cleaned.csv)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "features_full.csv",
        help="Path to save full engineered features (default: data/processed/features_full.csv)"
    )
    args = parser.parse_args()

    # Load cleaned dataset
    if not args.input.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {args.input}. Run clean_data.py first.")

    df_clean = pd.read_csv(args.input, index_col="date", parse_dates=True)
    msg_load = f"features: loaded {args.input.name} with {len(df_clean)} rows, {df_clean.index.min():%Y-%m-%d} to {df_clean.index.max():%Y-%m-%d}"
    logging.info(msg_load)
    print(msg_load)

    # Build features
    features_df = build_features(df_clean)

    # Save output for verification and testing
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index_label="date")

    # Metrics and diagnostics
    direction_share = (features_df["direction_5d"] == 1).mean()
    col_list = list(features_df.columns)

    msg_summary = (
        f"features: engineered dataset saved to {args.output.name}. "
        f"Shape: {features_df.shape}, Date range: {features_df.index.min():%Y-%m-%d} to {features_df.index.max():%Y-%m-%d}, "
        f"Share of direction_5d == 1: {direction_share:.4f} ({direction_share * 100:.2f}%). "
        f"Columns ({len(col_list)}): {col_list}"
    )
    logging.info(msg_summary)
    print("\n--- Feature Engineering Summary ---")
    print(f"Shape: {features_df.shape}")
    print(f"Date range: {features_df.index.min():%Y-%m-%d} to {features_df.index.max():%Y-%m-%d}")
    print(f"Share of direction_5d == 1 (Up): {direction_share:.4f} ({direction_share * 100:.2f}%)")
    print(f"Total columns ({len(col_list)}):")
    for i, c in enumerate(col_list, 1):
        print(f"  {i:2d}. {c}")


if __name__ == "__main__":
    main()
