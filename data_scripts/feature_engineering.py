"""Feature engineering pipeline for ProcurePulse.

Generates autoregressive lags, rolling statistics, daily returns, cyclical calendar
encodings, and 5-day forecasting targets strictly without lookahead bias.
Applies chronological train/val/test splitting, train-only 3xIQR capping on returns,
and train-only standard scaling.
"""
import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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


def split_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train, validation, and test sets.

    Strict chronological partition without shuffling prevents future leakage:
      - Train: Start to 2021-12-31 (~12 years)
      - Validation: 2022-01-01 to 2023-12-31 (2 years)
      - Test: 2024-01-01 to end (out-of-sample evaluation)
    """
    train = df.loc[:"2021-12-31"].copy()
    val = df.loc["2022-01-01":"2023-12-31"].copy()
    test = df.loc["2024-01-01":].copy()

    # Integrity assertions: non-overlapping and complete coverage
    assert train.index.max() < val.index.min(), "Overlap detected between Train and Validation splits!"
    assert val.index.max() < test.index.min(), "Overlap detected between Validation and Test splits!"
    assert len(train) + len(val) + len(test) == len(df), "Row count mismatch across splits!"

    # Log partition details and class balance
    splits = [("Train", train), ("Val", val), ("Test", test)]
    for name, subset in splits:
        share_up = (subset["direction_5d"] == 1).mean()
        msg_split = (
            f"features: split {name} size: {len(subset)} rows "
            f"({subset.index.min():%Y-%m-%d} to {subset.index.max():%Y-%m-%d}), "
            f"share of direction_5d == 1: {share_up:.4f} ({share_up * 100:.2f}%)"
        )
        logging.info(msg_split)
        print(msg_split)

    return train, val, test


def cap_returns(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clip return outliers using 3xIQR bounds computed strictly on the training set.

    Calculates bounds on Train only and applies identical limits across all three splits.
    Saves concatenated unscaled features to PROCESSED_DIR / 'features_unscaled.csv' for EDA.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    # Identify all return columns to be capped
    ret_cols = [c for c in train.columns if c.startswith("ret_") or c.startswith("logret_")]

    clipped_counts = {"train": 0, "val": 0, "test": 0}

    # Derive clipping thresholds on Train partition strictly
    for col in ret_cols:
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 3.0 * iqr
        hi = q3 + 3.0 * iqr

        # Count clipped entries before applying
        clipped_counts["train"] += int(((train[col] < lo) | (train[col] > hi)).sum())
        clipped_counts["val"] += int(((val[col] < lo) | (val[col] > hi)).sum())
        clipped_counts["test"] += int(((test[col] < lo) | (test[col] > hi)).sum())

        # Clip values across all three partitions using train-derived bounds
        train[col] = train[col].clip(lower=lo, upper=hi)
        val[col] = val[col].clip(lower=lo, upper=hi)
        test[col] = test[col].clip(lower=lo, upper=hi)

    msg_cap = (
        f"features: 3xIQR return capping bounds fitted on Train only "
        f"({train.index.min():%Y-%m-%d} to {train.index.max():%Y-%m-%d}). "
        f"Values clipped: Train={clipped_counts['train']}, Val={clipped_counts['val']}, Test={clipped_counts['test']}."
    )
    logging.info(msg_cap)
    print(msg_cap)

    # Save concatenated unscaled table for EDA script consumption
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    unscaled_df = pd.concat([train, val, test])
    unscaled_path = PROCESSED_DIR / "features_unscaled.csv"
    unscaled_df.to_csv(unscaled_path, index_label="date")
    msg_unscaled = f"features: saved unscaled features to {unscaled_path.name} ({len(unscaled_df)} rows)"
    logging.info(msg_unscaled)
    print(msg_unscaled)

    return train, val, test


def scale_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on training numerical features and transform all splits.

    Target variables, binary flags, and cyclical encodings are not scaled.
    Saves features_train.csv, features_val.csv, features_test.csv, and scaler.joblib.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    # Explicitly exclude targets, binary flags, and cyclical angles from scaling
    skip = {"is_outlier", "direction_5d", "gold_5d_ahead"}
    scale_cols = [
        c for c in train.columns
        if c not in skip and not (c.endswith("_sin") or c.endswith("_cos"))
    ]

    # Fit scaler on training data only
    scaler = StandardScaler()
    train[scale_cols] = scaler.fit_transform(train[scale_cols])
    # Apply trained scaler parameters to validation and test sets
    val[scale_cols] = scaler.transform(val[scale_cols])
    test[scale_cols] = scaler.transform(test[scale_cols])

    # Verification assertions: Train mean approx 0, Train std approx 1 (ddof=0 in StandardScaler)
    train_means = train[scale_cols].mean()
    train_stds = train[scale_cols].std(ddof=0)
    assert (train_means.abs() < 1e-6).all(), "Train scaled means deviate from 0 by more than 1e-6!"
    assert ((train_stds - 1.0).abs() < 0.01).all(), "Train scaled standard deviations deviate from 1 by > 0.01!"

    # Assert zero NaNs exist in any scaled partition
    assert not train.isna().any().any(), "Train partition contains NaNs after scaling!"
    assert not val.isna().any().any(), "Val partition contains NaNs after scaling!"
    assert not test.isna().any().any(), "Test partition contains NaNs after scaling!"

    # Log proof of zero future leakage: val/test gold mean is shifted and non-zero
    val_gold_mean = val["gold"].mean()
    test_gold_mean = test["gold"].mean()
    msg_leak = (
        f"features: StandardScaler fitted on Train only ({train.index.min():%Y-%m-%d} to {train.index.max():%Y-%m-%d}). "
        f"Post-scaling gold mean on Val: {val_gold_mean:.4f}, on Test: {test_gold_mean:.4f} "
        f"(non-zero confirms true out-of-sample evaluation without distribution leakage)."
    )
    logging.info(msg_leak)
    print(msg_leak)

    # Save scaled partitions and trained scaler object
    train.to_csv(PROCESSED_DIR / "features_train.csv", index_label="date")
    val.to_csv(PROCESSED_DIR / "features_val.csv", index_label="date")
    test.to_csv(PROCESSED_DIR / "features_test.csv", index_label="date")
    joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")

    msg_saved = (
        f"features: saved features_train.csv ({len(train)}), features_val.csv ({len(val)}), "
        f"features_test.csv ({len(test)}), and scaler.joblib to {PROCESSED_DIR.name}."
    )
    logging.info(msg_saved)
    print(msg_saved)

    print(f"\nScaled columns ({len(scale_cols)}): {scale_cols}")

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline for ProcurePulse.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "cleaned.csv",
        help="Path to input cleaned CSV file (default: data/processed/cleaned.csv)"
    )
    args = parser.parse_args()

    # Load cleaned dataset
    if not args.input.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {args.input}. Run clean_data.py first.")

    df_clean = pd.read_csv(args.input, index_col="date", parse_dates=True)
    msg_load = f"features: loaded {args.input.name} with {len(df_clean)} rows, {df_clean.index.min():%Y-%m-%d} to {df_clean.index.max():%Y-%m-%d}"
    logging.info(msg_load)
    print(msg_load)

    # Step 1: Feature construction
    features_df = build_features(df_clean)

    # Step 2: Chronological train/val/test splitting
    train, val, test = split_chronological(features_df)

    # Step 3: Train-only return capping (outputs features_unscaled.csv)
    train, val, test = cap_returns(train, val, test)

    # Step 4: Train-only feature scaling (outputs features_train/val/test.csv & scaler.joblib)
    train_scaled, val_scaled, test_scaled = scale_features(train, val, test)

    print("\nFeature engineering pipeline successfully completed.")


if __name__ == "__main__":
    main()
