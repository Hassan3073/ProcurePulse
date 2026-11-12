import contextlib
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_SCRIPTS = ROOT / "data_scripts"
PROCESSED_DIR = ROOT / "data" / "processed"
if str(DATA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPTS))

import clean_data
import feature_engineering

PRICE_COLUMNS = ["gold", "silver", "copper", "platinum", "dxy", "wti", "us10y", "cpi"]
TARGETS = ["gold_5d_ahead", "direction_5d"]
TRAIN_END = "2021-12-31"
HORIZON = 5
MIN_ROWS = 100


def _quiet(func, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


def feature_columns(processed_dir=PROCESSED_DIR):
    header = pd.read_csv(Path(processed_dir) / "features_train.csv", index_col="date", nrows=0)
    return [c for c in header.columns if c not in TARGETS]


def load_builtin_prices(processed_dir=PROCESSED_DIR):
    path = Path(processed_dir) / "cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Restore the frozen data into data/processed/")
    return pd.read_csv(path, index_col="date", parse_dates=True).sort_index()


def read_upload(file):
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise ValueError(f"Could not read the file as CSV: {exc}") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ["date"] + PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}. Expected: date, {', '.join(PRICE_COLUMNS)}")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{int(dates.isna().sum())} rows have a date that is not YYYY-MM-DD")
    prices = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    prices.index = pd.DatetimeIndex(dates, name="date")
    prices = prices.sort_index()
    prices = prices[prices.index.dayofweek <= 4]
    prices = _quiet(clean_data.remove_duplicates, prices)
    prices = _quiet(clean_data.handle_missing, prices)
    prices = _quiet(clean_data.fix_impossible_values, prices)
    prices = _quiet(clean_data.flag_outliers, prices)
    if len(prices) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} trading days after cleaning, got {len(prices)}")
    return prices


def _return_caps(processed_dir):
    unscaled = pd.read_csv(Path(processed_dir) / "features_unscaled.csv", index_col="date", parse_dates=True)
    train = unscaled.loc[:TRAIN_END]
    caps = {}
    for col in train.columns:
        if col.startswith("ret_") or col.startswith("logret_"):
            q1, q3 = train[col].quantile(0.25), train[col].quantile(0.75)
            caps[col] = (q1 - 3.0 * (q3 - q1), q3 + 3.0 * (q3 - q1))
    return caps


def build_features(prices, processed_dir=PROCESSED_DIR):
    prices = prices.sort_index()
    if "is_outlier" not in prices.columns:
        prices = _quiet(clean_data.flag_outliers, prices)
    future = pd.bdate_range(prices.index[-1] + pd.offsets.BDay(1), periods=HORIZON)
    padding = pd.DataFrame([prices.iloc[-1].to_numpy()] * HORIZON, index=future, columns=prices.columns)
    padded = pd.concat([prices, padding])
    padded.index.name = "date"
    feats = _quiet(feature_engineering.build_features, padded)
    feats = feats.loc[feats.index <= prices.index[-1]].drop(columns=TARGETS)
    if feats.empty:
        raise ValueError("Not enough history to build features (need more than 30 trading days)")
    for col, (lo, hi) in _return_caps(processed_dir).items():
        feats[col] = feats[col].clip(lower=lo, upper=hi)
    scaler = joblib.load(Path(processed_dir) / "scaler.joblib")
    scale_cols = list(scaler.feature_names_in_)
    feats[scale_cols] = scaler.transform(feats[scale_cols])
    feats["is_outlier"] = feats["is_outlier"].astype(np.int64)
    return feats[feature_columns(processed_dir)]