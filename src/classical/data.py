"""Load the Deliverable 1 feature files for the scikit-learn models."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models" / "classical"

TARGETS = ["gold_5d_ahead", "direction_5d"]
LEVEL_COLS = ["gold", "silver", "copper", "platinum", "dxy", "wti", "us10y", "cpi",
              "gold_lag1", "gold_lag5", "gold_lag20", "gold_ma7", "gold_ma30"]
FEATURE_SETS = ["all", "stationary"]
HORIZON = 5  # the label looks 5 trading days ahead
SEED = 42


def feature_columns(df, feature_set="all"):
    """Return the model input columns: 'all' (27) or 'stationary' (14, no price levels)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    cols = [c for c in df.columns if c not in TARGETS]
    if feature_set == "stationary":
        cols = [c for c in cols if c not in LEVEL_COLS]
    return cols


def load_frame(split):
    """Read features_<split>.csv with the date as index."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
    path = PROCESSED_DIR / f"features_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python main.py --from 5")
    return pd.read_csv(path, index_col="date", parse_dates=True)


def load_split(split, feature_set="all", purge=True):
    """Return (X, y) for 'train', 'val' or 'test'.

    purge=True drops the last 5 rows of train and val: their labels look into the
    next period, so using them to fit a model would leak future prices.
    """
    df = load_frame(split)
    if purge and split in ("train", "val"):
        df = df.iloc[:-HORIZON]
    X = df[feature_columns(df, feature_set)]
    y = df["direction_5d"].astype(int)
    return X, y