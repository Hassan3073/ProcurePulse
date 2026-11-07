"""Sliding-window datasets for the PyTorch LSTM."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TARGETS = ["gold_5d_ahead", "direction_5d"]
LEVEL_COLS = ["gold", "silver", "copper", "platinum", "dxy", "wti", "us10y", "cpi",
              "gold_lag1", "gold_lag5", "gold_lag20", "gold_ma7", "gold_ma30"]
HORIZON, ORDER = 5, ["train", "val", "test"]


def make_windows(X, y, window):
    """Pair each feature window with the label on its last day."""
    xs = np.stack([X[i - window + 1:i + 1] for i in range(window - 1, len(X))])
    return xs.astype("float32"), y[window - 1:].astype("float32")


def load_sequences(split, window, feature_set="all", purge=False):
    """Return (windows [N, window, F], labels [N], dates [N]) for one split."""
    if split not in ORDER:
        raise ValueError(f"Unknown split: {split!r}; expected one of {ORDER}")
    if feature_set not in ("all", "stationary"):
        raise ValueError(f"Unknown feature_set: {feature_set!r}; expected 'all' or 'stationary'")
    if window < 1:
        raise ValueError("window must be at least 1")

    frames = {s: pd.read_csv(PROCESSED_DIR / f"features_{s}.csv", index_col="date",
                             parse_dates=True)
              for s in ORDER}
    df = frames[split]
    cols = [c for c in df.columns if c not in TARGETS]
    if feature_set == "stationary":
        cols = [c for c in cols if c not in LEVEL_COLS]
    X, y = df[cols].to_numpy(), df["direction_5d"].to_numpy()
    dates = pd.DatetimeIndex(df.index)
    i = ORDER.index(split)
    if i > 0 and window > 1:
        # Context contains only past features; previous-split labels are never used.
        context = frames[ORDER[i - 1]][cols].to_numpy()[-(window - 1):]
        X, y = np.vstack([context, X]), np.concatenate([np.zeros(window - 1), y])
    elif i == 0:
        dates = dates[window - 1:]
    xs, ys = make_windows(X, y, window)
    if purge and split in ("train", "val"):
        xs, ys, dates = xs[:-HORIZON], ys[:-HORIZON], dates[:-HORIZON]
    return xs, ys, dates
