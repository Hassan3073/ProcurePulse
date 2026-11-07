"""Sliding-window datasets for the PyTorch LSTM."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


def get_feature_columns(feature_set="all"):
    """Read the training CSV header and return the selected feature names."""
    if feature_set not in ("all", "stationary"):
        raise ValueError(f"Unknown feature_set: {feature_set!r}; expected 'all' or 'stationary'")
    header = pd.read_csv(PROCESSED_DIR / "features_train.csv", index_col="date", nrows=0)
    cols = [c for c in header.columns if c not in TARGETS]
    if feature_set == "stationary":
        cols = [c for c in cols if c not in LEVEL_COLS]
    return cols


def make_dataset(split, window, feature_set="all", purge=False):
    """Return a float32 TensorDataset of windows and labels, plus aligned dates."""
    xs, ys, dates = load_sequences(split, window, feature_set, purge)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
    return dataset, dates


def make_loader(dataset, batch_size, shuffle):
    """Build a single-process DataLoader with reproducible shuffling."""
    generator = torch.Generator().manual_seed(42) if shuffle else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       generator=generator, num_workers=0)


def make_loaders(window, batch_size, feature_set="all", purge_train=True, purge_val=False):
    """Return a dict with train, val, test, dates, and n_features keys.

    train/val/test are DataLoaders; only train is shuffled. dates maps each split
    to its aligned pandas DatetimeIndex, and n_features is the feature count (int).
    Training is purged by default; validation is optional and test is never purged.
    """
    loaders, dates = {}, {}
    purge_by_split = {"train": purge_train, "val": purge_val, "test": False}
    for split in ORDER:
        dataset, dates[split] = make_dataset(split, window, feature_set,
                                            purge=purge_by_split[split])
        loaders[split] = make_loader(dataset, batch_size, shuffle=split == "train")
    return {**loaders, "dates": dates, "n_features": len(get_feature_columns(feature_set))}
