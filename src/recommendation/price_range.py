import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.csv"

METALS = ["gold", "silver", "copper", "platinum"]
UNITS = {"gold": "USD/oz", "silver": "USD/oz", "copper": "USD/lb", "platinum": "USD/oz"}
HORIZON = 5
TRAIN_END = "2021-12-31"
PERCENTILES = (0.10, 0.50, 0.90)


def load_prices(path=CLEANED_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Restore the frozen data into data/processed/")
    df = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    return df[METALS]


def expected_range(prices, train_end=TRAIN_END, horizon=HORIZON):
    prices = prices.dropna()
    hist = prices.loc[:train_end]
    change = (hist.shift(-horizon) / hist - 1).dropna()
    if len(change) < 100:
        raise ValueError(f"Need at least 100 past {horizon}-day changes, got {len(change)}")
    p_low, p_mid, p_high = change.quantile(list(PERCENTILES))
    last_price = float(prices.iloc[-1])
    return {
        "last_date": prices.index[-1].date().isoformat(),
        "last_price": last_price,
        "low": last_price * (1 + p_low),
        "mid": last_price * (1 + p_mid),
        "high": last_price * (1 + p_high),
        "pct_low": float(p_low),
        "pct_mid": float(p_mid),
        "pct_high": float(p_high),
        "n_changes": int(len(change)),
    }