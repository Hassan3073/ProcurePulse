import io

import pandas as pd

from src.inference.features import PRICE_COLUMNS, read_upload

import fetch_external
import fetch_metals
from config import MARKET_SIGNALS, METALS


def download_latest(days=365, today=None):
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    start = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    series = {name: fetch_metals.fetch_yahoo(ticker, start) for name, ticker in {**METALS, **MARKET_SIGNALS}.items()}
    for name in ("us10y", "cpi"):
        series[name] = fetch_external.fetch_backup(name, start)[0]
    prices = pd.concat(series, axis=1, sort=True)[PRICE_COLUMNS]
    prices["cpi"] = prices["cpi"].ffill()
    prices = prices[prices.index < today]
    prices.index.name = "date"
    return read_upload(io.StringIO(prices.to_csv()))