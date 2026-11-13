import numpy as np
import pandas as pd
import pytest

from src.inference import live

TODAY = "2026-11-12"


def fake_yahoo(ticker, start):
    dates = pd.bdate_range(start, TODAY)
    rng = np.random.default_rng(len(ticker))
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)


def fake_backup(name, start):
    if name == "cpi":
        return pd.Series(320.0, index=pd.date_range(start, TODAY, freq="MS")), "BLS"
    return fake_yahoo(name, start) / 25, "Yahoo"


def test_download_latest_is_cleaned_and_stops_before_today(monkeypatch):
    monkeypatch.setattr(live.fetch_metals, "fetch_yahoo", fake_yahoo)
    monkeypatch.setattr(live.fetch_external, "fetch_backup", fake_backup)
    prices = live.download_latest(today=TODAY)
    assert list(prices.columns) == live.PRICE_COLUMNS + ["is_outlier"]
    assert prices.index[-1] < pd.Timestamp(TODAY)
    assert len(prices) > 200 and prices.notna().all().all()


def test_download_error_is_raised(monkeypatch):
    def broken(ticker, start):
        raise RuntimeError("Could not download GC=F after 3 attempts")
    monkeypatch.setattr(live.fetch_metals, "fetch_yahoo", broken)
    with pytest.raises(RuntimeError, match="Could not download"):
        live.download_latest(today=TODAY)