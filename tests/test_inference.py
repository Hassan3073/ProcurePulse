import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.features import PRICE_COLUMNS, build_features, load_builtin_prices, read_upload

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
HAS_DATA = (PROCESSED / "features_test.csv").exists() and (PROCESSED / "cleaned.csv").exists()
HAS_MODELS = all((ROOT / "models" / "classical" / f"{m}.joblib").exists() for m in ["logreg", "random_forest", "hist_gb"])
TARGETS = ["gold_5d_ahead", "direction_5d"]


def fake_csv(n=150, drop=None):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2026-01-01", periods=n)
    base = {"gold": 4000, "silver": 50, "copper": 5, "platinum": 1500, "dxy": 100, "wti": 70, "us10y": 4, "cpi": 320}
    df = pd.DataFrame({c: v * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for c, v in base.items()}, index=dates)
    df.index.name = "date"
    if drop:
        df = df.drop(columns=drop)
    return io.StringIO(df.reset_index().to_csv(index=False))


def test_upload_missing_column_raises():
    with pytest.raises(ValueError, match="Missing columns"):
        read_upload(fake_csv(drop=["cpi"]))


def test_upload_too_short_raises():
    with pytest.raises(ValueError, match="at least"):
        read_upload(fake_csv(n=40))


def test_upload_bad_date_raises():
    text = fake_csv().getvalue().replace("2026-01-05", "not-a-date", 1)
    with pytest.raises(ValueError, match="date"):
        read_upload(io.StringIO(text))


def test_upload_is_cleaned():
    prices = read_upload(fake_csv())
    assert list(prices.columns) == PRICE_COLUMNS + ["is_outlier"]
    assert len(prices) == 150 and prices.notna().all().all()


@pytest.mark.skipif(not HAS_DATA, reason="needs data/processed")
def test_builtin_features_match_deliverable1():
    feats = build_features(load_builtin_prices())
    for split in ["val", "test"]:
        ref = pd.read_csv(PROCESSED / f"features_{split}.csv", index_col="date", parse_dates=True).drop(columns=TARGETS)
        assert list(feats.columns) == list(ref.columns)
        assert np.allclose(feats.loc[ref.index].to_numpy(), ref.to_numpy(), atol=1e-9)


@pytest.mark.skipif(not HAS_DATA, reason="needs data/processed")
def test_features_reach_latest_price_date():
    prices = load_builtin_prices()
    feats = build_features(prices)
    assert feats.index[-1] == prices.index[-1]
    assert feats.notna().all().all()


@pytest.mark.skipif(not HAS_DATA, reason="needs data/processed")
def test_uploaded_prices_give_features():
    feats = build_features(read_upload(fake_csv()))
    assert len(feats) > 60 and feats.notna().all().all()


@pytest.mark.skipif(not (HAS_DATA and HAS_MODELS), reason="needs data/processed and models/classical")
def test_classical_matches_deliverable2_predictions():
    from src.inference.classical import predict_classical
    feats = build_features(load_builtin_prices())
    test = pd.read_csv(PROCESSED / "features_test.csv", index_col="date", parse_dates=True)
    probs = predict_classical(feats.loc[:test.index[-1]])
    for name, p in probs.items():
        ref = pd.read_csv(ROOT / "results" / "predictions" / f"{name}.csv")
        ref = ref[ref.split == "test"].iloc[-1]
        assert ref.date == str(test.index[-1].date())
        assert p == pytest.approx(ref.y_prob, abs=1e-4)