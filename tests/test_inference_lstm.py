"""B1 inference checks against the frozen checkpoint and predictions."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.inference import lstm


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch is not installed")
    paths = [lstm.WEIGHTS_PATH, lstm.CONFIG_PATH,
             ROOT / "data/processed/features_test.csv",
             ROOT / "results/predictions/lstm.csv"]
    for path in paths:
        if not path.is_file():
            pytest.skip(f"Required frozen B1 artifact is missing: {path}")
    features = pd.read_csv(paths[2], index_col="date", parse_dates=True)
    features = features.drop(columns=["gold_5d_ahead", "direction_5d"])
    config = json.loads(lstm.CONFIG_PATH.read_text(encoding="utf-8"))
    lstm.clear_cache()
    yield features, config, torch
    lstm.clear_cache()


def test_real_data(artifacts):
    features, _, _ = artifacts
    predictions = pd.read_csv(ROOT / "results/predictions/lstm.csv", parse_dates=["date"])
    expected = predictions.loc[predictions["split"] == "test"].iloc[-1]
    assert features.index[-1] == expected["date"] == pd.Timestamp("2026-10-14")
    result = lstm.predict_lstm(features)
    assert set(result) == {"lstm"}
    assert isinstance(result["lstm"], float)
    assert 0 <= result["lstm"] <= 1
    assert result["lstm"] == pytest.approx(expected["y_prob"], abs=1e-4)


def test_short_history(artifacts):
    features, config, _ = artifacts
    with pytest.raises(RuntimeError, match="^LSTM unavailable: "):
        lstm.predict_lstm(features.iloc[-(config["window"] - 1):])


@pytest.mark.parametrize("column", ["ret_gold", "gold"])
def test_missing_feature(artifacts, column):
    features, _, _ = artifacts
    with pytest.raises(RuntimeError, match="^LSTM unavailable: "):
        lstm.predict_lstm(features.drop(columns=[column]))


def test_cached_weights(artifacts, monkeypatch):
    features, _, torch = artifacts
    original_load = torch.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    first = lstm.predict_lstm(features)
    cached = lstm._cache
    assert lstm.predict_lstm(features) == first
    assert lstm._cache is cached
    assert len(calls) == 1


def test_missing_weights(artifacts, monkeypatch, tmp_path):
    features, _, _ = artifacts
    lstm.clear_cache()
    monkeypatch.setattr(lstm, "WEIGHTS_PATH", tmp_path / "missing.pt")
    with pytest.raises(RuntimeError, match="^LSTM unavailable: "):
        lstm.predict_lstm(features)


def test_nan_in_window(artifacts):
    features, config, _ = artifacts
    features = features.copy()
    features.loc[features.index[-config["window"]], "ret_gold"] = float("nan")
    with pytest.raises(RuntimeError, match="^LSTM unavailable: "):
        lstm.predict_lstm(features)


def test_longer_history(artifacts):
    features, config, _ = artifacts
    full = lstm.predict_lstm(features)["lstm"]
    recent = lstm.predict_lstm(features.iloc[-config["window"]:])["lstm"]
    assert recent == full
