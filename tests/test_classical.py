"""Tests for src/classical (A2). They use small fake feature files, not the real data."""
import numpy as np
import pandas as pd
import pytest

from src.classical import data
from src.classical.models import MODEL_NAMES, get_model, get_models

OTHER_COLS = ["is_outlier", "ret_gold", "ret_silver", "ret_copper", "ret_platinum",
              "ret_dxy", "ret_wti", "ret_us10y", "logret_gold", "gold_vol30",
              "month_sin", "month_cos", "dow_sin", "dow_cos"]


@pytest.fixture
def fake_processed(tmp_path, monkeypatch):
    """Write fake features_<split>.csv files shaped like the real ones."""
    rng = np.random.default_rng(0)
    cols = data.LEVEL_COLS + OTHER_COLS
    for split, n in [("train", 60), ("val", 30), ("test", 30)]:
        df = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols,
                          index=pd.bdate_range("2020-01-01", periods=n, name="date"))
        df["gold_5d_ahead"] = rng.normal(size=n)
        df["direction_5d"] = rng.integers(0, 2, size=n)
        df.to_csv(tmp_path / f"features_{split}.csv")
    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path)
    return tmp_path


def test_targets_never_in_features(fake_processed):
    for split in ["train", "val", "test"]:
        for feature_set in data.FEATURE_SETS:
            X, y = data.load_split(split, feature_set)
            assert "gold_5d_ahead" not in X.columns
            assert "direction_5d" not in X.columns


def test_feature_set_sizes(fake_processed):
    X_all, _ = data.load_split("train", "all")
    X_stat, _ = data.load_split("train", "stationary")
    assert X_all.shape[1] == 27
    assert X_stat.shape[1] == 14
    assert not set(data.LEVEL_COLS) & set(X_stat.columns)


def test_purge_removes_five_rows(fake_processed):
    X_full, _ = data.load_split("train", purge=False)
    X_purged, y_purged = data.load_split("train", purge=True)
    assert len(X_full) - len(X_purged) == 5
    assert len(X_purged) == len(y_purged)
    X_test_full, _ = data.load_split("test", purge=False)
    X_test, _ = data.load_split("test", purge=True)
    assert len(X_test) == len(X_test_full)  # the test split is never purged


def test_labels_are_binary(fake_processed):
    _, y = data.load_split("train")
    assert set(y.unique()) <= {0, 1}


def test_bad_arguments(fake_processed):
    with pytest.raises(ValueError):
        data.load_split("train", "prices")
    with pytest.raises(ValueError):
        data.load_split("validation")


def test_every_model_fits_and_predicts(fake_processed):
    X, y = data.load_split("train")
    models = get_models()
    assert list(models) == MODEL_NAMES
    for name, (model, grid) in models.items():
        model.fit(X, y)
        prob = model.predict_proba(X)[:, 1]
        assert prob.shape == (len(X),)
        assert ((prob >= 0) & (prob <= 1)).all()
        for param in grid:
            assert param in model.get_params(), f"{name}: unknown parameter {param}"


def test_unknown_model():
    with pytest.raises(ValueError):
        get_model("xgboost")