"""Tests for src/evaluation (A4)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate
from src.evaluation.metrics import compute_metrics

Y = np.array([0, 1, 0, 1, 1, 0, 1, 1])


def test_perfect_predictor():
    m = compute_metrics(Y, Y.astype(float))
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_majority_baseline():
    m = compute_metrics(Y, np.ones(len(Y)))  # always predicts "up"
    assert m["roc_auc"] == 0.5
    assert m["recall"] == 1.0
    assert m["precision"] == pytest.approx(Y.mean())


def test_single_class_gives_nan_auc():
    m = compute_metrics([1, 1, 1], [0.2, 0.7, 0.9])
    assert np.isnan(m["roc_auc"])


def fake_file(path, model, drop=None):
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=8).strftime("%Y-%m-%d"),
                       "split": ["val"] * 4 + ["test"] * 4, "model": model,
                       "y_true": Y, "y_prob": np.linspace(0.1, 0.9, 8)})
    df["y_pred"] = (df["y_prob"] >= 0.5).astype(int)
    if drop:
        df = df.drop(columns=drop)
    df.to_csv(path / f"{model}.csv", index=False)


def test_missing_column_raises(tmp_path):
    fake_file(tmp_path, "logreg", drop="y_prob")
    with pytest.raises(ValueError, match="y_prob"):
        evaluate.load_all(tmp_path)


def test_any_number_of_models(tmp_path):
    for model in ["logreg", "lstm"]:
        fake_file(tmp_path, model)
    results, table = evaluate.evaluate(evaluate.load_all(tmp_path))
    assert set(results) == {"logreg", "lstm"}
    assert set(results["lstm"]) == {"val", "test"}
    assert len(table) == 4