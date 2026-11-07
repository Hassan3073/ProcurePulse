"""B5 tests using tiny hand-made frames and temporary CSVs only."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import error_analysis as analysis

PRED_COLUMNS = ["date", "split", "model", "y_true", "y_prob", "y_pred"]
SEGMENT_COLUMNS = ["model", "split", "segment_type", "segment", "n", "accuracy",
                   "recall_up", "recall_down", "pred_up_rate", "true_up_rate"]
BIAS_COLUMNS = ["model", "split", "true_up_rate", "pred_up_rate", "gap",
                "majority_baseline_acc", "model_acc"]


@pytest.fixture
def tiny_frame():
    predictions = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-02-28", "2020-03-02", "2020-04-30", "2020-05-01", "2021-12-31",
                                "2022-01-03", "2022-12-30", "2023-12-29", "2024-01-02", "2026-06-01"]),
        "split": ["train"] * 5 + ["val"] * 3 + ["test"] * 2,
        "model": "toy",
        "y_true": [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        "y_prob": np.where(np.array(predictions) == 1, 0.8, 0.2),
        "y_pred": predictions,
        "gold_vol30": np.arange(1, 11, dtype=float),
        "is_outlier": [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
    })


@pytest.mark.parametrize("segment_type", ["class", "volatility", "outlier", "year"])
def test_segment_counts_cover_each_model_split(tiny_frame, segment_type):
    second = tiny_frame.iloc[2:].assign(model="second")
    frame = pd.concat([tiny_frame, second], ignore_index=True)
    original = frame.copy(deep=True)
    segmented = analysis.add_segments(frame, analysis.train_vol_cuts(frame))
    metrics = analysis.segment_metrics(segmented)
    expected = frame.groupby(["model", "split"]).size()
    actual = metrics.loc[metrics["segment_type"].eq(segment_type)].groupby(["model", "split"])["n"].sum()
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    pd.testing.assert_frame_equal(frame, original)
    assert metrics.columns.tolist() == SEGMENT_COLUMNS
    assert not metrics[["n", "accuracy"]].isna().any().any()
    pd.testing.assert_frame_equal(metrics, metrics.sort_values(
        ["model", "split", "segment_type", "segment"]).reset_index(drop=True))


def test_always_up_has_zero_down_recall_and_is_flagged(tiny_frame, capsys):
    frame = tiny_frame.assign(y_pred=1, y_prob=1.0)
    segmented = analysis.add_segments(frame, analysis.train_vol_cuts(frame))
    metrics = analysis.segment_metrics(segmented)
    down = metrics.loc[metrics["split"].eq("test") & metrics["segment_type"].eq("class")
                       & metrics["segment"].eq("true_down")].iloc[0]
    assert down["recall_down"] == 0
    analysis.print_summary(segmented, analysis.bias_check(segmented))
    output = capsys.readouterr().out
    assert "toy:" in output
    assert "COLLAPSED to one class" in output


def test_hand_computed_metrics_on_ten_rows(tiny_frame):
    frame = tiny_frame.assign(date=pd.date_range("2021-01-01", periods=10), split="train",
                              y_true=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                              y_pred=[1, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    metrics = analysis.segment_metrics(analysis.add_segments(frame, analysis.train_vol_cuts(frame)))
    year = metrics.loc[metrics["segment_type"].eq("year")].iloc[0]
    assert year["n"] == 10
    assert year["accuracy"] == pytest.approx(0.7)
    assert year["recall_up"] == pytest.approx(4 / 6)
    assert year["recall_down"] == pytest.approx(3 / 4)
    assert year["pred_up_rate"] == pytest.approx(0.5)
    assert year["true_up_rate"] == pytest.approx(0.6)


def test_bias_uses_each_models_train_majority_for_every_split():
    rows = []
    for model, train_labels in (("up_majority", [1, 1, 1, 0]), ("down_majority", [0, 0, 0, 1])):
        for split, labels, predictions in (
            ("train", train_labels, train_labels),
            ("val", [0, 0, 1], [1, 0, 1]),
            ("test", [0, 0, 0, 1], [0, 1, 0, 1]),
        ):
            rows.extend({"model": model, "split": split, "y_true": label, "y_pred": prediction}
                        for label, prediction in zip(labels, predictions))
    result = analysis.bias_check(pd.DataFrame(rows))
    assert result.columns.tolist() == BIAS_COLUMNS
    indexed = result.set_index(["model", "split"])
    for model in ("up_majority", "down_majority"):
        val, test = indexed.loc[(model, "val")], indexed.loc[(model, "test")]
        assert val["gap"] == pytest.approx(1 / 3)
        assert val["model_acc"] == pytest.approx(2 / 3)
        assert test["true_up_rate"] == 0.25
        assert test["pred_up_rate"] == 0.5
        assert test["gap"] == 0.25
        assert test["model_acc"] == 0.75
    assert indexed.loc[("up_majority", "val"), "majority_baseline_acc"] == pytest.approx(1 / 3)
    assert indexed.loc[("up_majority", "test"), "majority_baseline_acc"] == 0.25
    assert indexed.loc[("down_majority", "val"), "majority_baseline_acc"] == pytest.approx(2 / 3)
    assert indexed.loc[("down_majority", "test"), "majority_baseline_acc"] == 0.75


def test_volatility_cuts_ignore_validation_and_test_values(tiny_frame):
    expected = analysis.train_vol_cuts(tiny_frame)
    changed = tiny_frame.copy()
    changed.loc[changed["split"].eq("val"), "gold_vol30"] = -1000000.0
    changed.loc[changed["split"].eq("test"), "gold_vol30"] = 1000000.0
    assert analysis.train_vol_cuts(changed) == expected


def test_volatility_cuts_count_each_train_date_once(tiny_frame):
    shorter_model = tiny_frame.iloc[2:].assign(model="shorter")
    joined = pd.concat([tiny_frame, shorter_model], ignore_index=True)
    expected = tiny_frame.loc[tiny_frame["split"].eq("train"), "gold_vol30"].quantile([1 / 3, 2 / 3])
    np.testing.assert_allclose(analysis.train_vol_cuts(joined), expected.to_numpy())


def test_constant_volatility_still_assigns_every_row(tiny_frame):
    frame = tiny_frame.assign(gold_vol30=2.0)
    cuts = analysis.train_vol_cuts(frame)
    assert cuts == (2.0, 2.0)
    segmented = analysis.add_segments(frame, cuts)
    assert not segmented["volatility"].isna().any()
    assert segmented["volatility"].eq("low").all()
    metrics = analysis.segment_metrics(segmented)
    assert metrics.loc[metrics["segment_type"].eq("volatility"), "n"].sum() == len(frame)


def test_events_obey_dates_splits_and_nonempty_groups(tiny_frame):
    metrics = analysis.segment_metrics(analysis.add_segments(tiny_frame, analysis.train_vol_cuts(tiny_frame)))
    events = metrics.loc[metrics["segment_type"].eq("event")]
    assert dict(zip(events["segment"], events["n"])) == {
        "covid_2020": 2, "rate_rises_2022": 2, "surge_2026": 1,
    }
    assert dict(zip(events["segment"], events["split"])) == {
        "covid_2020": "train", "rate_rises_2022": "val", "surge_2026": "test",
    }
    wrong_splits = tiny_frame.assign(split="test")
    shifted = analysis.add_segments(wrong_splits, analysis.train_vol_cuts(tiny_frame))
    assert set(shifted["event"].dropna()) == {"surge_2026"}


def test_absent_class_recall_is_nan(tiny_frame):
    metrics = analysis.segment_metrics(analysis.add_segments(tiny_frame, analysis.train_vol_cuts(tiny_frame)))
    classes = metrics.loc[metrics["segment_type"].eq("class")]
    assert classes.loc[classes["segment"].eq("true_up"), "recall_down"].isna().all()
    assert classes.loc[classes["segment"].eq("true_down"), "recall_up"].isna().all()
    assert classes["accuracy"].notna().all()


@pytest.mark.parametrize("missing", PRED_COLUMNS)
def test_prediction_missing_column_names_file_and_column(tmp_path, tiny_frame, missing):
    path = tmp_path / "broken.csv"
    tiny_frame[PRED_COLUMNS].drop(columns=missing).to_csv(path, index=False)
    with pytest.raises(ValueError) as caught:
        analysis.load_predictions(tmp_path)
    assert path.name in str(caught.value)
    assert missing in str(caught.value)


@pytest.mark.parametrize("model_count", [1, 2, 5])
def test_load_predictions_supports_any_model_count(tmp_path, tiny_frame, model_count):
    for index in range(model_count):
        tiny_frame[PRED_COLUMNS].assign(model=f"model_{index}").to_csv(tmp_path / f"pred_{index}.csv", index=False)
    result = analysis.load_predictions(tmp_path)
    assert result["model"].nunique() == model_count
    assert len(result) == len(tiny_frame) * model_count
    assert result.columns.tolist() == PRED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


@pytest.mark.parametrize("missing", ["gold_vol30", "is_outlier"])
def test_unscaled_missing_column_names_file_and_column(tmp_path, tiny_frame, missing):
    path = tmp_path / "features.csv"
    tiny_frame[["date", "gold_vol30", "is_outlier"]].drop(columns=missing).to_csv(path, index=False)
    with pytest.raises(ValueError) as caught:
        analysis.join_unscaled(tiny_frame[PRED_COLUMNS], path)
    assert path.name in str(caught.value)
    assert missing in str(caught.value)


def test_inner_join_reports_and_drops_only_unmatched_rows(tmp_path, tiny_frame, capsys):
    path = tmp_path / "features.csv"
    tiny_frame.iloc[:-1][["date", "gold_vol30", "is_outlier"]].to_csv(path, index=False)
    joined = analysis.join_unscaled(tiny_frame[PRED_COLUMNS], path)
    assert len(joined) == len(tiny_frame) - 1
    assert joined["date"].tolist() == tiny_frame["date"].iloc[:-1].tolist()
    assert "toy: unmatched prediction rows=1" in capsys.readouterr().out


def test_train_rows_are_required_for_cuts(tiny_frame):
    with pytest.raises(ValueError, match="Training rows"):
        analysis.train_vol_cuts(tiny_frame.loc[tiny_frame["split"].ne("train")])


def test_cli_end_to_end_uses_only_temporary_inputs(tmp_path, tiny_frame, monkeypatch, capsys):
    pred_dir, out_dir, fig_dir = tmp_path / "preds", tmp_path / "tables", tmp_path / "figs"
    pred_dir.mkdir()
    tiny_frame[PRED_COLUMNS].to_csv(pred_dir / "toy.csv", index=False)
    tiny_frame[PRED_COLUMNS].assign(model="always_up", y_pred=1, y_prob=1.0).to_csv(
        pred_dir / "always_up.csv", index=False)
    unscaled = tmp_path / "features.csv"
    tiny_frame[["date", "gold_vol30", "is_outlier"]].to_csv(unscaled, index=False)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    analysis.main(["--pred-dir", str(pred_dir), "--unscaled", str(unscaled),
                   "--out-dir", str(out_dir), "--fig-dir", str(fig_dir)])
    metrics = pd.read_csv(out_dir / "segment_metrics.csv")
    bias = pd.read_csv(out_dir / "bias_check.csv")
    assert metrics.columns.tolist() == SEGMENT_COLUMNS
    assert bias.columns.tolist() == BIAS_COLUMNS
    assert not metrics[["n", "accuracy"]].isna().any().any()
    assert len(bias) == tiny_frame["split"].nunique() * 2
    for filename in ("accuracy_by_year.png", "accuracy_by_volatility.png", "recall_by_class.png"):
        path = fig_dir / filename
        assert path.is_file() and path.stat().st_size > 0
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    output = capsys.readouterr().out
    assert "Models found (2)" in output
    assert output.count("unmatched prediction rows=0") == 2
    assert "COLLAPSED to one class" in output
