"""Dataset tests for the LSTM (B1), using the existing processed splits."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.deep.dataset import (
    HORIZON,
    LEVEL_COLS,
    ORDER,
    PROCESSED_DIR,
    TARGETS,
    get_feature_columns,
    load_sequences,
    make_dataset,
    make_loader,
    make_loaders,
    make_windows,
)


@pytest.fixture(scope="module")
def processed_frames():
    """Skip only tests using this fixture when processed data is unavailable."""
    paths = {split: PROCESSED_DIR / f"features_{split}.csv" for split in ORDER}
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("run `python main.py --skip-download` first")
    return {split: pd.read_csv(path, index_col="date", parse_dates=True)
            for split, path in paths.items()}


@pytest.fixture(params=(20, 60))
def window(request):
    return request.param


@pytest.fixture(params=("all", "stationary"))
def feature_set(request):
    return request.param


@pytest.mark.parametrize("split", ORDER)
def test_sequence_alignment_shapes_and_dtypes(processed_frames, split, window, feature_set):
    frame = processed_frames[split]
    cols = get_feature_columns(feature_set)
    xs, ys, dates = load_sequences(split, window, feature_set)
    offset = window - 1 if split == "train" else 0
    expected_length = len(frame) - offset
    n_features = 27 if feature_set == "all" else 14

    assert xs.shape == (expected_length, window, n_features)
    assert ys.shape == (expected_length,)
    assert xs.dtype == np.float32
    assert ys.dtype == np.float32
    assert set(np.unique(ys)) <= {0, 1}
    assert isinstance(dates, pd.DatetimeIndex)
    assert len(dates) == expected_length
    pd.testing.assert_index_equal(dates, frame.index[offset:])
    np.testing.assert_array_equal(ys, frame["direction_5d"].to_numpy(dtype="float32")[offset:])
    np.testing.assert_array_equal(xs[-1], frame[cols].iloc[-window:].to_numpy(dtype="float32"))
    assert ys[-1] == frame["direction_5d"].iloc[-1]
    assert dates[-1] == frame.index[-1]


def test_targets_and_price_levels_excluded_from_features(processed_frames, feature_set):
    cols = get_feature_columns(feature_set)
    expected = [col for col in processed_frames["train"].columns if col not in TARGETS]
    if feature_set == "stationary":
        expected = [col for col in expected if col not in LEVEL_COLS]
        assert set(cols).isdisjoint(LEVEL_COLS)
    assert isinstance(cols, list)
    assert cols == expected
    assert "gold_5d_ahead" not in cols
    assert "direction_5d" not in cols
    assert "date" not in cols
    assert len(cols) == (27 if feature_set == "all" else 14)


@pytest.mark.parametrize("split,previous", [("val", "train"), ("test", "val")])
def test_context_uses_past_features_and_current_label(
    processed_frames, split, previous, window, feature_set
):
    cols = get_feature_columns(feature_set)
    xs, ys, dates = load_sequences(split, window, feature_set)
    context = processed_frames[previous][cols].iloc[-(window - 1):]
    current = processed_frames[split]

    np.testing.assert_array_equal(xs[0, :-1], context.to_numpy(dtype="float32"))
    np.testing.assert_array_equal(xs[0, -1], current[cols].iloc[0].to_numpy(dtype="float32"))
    assert ys[0] == current["direction_5d"].iloc[0]
    assert dates[0] == current.index[0]


@pytest.mark.parametrize("split", ORDER)
def test_purge_trims_only_train_and_val(processed_frames, split, window, feature_set):
    xs, ys, dates = load_sequences(split, window, feature_set, purge=False)
    purged_xs, purged_ys, purged_dates = load_sequences(split, window, feature_set, purge=True)
    dropped = HORIZON if split in ("train", "val") else 0
    kept = len(xs) - dropped

    assert HORIZON == 5
    assert len(purged_xs) == len(purged_ys) == len(purged_dates) == kept
    np.testing.assert_array_equal(purged_xs, xs[:kept])
    np.testing.assert_array_equal(purged_ys, ys[:kept])
    pd.testing.assert_index_equal(purged_dates, dates[:kept])


@pytest.mark.parametrize("split", ORDER)
def test_make_dataset_matches_sequences(processed_frames, split, window, feature_set):
    xs, ys, expected_dates = load_sequences(split, window, feature_set)
    dataset, dates = make_dataset(split, window, feature_set)
    expected_length = len(processed_frames[split]) - (window - 1 if split == "train" else 0)

    assert isinstance(dataset, torch.utils.data.TensorDataset)
    assert len(dataset) == expected_length
    assert all(tensor.dtype == torch.float32 for tensor in dataset.tensors)
    torch.testing.assert_close(dataset.tensors[0], torch.from_numpy(xs), rtol=0, atol=0)
    torch.testing.assert_close(dataset.tensors[1], torch.from_numpy(ys), rtol=0, atol=0)
    pd.testing.assert_index_equal(dates, expected_dates)


def test_shuffled_loaders_have_reproducible_first_batch(processed_frames, window, feature_set):
    dataset, _ = make_dataset("train", window, feature_set)
    first = make_loader(dataset, batch_size=32, shuffle=True)
    second = make_loader(dataset, batch_size=32, shuffle=True)

    assert isinstance(first, torch.utils.data.DataLoader)
    assert isinstance(first.sampler, torch.utils.data.RandomSampler)
    assert first.num_workers == second.num_workers == 0
    assert first.generator.initial_seed() == second.generator.initial_seed() == 42
    first_batch, second_batch = next(iter(first)), next(iter(second))
    for actual, expected in zip(first_batch, second_batch):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("purge_options", [{}, {"purge_train": False, "purge_val": True}])
def test_make_loaders_structure_batches_and_dates(processed_frames, window, feature_set, purge_options):
    batch_size = 32
    result = make_loaders(window, batch_size, feature_set, **purge_options)
    n_features = 27 if feature_set == "all" else 14
    purge_by_split = {"train": purge_options.get("purge_train", True),
                      "val": purge_options.get("purge_val", False), "test": False}

    assert set(result) == {"train", "val", "test", "dates", "n_features"}
    assert set(result["dates"]) == set(ORDER)
    assert isinstance(result["n_features"], int)
    assert result["n_features"] == n_features
    for split in ORDER:
        loader = result[split]
        offset = window - 1 if split == "train" else 0
        expected_dates = processed_frames[split].index[offset:]
        if purge_by_split[split]:
            expected_dates = expected_dates[:-HORIZON]
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert isinstance(loader.dataset, torch.utils.data.TensorDataset)
        assert loader.num_workers == 0
        assert loader.batch_size == batch_size
        assert not loader.drop_last
        assert len(loader.dataset) == len(expected_dates)
        assert isinstance(result["dates"][split], pd.DatetimeIndex)
        pd.testing.assert_index_equal(result["dates"][split], expected_dates)
        sampler_type = (torch.utils.data.RandomSampler if split == "train"
                        else torch.utils.data.SequentialSampler)
        assert isinstance(loader.sampler, sampler_type)
        batch_xs, batch_ys = next(iter(loader))
        batch_length = min(batch_size, len(expected_dates))
        assert batch_xs.shape == (batch_length, window, n_features)
        assert batch_ys.shape == (batch_length,)
        if split != "train":
            torch.testing.assert_close(batch_xs, loader.dataset.tensors[0][:batch_length])
            torch.testing.assert_close(batch_ys, loader.dataset.tensors[1][:batch_length])


@pytest.mark.parametrize("kwargs,message", [
    ({"split": "validation", "window": 20}, "Unknown split"),
    ({"split": "train", "window": 20, "feature_set": "prices"}, "Unknown feature_set"),
    ({"split": "train", "window": 0}, "window must be at least 1"),
    ({"split": "train", "window": -1}, "window must be at least 1"),
])
def test_invalid_sequence_arguments_raise_value_error(processed_frames, kwargs, message):
    with pytest.raises(ValueError, match=message):
        load_sequences(**kwargs)


def test_unknown_feature_columns_set_raises_value_error(processed_frames):
    with pytest.raises(ValueError, match="Unknown feature_set"):
        get_feature_columns("prices")


@pytest.mark.parametrize("split", ORDER)
def test_window_one_uses_only_current_split(processed_frames, split, feature_set):
    frame = processed_frames[split]
    cols = get_feature_columns(feature_set)
    xs, ys, dates = load_sequences(split, window=1, feature_set=feature_set)

    assert xs.shape == (len(frame), 1, len(cols))
    assert ys.shape == (len(frame),)
    np.testing.assert_array_equal(xs[:, 0], frame[cols].to_numpy(dtype="float32"))
    np.testing.assert_array_equal(ys, frame["direction_5d"].to_numpy(dtype="float32"))
    pd.testing.assert_index_equal(dates, frame.index)


def test_make_windows_pairs_each_window_with_its_last_label(processed_frames, window, feature_set):
    frame = processed_frames["train"]
    X = frame[get_feature_columns(feature_set)].to_numpy()
    y = frame["direction_5d"].to_numpy()
    xs, ys = make_windows(X, y, window)

    assert len(xs) == len(ys) == len(frame) - (window - 1)
    np.testing.assert_array_equal(xs[0], X[:window].astype("float32"))
    np.testing.assert_array_equal(xs[-1], X[-window:].astype("float32"))
    np.testing.assert_array_equal(ys, y[window - 1:].astype("float32"))


# --- model tests (B2) go below ---
