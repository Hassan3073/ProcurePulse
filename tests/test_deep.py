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

from src.deep.model import LSTMClassifier, make_loss, make_optimizer


@pytest.fixture
def model_seed():
    """Seed B2 tests without changing the caller's random state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield


@pytest.fixture
def model_n_features(processed_frames, feature_set, model_seed):
    return len(get_feature_columns(feature_set))


def test_model_forward_shape_and_dtype(model_n_features, window):
    model = LSTMClassifier(model_n_features)
    with torch.no_grad():
        logits = model(torch.zeros(8, window, model_n_features))
    assert logits.shape == (8,)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("bias", [-2.0, 2.0])
def test_model_returns_unsquashed_logits(model_n_features, bias):
    model = LSTMClassifier(model_n_features).eval()
    x = torch.randn(8, 20, model_n_features) * 100
    with torch.no_grad():
        # A known output bias makes this an activation check, not a lucky draw.
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(bias)
        logits = model(x)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, torch.full((8,), bias), rtol=0, atol=0)
    assert ((logits < 0) | (logits > 1)).all()
    assert not any(isinstance(layer, torch.nn.Sigmoid) for layer in model.modules())


def test_model_single_layer_disables_lstm_dropout_without_warning(model_n_features):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LSTMClassifier(model_n_features, hidden_size=16, num_layers=1, dropout=0.3)
        with torch.no_grad():
            logits = model(torch.zeros(1, 20, model_n_features))
    assert not caught
    assert model.lstm.num_layers == 1
    assert model.lstm.hidden_size == 16
    assert model.lstm.dropout == 0.0
    assert model.head[0].p == 0.3
    assert logits.shape == (1,)


def test_model_default_architecture(model_n_features):
    model = LSTMClassifier(model_n_features)
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert model.lstm.input_size == model_n_features
    assert model.lstm.num_layers == 2
    assert model.lstm.hidden_size == 64
    assert model.lstm.batch_first is True
    assert model.lstm.dropout == 0.2
    assert isinstance(model.head, torch.nn.Sequential)
    assert [type(layer) for layer in model.head] == [
        torch.nn.Dropout, torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear
    ]
    assert model.head[0].p == 0.2
    assert (model.head[1].in_features, model.head[1].out_features) == (64, 32)
    assert (model.head[3].in_features, model.head[3].out_features) == (32, 1)


def test_model_initialization_is_reproducible(model_n_features):
    torch.manual_seed(42)
    first = LSTMClassifier(model_n_features)
    torch.manual_seed(42)
    second = LSTMClassifier(model_n_features)
    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    assert first_parameters.keys() == second_parameters.keys()
    for name, parameter in first_parameters.items():
        torch.testing.assert_close(parameter, second_parameters[name], rtol=0, atol=0)


def test_model_eval_is_deterministic_and_training_uses_dropout(model_n_features):
    model = LSTMClassifier(model_n_features, hidden_size=16, dropout=0.5)
    x = torch.randn(8, 20, model_n_features)
    with torch.no_grad():
        model.eval()
        eval_first, eval_second = model(x), model(x)
        model.train()
        train_first, train_second = model(x), model(x)
    torch.testing.assert_close(eval_first, eval_second, rtol=0, atol=0)
    assert not torch.equal(train_first, train_second)


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_model_loss_weight_dtype_shape_and_confident_predictions(input_type):
    labels = np.array([0] * 6 + [1] * 4, dtype=np.float64)
    y_train = torch.from_numpy(labels) if input_type == "torch" else labels
    loss = make_loss(y_train)
    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)
    assert loss.pos_weight.shape == (1,)
    assert loss.pos_weight.dtype == torch.float32
    assert loss.pos_weight.item() == pytest.approx(1.5)

    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    correct_logits = (2 * targets - 1) * 8
    right_loss = loss(correct_logits, targets)
    wrong_loss = loss(-correct_logits, targets)
    assert right_loss.shape == wrong_loss.shape == torch.Size([])
    assert right_loss.dtype == wrong_loss.dtype == torch.float32
    assert torch.isfinite(right_loss) and torch.isfinite(wrong_loss)
    assert wrong_loss > right_loss


@pytest.mark.parametrize("input_type", ["numpy", "torch"])
def test_model_loss_handles_all_zero_labels(input_type):
    labels = np.zeros(6, dtype=np.int64)
    y_train = torch.from_numpy(labels) if input_type == "torch" else labels
    loss = make_loss(y_train)
    assert loss.pos_weight.shape == (1,)
    assert loss.pos_weight.dtype == torch.float32
    assert loss.pos_weight.item() == len(labels)
    value = loss(torch.zeros(len(labels)), torch.zeros(len(labels)))
    assert value.shape == torch.Size([])
    assert torch.isfinite(value)


@pytest.mark.parametrize("lr", [0.001, 0.01])
def test_model_optimizer_is_adam_with_requested_settings(model_n_features, lr):
    model = LSTMClassifier(model_n_features)
    optimizer = make_optimizer(model, lr)
    assert isinstance(optimizer, torch.optim.Adam)
    for group in optimizer.param_groups:
        assert group["lr"] == lr
        assert group["weight_decay"] == 1e-4
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for parameter in model.parameters()
    }


def test_model_training_step_and_tiny_batch_overfit(model_n_features):
    model = LSTMClassifier(model_n_features, hidden_size=8, num_layers=1, dropout=0.0)
    targets = torch.tensor([0.0, 1.0] * 4)
    x = torch.randn(len(targets), 5, model_n_features) * 0.05
    x[:, :, 0] = (2 * targets - 1).unsqueeze(1)
    loss = make_loss(targets)
    optimizer = make_optimizer(model, lr=0.03)
    with torch.no_grad():
        initial_loss = loss(model(x), targets).item()

    model.train()
    for _ in range(30):
        optimizer.zero_grad()
        value = loss(model(x), targets)
        assert torch.isfinite(value)
        value.backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(grad_norm)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = loss(model(x), targets).item()
    assert final_loss < initial_loss * 0.5


def test_model_accepts_a_b1_training_batch(processed_frames, model_seed):
    loaders = make_loaders(20, 64)
    xs, ys = next(iter(loaders["train"]))
    n_features = len(get_feature_columns("all"))
    model = LSTMClassifier(n_features).eval()
    with torch.no_grad():
        logits = model(xs)
    assert loaders["n_features"] == n_features
    assert logits.shape == ys.shape == (xs.shape[0],)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("kwargs,message", [
    ({"n_features": 0}, "n_features must be at least 1"),
    ({"n_features": -1}, "n_features must be at least 1"),
    ({"hidden_size": 0}, "hidden_size must be at least 1"),
    ({"hidden_size": -1}, "hidden_size must be at least 1"),
    ({"num_layers": 0}, "num_layers must be at least 1"),
    ({"num_layers": -1}, "num_layers must be at least 1"),
    ({"dropout": -0.1}, "dropout must be in"),
    ({"dropout": 1.0}, "dropout must be in"),
    ({"dropout": 1.1}, "dropout must be in"),
])
def test_model_invalid_arguments_raise_value_error(model_seed, kwargs, message):
    arguments = {"n_features": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        LSTMClassifier(**arguments)


@pytest.mark.parametrize("lr", [0.0, -0.01])
def test_model_invalid_learning_rate_raises_value_error(model_seed, lr):
    model = LSTMClassifier(n_features=1)
    with pytest.raises(ValueError, match="lr must be greater than 0"):
        make_optimizer(model, lr)
