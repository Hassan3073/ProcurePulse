"""Read-only, single-window LSTM inference for B1."""
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "models" / "deep" / "lstm_config.json"
WEIGHTS_PATH = ROOT / "models" / "deep" / "lstm_best.pt"
# Ordered input contract from the frozen training feature schema.
FEATURE_COLUMNS = (
    "gold", "silver", "copper", "platinum", "dxy", "wti", "us10y", "cpi",
    "is_outlier", "ret_gold", "ret_silver", "ret_copper", "ret_platinum",
    "ret_dxy", "ret_wti", "ret_us10y", "logret_gold", "gold_lag1",
    "gold_lag5", "gold_lag20", "gold_ma7", "gold_ma30", "gold_vol30",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
)
_cache = None


def clear_cache():
    """Discard the loaded model and config."""
    global _cache
    _cache = None


def _load(columns):
    """Lazily load the CPU model and its configured feature selection."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        import torch
        from src.deep.dataset import LEVEL_COLS
        from src.deep.model import LSTMClassifier
    except ImportError as exc:
        raise RuntimeError(f"LSTM unavailable: {exc}") from exc

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["feature_set"] not in ("all", "stationary"):
        raise ValueError("unknown feature_set")
    window = config["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    selected = [c for c in columns
                if config["feature_set"] != "stationary" or c not in LEVEL_COLS]
    n_features = len(selected)
    if "n_features" in config and config["n_features"] != n_features:
        raise ValueError("feature count does not match config n_features")
    # shortcut: CPU only, single window, no batching
    model = LSTMClassifier(n_features, hidden_size=config["hidden"],
                           num_layers=config["layers"], dropout=config["dropout"]).cpu()
    try:
        state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _cache = (torch, model, config, selected)
    return _cache


def predict_lstm(features):
    """Return the last day's probability of gold rising in five trading days."""
    try:
        import pandas as pd

        if not isinstance(features, pd.DataFrame):
            raise ValueError("features must be a pandas DataFrame")
        if list(features.columns) != list(FEATURE_COLUMNS):
            raise ValueError("required feature columns are missing or out of order")
        if (not isinstance(features.index, pd.DatetimeIndex)
                or features.index.name != "date" or features.index.hasnans
                or not features.index.is_monotonic_increasing):
            raise ValueError("features require a sorted DatetimeIndex named date")
        torch, model, config, selected = _load(features.columns)
        window = config["window"]
        if len(features) < window:
            raise ValueError(f"need at least {window} rows")
        recent = features.iloc[-window:]
        if recent.isna().to_numpy().any():
            raise ValueError("NaN in the last window")
        values = recent[selected].to_numpy(dtype="float32")
        tensor = torch.from_numpy(values).unsqueeze(0)
        if not torch.isfinite(tensor).all().item():
            raise ValueError("non-finite values in the last window")
        with torch.no_grad():
            probability = float(torch.sigmoid(model(tensor)).item())
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("model returned an invalid probability")
        return {"lstm": probability}
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("LSTM unavailable: "):
            raise
        raise RuntimeError(f"LSTM unavailable: {exc}") from exc
