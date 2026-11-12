from functools import lru_cache
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "classical"
MODEL_NAMES = ["logreg", "random_forest", "hist_gb"]


@lru_cache(maxsize=None)
def load_model(name, models_dir=MODELS_DIR):
    path = Path(models_dir) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download the weights from the README link into models/classical/")
    return joblib.load(path)


def predict_classical(features, models_dir=MODELS_DIR, names=MODEL_NAMES):
    probs = {}
    for name in names:
        bundle = load_model(name, models_dir)
        row = features[bundle["features"]].iloc[[-1]]
        probs[name] = float(bundle["model"].predict_proba(row)[0, 1])
    return probs