"""The four scikit-learn models and the hyperparameter grid searched for each."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.classical.data import SEED

MODEL_NAMES = ["majority", "logreg", "random_forest", "hist_gb"]


def get_model(name):
    """Return (untrained model, parameter grid) for one model name."""
    if name == "majority":
        model = DummyClassifier(strategy="most_frequent")
        grid = {}
    elif name == "logreg":
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        grid = {"C": [0.01, 0.1, 1, 10]}
    elif name == "random_forest":
        model = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
        grid = {
            "n_estimators": [200, 500],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 5, 20],
        }
    elif name == "hist_gb":
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
        grid = {
            "learning_rate": [0.03, 0.1],
            "max_depth": [3, 5],
            "max_iter": [200, 500],
            "l2_regularization": [0, 1],
        }
    else:
        raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}")
    return model, grid


def get_models():
    """Return {name: (model, grid)} for all four models."""
    return {name: get_model(name) for name in MODEL_NAMES}