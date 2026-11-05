"""A3: tune, train and save the four scikit-learn models, then write their predictions.

Run from the repository root:  python -m src.classical.train_classical
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.classical.data import (FEATURE_SETS, HORIZON, MODELS_DIR, RESULTS_DIR, ROOT,
                                SEED, load_split)
from src.classical.models import MODEL_NAMES, get_model

TUNING_DIR = RESULTS_DIR / "tuning"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = ROOT / "report" / "figures" / "classical"


def tune(name, feature_set):
    """Walk-forward grid search on the purged training set. Returns (search, tuning rows)."""
    X, y = load_split("train", feature_set)
    model, grid = get_model(name)
    cv = TimeSeriesSplit(n_splits=5, gap=HORIZON)
    search = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X, y)
    res = search.cv_results_
    rows = []
    for i, params in enumerate(res["params"]):
        rows.append({
            "model": name,
            "feature_set": feature_set,
            "params": json.dumps(params, sort_keys=True),
            "score_name": "cv_roc_auc",
            "mean_score": round(res["mean_test_score"][i], 4),
            "std_score": round(res["std_test_score"][i], 4),
            "selected": 0,
        })
    return search, rows


def val_auc(model, feature_set):
    """AUC on the whole validation split, used to choose between the feature sets."""
    X, y = load_split("val", feature_set, purge=False)
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


def predict_all(name, model, feature_set):
    """Prediction rows for every train, val and test day (format 5.3)."""
    frames = []
    for split in ["train", "val", "test"]:
        X, y = load_split(split, feature_set, purge=False)
        prob = model.predict_proba(X)[:, 1]
        frames.append(pd.DataFrame({
            "date": X.index.strftime("%Y-%m-%d"),
            "split": split,
            "model": name,
            "y_true": y.to_numpy(),
            "y_prob": prob.round(6),
            "y_pred": (prob >= 0.5).astype(int),
        }))
    return pd.concat(frames, ignore_index=True)


def plot_importance(name, model, feature_set):
    """Permutation importance on the validation split: the AUC lost when a feature is shuffled."""
    X, y = load_split("val", feature_set, purge=False)
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10,
                                    random_state=SEED, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=X.columns).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax, color="#2a7ab0")
    ax.set_xlabel("Drop in validation AUC when the feature is shuffled")
    ax.set_title(f"Feature importance: {name} ({feature_set} features)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    for folder in [TUNING_DIR, PRED_DIR, MODELS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], {}

    for name in MODEL_NAMES:
        best = None
        for feature_set in FEATURE_SETS:
            start = time.time()
            search, rows = tune(name, feature_set)
            auc = val_auc(search.best_estimator_, feature_set)
            print(f"{name:14s} {feature_set:10s} cv_auc={search.best_score_:.3f} "
                  f"val_auc={auc:.3f} ({time.time() - start:.0f}s)")
            all_rows += rows
            if best is None or auc > best["val_auc"]:
                best = {"feature_set": feature_set, "val_auc": auc,
                        "params": json.dumps(search.best_params_, sort_keys=True),
                        "model": search.best_estimator_}

        for row in all_rows:
            if (row["model"] == name and row["feature_set"] == best["feature_set"]
                    and row["params"] == best["params"]):
                row["selected"] = 1

        model = best["model"]  # GridSearchCV already refit it on the purged training set
        X, _ = load_split("train", best["feature_set"])
        joblib.dump({"model": model, "feature_set": best["feature_set"],
                     "features": list(X.columns)}, MODELS_DIR / f"{name}.joblib")
        predict_all(name, model, best["feature_set"]).to_csv(PRED_DIR / f"{name}.csv", index=False)
        summary[name] = best
        print(f"  -> chose {best['feature_set']} {best['params']}")

    pd.DataFrame(all_rows).to_csv(TUNING_DIR / "classical_tuning.csv", index=False)

    tree = max(["random_forest", "hist_gb"], key=lambda n: summary[n]["val_auc"])
    plot_importance(tree, summary[tree]["model"], summary[tree]["feature_set"])
    print(f"Saved tuning table, 4 prediction files, 4 models and the {tree} importance chart.")


if __name__ == "__main__":
    main()