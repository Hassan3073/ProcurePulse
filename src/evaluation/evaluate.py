"""A4: score every prediction file in results/predictions/ and write the comparison tables.

Run from the repository root:  python -m src.evaluation.evaluate
"""
import json
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import make_all_plots

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "results" / "predictions"
METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"
REQUIRED = ["date", "split", "model", "y_true", "y_prob", "y_pred"]
SPLITS = ["train", "val", "test"]


def read_predictions(path):
    """Read one prediction file and check it follows the agreed format (5.3)."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing column(s) {missing}; expected {REQUIRED}")
    bad = set(df["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"{Path(path).name} has unknown split value(s) {sorted(bad)}")
    return df


def load_all(pred_dir=PRED_DIR):
    """Read every results/predictions/*.csv into one table."""
    files = sorted(Path(pred_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {pred_dir}")
    return pd.concat([read_predictions(f) for f in files], ignore_index=True)


def evaluate(preds):
    """Return ({model: {split: metrics}}, comparison table) for all models and splits."""
    results, rows = {}, []
    for model, model_df in preds.groupby("model", sort=False):
        results[model] = {}
        for split in SPLITS:
            part = model_df[model_df["split"] == split]
            if part.empty:
                continue
            m = compute_metrics(part["y_true"], part["y_prob"])
            results[model][split] = m
            rows.append({"model": model, "split": split, **m})
    table = pd.DataFrame(rows)
    return results, table


def main():
    preds = load_all()
    results, table = evaluate(preds)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    table.round(4).to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    make_all_plots(preds, table)

    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    print(f"Evaluated {len(results)} models: {', '.join(results)}")
    print("\nTest set:")
    print(table[table["split"] == "test"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()