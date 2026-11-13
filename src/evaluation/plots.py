"""Evaluation charts for the report, saved to report/figures/evaluation/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report" / "figures" / "evaluation"


def plot_roc(preds, split="test"):
    fig, ax = plt.subplots(figsize=(6, 6))
    for model, part in preds[preds["split"] == split].groupby("model", sort=False):
        RocCurveDisplay.from_predictions(part["y_true"], part["y_prob"], name=model, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.5)")
    ax.set_title(f"ROC curves on the {split} set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"roc_{split}.png", dpi=150)
    plt.close(fig)


def plot_confusions(preds, split="test"):
    for model, part in preds[preds["split"] == split].groupby("model", sort=False):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            part["y_true"], part["y_pred"], display_labels=["Down", "Up"],
            cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(f"{model}: {split} set")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"confusion_{model}.png", dpi=150)
        plt.close(fig)


def plot_metric_bars(table, split="test"):
    metrics = ["precision", "recall", "f1", "roc_auc"]
    data = table[table["split"] == split].set_index("model")[metrics]
    ax = data.plot.bar(figsize=(8, 4.5), rot=0, width=0.8)
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison on the {split} set")
    ax.legend(ax.containers, ["Precision", "Recall", "F1", "AUC-ROC"], loc="upper right", ncol=4)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"metrics_{split}.png", dpi=150)
    plt.close(fig)


def make_all_plots(preds, table):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc(preds)
    plot_confusions(preds)
    plot_metric_bars(table)