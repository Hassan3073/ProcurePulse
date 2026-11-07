"""Plot LSTM learning curves and its configured architecture without opening windows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.deep.dataset import get_feature_columns
from src.deep.train_lstm import CONFIG_PATH, DEFAULTS, HISTORY_COLUMNS, HISTORY_PATH, ROOT, resolve_params

OUT_DIR = ROOT / "report" / "figures" / "deep"


def _read_history(path):
    if not path.is_file():
        raise ValueError(f"History file does not exist: {path}")
    history = pd.read_csv(path)
    missing = set(HISTORY_COLUMNS) - set(history.columns)
    if missing or history.empty:
        raise ValueError(f"History must contain nonempty columns: {HISTORY_COLUMNS}")
    history = history[HISTORY_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(history.to_numpy()).all():
        raise ValueError("History values must all be finite")
    if (history["epoch"] < 1).any() or (history["epoch"] % 1 != 0).any() or history["epoch"].duplicated().any():
        raise ValueError("History epochs must be unique positive integers")
    return history.sort_values("epoch").reset_index(drop=True)


def _read_architecture(path):
    if path.is_file():
        config = json.loads(path.read_text(encoding="utf-8"))
        params = resolve_params({}, str(path))
        n_features = config.get("n_features")
        if n_features is None:
            n_features = len(get_feature_columns(params["feature_set"]))
    else:
        print(f"Note: config missing at {path}; using DEFAULTS and the 'all' feature count.")
        params, config = DEFAULTS.copy(), {}
        n_features = len(get_feature_columns("all"))
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
        raise ValueError("Config n_features must be a positive integer")
    return {**params, "n_features": n_features}, config


def plot_training(history_path=HISTORY_PATH, config_path=CONFIG_PATH, out_dir=OUT_DIR):
    """Write loss, accuracy, and architecture PNGs; importing this module writes nothing."""
    history_path, config_path, out_dir = Path(history_path), Path(config_path), Path(out_dir)
    history = _read_history(history_path)
    architecture, config = _read_architecture(config_path)
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    if "best_epoch" in config and config["best_epoch"] != best_epoch:
        print(f"Warning: config best_epoch={config['best_epoch']} differs from history best_epoch={best_epoch}; "
              "using the history minimum for both curves.")

    # Import plotting libraries lazily so importing this module creates no font cache.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.ticker import MaxNLocator

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric, ylabel, title in (
        ("loss", "Class-weighted BCE loss (eval mode)", "LSTM training and validation loss"),
        ("acc", "Accuracy", "LSTM training and validation accuracy"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(history["epoch"], history[f"train_{metric}"], color="#285E8E", linewidth=2,
                label="Train (eval mode)")
        ax.plot(history["epoch"], history[f"val_{metric}"], color="#C65E28", linewidth=2,
                label="Validation")
        ax.axvline(best_epoch, color="#586477", linestyle="--", linewidth=1.5,
                   label=f"best epoch = {best_epoch}")
        if metric == "acc":
            ax.axhline(0.5, color="#777777", linestyle=":", label="0.5 reference")
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=15, pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
        fig.tight_layout()
        filename = "lstm_loss_curve.png" if metric == "loss" else "lstm_accuracy_curve.png"
        path = out_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    boxes = [(f"Input  L x F\n(L={architecture['window']}, F={architecture['n_features']})", "#E9F0F7")]
    boxes.extend((f"LSTM layer {layer}\n(hidden={architecture['hidden']})", "#DDEBFA")
                 for layer in range(1, architecture["layers"] + 1))
    boxes.extend([
        (f"Dropout\n(p={architecture['dropout']})", "#EEE5F6"),
        ("Dense 32\n+ ReLU", "#E1F0E9"),
        ("Dense 1\n(logit)", "#E1F0E9"),
        ("Sigmoid ->\nP(up in 5 days)", "#FFF0D6"),
    ])
    fig, ax = plt.subplots(figsize=(2.4 * len(boxes) + 0.6, 4))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.95, bottom=0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.91, "LSTM architecture", ha="center", va="center", fontsize=18, weight="bold")
    gap = 0.018
    width = (0.96 - gap * (len(boxes) - 1)) / len(boxes)
    y, height = 0.36, 0.34
    for index, (label, color) in enumerate(boxes):
        x = 0.02 + index * (width + gap)
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                    boxstyle="round,pad=0.003,rounding_size=0.015",
                                    facecolor=color, edgecolor="#718096", linewidth=1.3))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)
        if index < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.003, y + height / 2),
                                         (x + width + gap - 0.003, y + height / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#526174", linewidth=1.5))
    fig.text(0.5, 0.1,
             "Loss: class-weighted BCEWithLogits | Optimizer: Adam (weight decay 1e-4) | grad-clip 1.0",
             ha="center", va="center", fontsize=10, color="#4A5568")
    path = out_dir / "lstm_architecture.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)
    for path in written:
        print(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    plot_training(args.history, args.config, args.out_dir)


if __name__ == "__main__":
    main()
