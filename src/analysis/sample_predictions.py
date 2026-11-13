"""Plot frozen logistic-regression and LSTM test predictions for the report."""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "report/figures/analysis/sample_predictions.png"
MODELS = {"logreg": "Logistic regression", "lstm": "LSTM"}
GREEN, AMBER, RED, GREY = "#4CAF50", "#D4A017", "#E5534B", "#969BA3"


def load_predictions(path, days):
    """Validate and select the last requested test days without changing files."""
    path = Path(path)
    try:
        frame = pd.read_csv(path)
        required = ["date", "split", "y_true", "y_prob"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        frame = frame.loc[frame["split"].eq("test"), required].copy()
        if frame.empty:
            raise ValueError("no test rows")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.tz_localize(None)
        for column in ("y_true", "y_prob"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame.isna().any().any() or frame["date"].duplicated().any():
            raise ValueError("missing values or duplicate test dates")
        if not frame["y_true"].isin([0, 1]).all() or not frame["y_prob"].between(0, 1).all():
            raise ValueError("y_true must be binary and y_prob must be within [0, 1]")
        return frame.sort_values("date").tail(days)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def plot_predictions(out=OUT_PATH, days=120):
    """Write one deterministic PNG from the saved probabilities."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer")
    frames = {model: load_predictions(ROOT / f"results/predictions/{model}.csv", days)
              for model in MODELS}
    out = Path(out).resolve()
    if out.suffix.lower() != ".png":
        raise ValueError("Output must be a .png file")
    if any(out.is_relative_to(ROOT / name) for name in ("data", "models", "results")):
        raise ValueError("Output cannot be inside data/, models/, or results/")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7), facecolor="white")
    summaries = []
    try:
        for ax, (model, frame) in zip(axes, frames.items()):
            accuracy = frame["y_prob"].ge(0.5).eq(frame["y_true"]).mean()
            ax.plot(frame["date"], frame["y_prob"], color=AMBER, linewidth=1.7)
            ax.axhline(0.5, color=GREY, linestyle="--", linewidth=1)
            for actual, color in ((1, GREEN), (0, RED)):
                rows = frame.loc[frame["y_true"].eq(actual)]
                # Keep the dots at the true 0/1 values, visible across the axes edge.
                ax.scatter(rows["date"], rows["y_true"], color=color, s=17,
                           edgecolors="white", linewidths=0.3, clip_on=False, zorder=3)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel("Chance of up / outcome")
            ax.set_title(f"{MODELS[model]} | Accuracy: {accuracy:.2%} on {len(frame)} test days",
                         loc="left", pad=14, fontsize=12)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(x=0.02)
            summaries.append(f"{MODELS[model]}: {len(frame)} test days, "
                             f"{frame['date'].min():%Y-%m-%d} to {frame['date'].max():%Y-%m-%d}; "
                             f"accuracy={accuracy:.2%}")
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[-1].set_xlabel("Test date")
        fig.text(0.08, 0.035, "Line: predicted chance of up. Dots: what really happened (green = up, red = down).\n"
                 "Above the dashed 50% line the model predicts up.", fontsize=10, linespacing=1.5)
        fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=2.4)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, facecolor="white", metadata={"Software": "ProcurePulse"})
    finally:
        plt.close(fig)
    print(out)
    for summary in summaries:
        print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--days", type=int, default=120)
    args = parser.parse_args()
    try:
        plot_predictions(args.out, args.days)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
