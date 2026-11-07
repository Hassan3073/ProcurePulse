"""Analyse frozen prediction CSVs by segment and check prediction/class imbalance."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "results" / "predictions"
UNSCALED_PATH = ROOT / "data" / "processed" / "features_unscaled.csv"
OUT_DIR = ROOT / "results" / "analysis"
FIG_DIR = ROOT / "report" / "figures" / "analysis"
PRED_COLUMNS = ["date", "split", "model", "y_true", "y_prob", "y_pred"]
SEGMENT_COLUMNS = ["model", "split", "segment_type", "segment", "n", "accuracy",
                   "recall_up", "recall_down", "pred_up_rate", "true_up_rate"]
BIAS_COLUMNS = ["model", "split", "true_up_rate", "pred_up_rate", "gap",
                "majority_baseline_acc", "model_acc"]
VOLATILITY_LABELS = ["low", "medium", "high"]
SPLITS = ["train", "val", "test"]


def _require_columns(frame, columns, path):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s): {', '.join(missing)}")


def load_predictions(pred_dir):
    """Read every prediction CSV, preserving its model labels and available rows."""
    paths = sorted(Path(pred_dir).glob("*.csv"))
    if not paths:
        raise ValueError(f"No prediction CSV files found in {pred_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        _require_columns(frame, PRED_COLUMNS, path)
        frame = frame[PRED_COLUMNS].copy()
        if frame.isna().any().any():
            raise ValueError(f"{path}: prediction columns must not contain missing values")
        try:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            for column in ("y_true", "y_prob", "y_pred"):
                frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: invalid date or prediction value: {exc}") from exc
        if not frame["split"].isin(SPLITS).all():
            raise ValueError(f"{path}: split must be train, val, or test")
        if not frame[["y_true", "y_pred"]].isin([0, 1]).all().all():
            raise ValueError(f"{path}: y_true and y_pred must contain only 0 or 1")
        if not frame["y_prob"].between(0, 1).all():
            raise ValueError(f"{path}: y_prob must be within [0, 1]")
        frame["model"] = frame["model"].astype(str)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError(f"No prediction rows found in {pred_dir}")
    if result.duplicated(["model", "split", "date"]).any():
        raise ValueError(f"{pred_dir}: duplicate model/split/date prediction rows")
    return result.sort_values(["model", "split", "date"]).reset_index(drop=True)


def join_unscaled(predictions, unscaled_path):
    """Inner-join unscaled features, reporting unmatched rows for every model."""
    features = pd.read_csv(unscaled_path)
    columns = ["date", "gold_vol30", "is_outlier"]
    _require_columns(features, columns, unscaled_path)
    features = features[columns].copy()
    features["date"] = pd.to_datetime(features["date"], errors="raise")
    if features["date"].isna().any() or features["date"].duplicated().any():
        raise ValueError(f"{unscaled_path}: feature dates must be present and unique")
    for column in ("gold_vol30", "is_outlier"):
        features[column] = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(features[column]).all():
            raise ValueError(f"{unscaled_path}: {column} must contain finite values")
    if not features["is_outlier"].isin([0, 1]).all():
        raise ValueError(f"{unscaled_path}: is_outlier must contain only 0 or 1")
    for model, rows in predictions.groupby("model", sort=True):
        unmatched = int((~rows["date"].isin(features["date"])).sum())
        print(f"{model}: unmatched prediction rows={unmatched}")
    return predictions.merge(features, on="date", how="inner", validate="many_to_one")


def train_vol_cuts(df):
    """Compute shared tercile cuts from unique TRAIN dates, never val/test rows.

    Deduplicating dates prevents model coverage (including shorter LSTM history)
    from weighting some training days more heavily than others.
    """
    rows = df.loc[df["split"].eq("train"), ["date", "gold_vol30"]].drop_duplicates("date")
    if rows.empty:
        raise ValueError("Training rows are required to compute volatility cuts")
    if not np.isfinite(rows["gold_vol30"]).all():
        raise ValueError("Training gold_vol30 values must be finite")
    cuts = rows["gold_vol30"].quantile([1 / 3, 2 / 3])
    return tuple(float(value) for value in cuts)


def add_segments(df, vol_cuts):
    """Return a copy with volatility, class, outlier, year, and event labels."""
    if len(vol_cuts) != 2 or not np.isfinite(vol_cuts).all() or vol_cuts[0] > vol_cuts[1]:
        raise ValueError("vol_cuts must contain two finite, nondecreasing thresholds")
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any() or not np.isfinite(result["gold_vol30"]).all():
        raise ValueError("Segment dates and gold_vol30 must not be missing or non-finite")
    low, high = vol_cuts
    if low == high:
        # The medium interval is empty when training quantiles coincide.
        volatility = pd.cut(result["gold_vol30"], [-np.inf, low, np.inf], labels=["low", "high"])
        result["volatility"] = volatility.cat.set_categories(VOLATILITY_LABELS, ordered=True)
    else:
        result["volatility"] = pd.cut(result["gold_vol30"], [-np.inf, low, high, np.inf],
                                      labels=VOLATILITY_LABELS)
    result["class"] = np.where(result["y_true"].eq(1), "true_up", "true_down")
    result["outlier"] = np.where(result["is_outlier"].eq(1), "outlier", "normal")
    result["year"] = result["date"].dt.year.astype(str)
    result["event"] = pd.Series(pd.NA, index=result.index, dtype="string")
    covid = result["split"].eq("train") & result["date"].between("2020-03-01", "2020-04-30")
    rates = result["split"].eq("val") & result["date"].dt.year.eq(2022)
    surge = result["split"].eq("test") & result["date"].dt.year.eq(2026)
    for mask, event in ((covid, "covid_2020"), (rates, "rate_rises_2022"), (surge, "surge_2026")):
        result.loc[mask, "event"] = event
    return result


def _metrics(rows):
    up, down = rows["y_true"].eq(1), rows["y_true"].eq(0)
    return {"n": len(rows), "accuracy": float(rows["y_pred"].eq(rows["y_true"]).mean()),
            "recall_up": float(rows.loc[up, "y_pred"].eq(1).mean()) if up.any() else float("nan"),
            "recall_down": float(rows.loc[down, "y_pred"].eq(0).mean()) if down.any() else float("nan"),
            "pred_up_rate": float(rows["y_pred"].mean()), "true_up_rate": float(rows["y_true"].mean())}


def segment_metrics(df):
    """Summarize observed segments from add_segments; omit empty groups/events."""
    records = []
    for (model, split), rows in df.groupby(["model", "split"], sort=True, observed=True):
        for segment_type in ("class", "volatility", "outlier", "year", "event"):
            for segment, group in rows.groupby(segment_type, observed=True, sort=True, dropna=True):
                records.append({"model": model, "split": split, "segment_type": segment_type,
                                "segment": str(segment), **_metrics(group)})
    return pd.DataFrame(records, columns=SEGMENT_COLUMNS).sort_values(
        ["model", "split", "segment_type", "segment"]).reset_index(drop=True)


def bias_check(df):
    """Compare each model with its own TRAIN majority class; ties choose class 0."""
    records = []
    for model, model_rows in df.groupby("model", sort=True):
        train_rows = model_rows.loc[model_rows["split"].eq("train")]
        if train_rows.empty:
            raise ValueError(f"Model {model!r} has no train rows for the majority baseline")
        majority_class = int(train_rows["y_true"].mean() > 0.5)
        for split, rows in model_rows.groupby("split", sort=True):
            stats = _metrics(rows)
            records.append({"model": model, "split": split,
                            "true_up_rate": stats["true_up_rate"], "pred_up_rate": stats["pred_up_rate"],
                            "gap": stats["pred_up_rate"] - stats["true_up_rate"],
                            "majority_baseline_acc": float(rows["y_true"].eq(majority_class).mean()),
                            "model_acc": stats["accuracy"]})
    return pd.DataFrame(records, columns=BIAS_COLUMNS).sort_values(["model", "split"]).reset_index(drop=True)


def print_summary(df, bias):
    """Report test rates, baseline comparisons, recalls, and zero-recall collapse flags."""
    test_bias = bias.loc[bias["split"].eq("test")].set_index("model")
    for model, rows in df.loc[df["split"].eq("test")].groupby("model", sort=True):
        stats = _metrics(rows)
        flag = " | COLLAPSED to one class" if stats["recall_up"] == 0 or stats["recall_down"] == 0 else ""
        print(f"{model}: test pred_up_rate={stats['pred_up_rate']:.6f} vs true_up_rate={stats['true_up_rate']:.6f}; "
              f"accuracy={stats['accuracy']:.6f} vs majority_baseline_acc={test_bias.loc[model, 'majority_baseline_acc']:.6f}; "
              f"recall_up={stats['recall_up']:.6f}, recall_down={stats['recall_down']:.6f}{flag}")


def plot_metrics(segments, fig_dir):
    """Draw all models together using deterministic colors and a headless backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(segments["model"].unique())
    cmap = plt.get_cmap("tab10" if len(models) <= 10 else "turbo")
    colors = {model: cmap(index if len(models) <= 10 else index / max(len(models) - 1, 1))
              for index, model in enumerate(models)}
    legend_rows = int(np.ceil((len(models) + 1) / 4))

    def finish(fig, ax, filename):
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.23)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(4, len(models) + 1),
                  fontsize=9, frameon=False)
        fig.tight_layout()
        path = fig_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(path)

    years = segments.loc[segments["segment_type"].eq("year")].copy()
    years["year"] = years["segment"].astype(int)
    keys = sorted(set(zip(years["split"], years["year"])), key=lambda key: (key[1], SPLITS.index(key[0])))
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.6), 5.5 + legend_rows * 0.3))
    for model in models:
        lookup = years.loc[years["model"].eq(model)].set_index(["split", "year"])["accuracy"]
        ax.plot(x, [lookup.get(key, np.nan) for key in keys], label=model, color=colors[model],
                marker="o", markersize=3, linewidth=1.7)
    start = 0
    for stop in range(1, len(keys) + 1):
        if stop == len(keys) or keys[stop][0] != keys[start][0]:
            split = keys[start][0]
            ax.axvspan(start - 0.5, stop - 0.5, color=("#DEEAF5", "#E4F1E8", "#FFF0D9")[SPLITS.index(split)],
                       alpha=0.6, zorder=0)
            ax.text((start + stop - 1) / 2, 1.02, split.capitalize(), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10)
            if stop < len(keys):
                ax.axvline(stop - 0.5, color="#667085", linestyle="--", linewidth=1)
            start = stop
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1, label="0.5 reference")
    ax.set_xticks(x, [str(year) for _, year in keys], rotation=45 if len(keys) > 10 else 0)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_title("Prediction accuracy by calendar year", fontsize=15, pad=32)
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Accuracy")
    finish(fig, ax, "accuracy_by_year.png")

    for segment_type, labels, title, ylabel, filename in (
        ("volatility", VOLATILITY_LABELS, "Test-split accuracy by volatility", "Accuracy", "accuracy_by_volatility.png"),
        ("class", ["true_up", "true_down"], "Test-split recall by true class", "Recall", "recall_by_class.png"),
    ):
        table = segments.loc[segments["split"].eq("test") & segments["segment_type"].eq(segment_type)]
        fig, ax = plt.subplots(figsize=(10, 5.5 + legend_rows * 0.3))
        x = np.arange(len(labels))
        width = 0.8 / len(models)
        for index, model in enumerate(models):
            rows = table.loc[table["model"].eq(model)].set_index("segment")
            values = [rows.loc[label, "accuracy" if segment_type == "volatility" else
                               ("recall_up" if label == "true_up" else "recall_down")]
                      if label in rows.index else np.nan for label in labels]
            ax.bar(x + (index - (len(models) - 1) / 2) * width, values, width=width,
                   label=model, color=colors[model])
        ax.set_xticks(x, [label.replace("_", " ").capitalize() for label in labels])
        ax.set_title(title, fontsize=15, pad=12)
        ax.set_xlabel("Train-defined volatility tercile" if segment_type == "volatility" else "True class")
        ax.set_ylabel(ylabel)
        finish(fig, ax, filename)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    parser.add_argument("--unscaled", type=Path, default=UNSCALED_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args(argv)
    predictions = load_predictions(args.pred_dir)
    models = sorted(predictions["model"].unique())
    print(f"Models found ({len(models)}): {', '.join(models)}")
    joined = join_unscaled(predictions, args.unscaled)
    cuts = train_vol_cuts(joined)
    print(f"Shared train volatility cuts: {cuts[0]:.6f}, {cuts[1]:.6f}")
    segmented = add_segments(joined, cuts)
    metrics, bias = segment_metrics(segmented), bias_check(segmented)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "segment_metrics.csv", index=False)
    bias.to_csv(args.out_dir / "bias_check.csv", index=False)
    print(f"Saved {len(metrics)} segment rows and {len(bias)} bias rows to {args.out_dir}")
    plot_metrics(metrics, args.fig_dir)
    print_summary(segmented, bias)


if __name__ == "__main__":
    main()
