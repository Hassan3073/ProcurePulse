"""Phase 2 visualisation: charts of the raw (uncleaned) data, saved to report/figures/raw/."""
import matplotlib

matplotlib.use("Agg")  # save files only, no window
import matplotlib.pyplot as plt
import pandas as pd

from config import ALL_COLS, FIG_RAW_DIR, METAL_COLS, RAW_DIR

# One fixed colour per series, used in every chart so each series keeps its identity.
COLORS = dict(zip(ALL_COLS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                             "#e87ba4", "#008300", "#6250d6", "#e34948"]))
LABELS = {
    "gold": ("Gold futures (COMEX)", "USD per troy ounce"),
    "silver": ("Silver futures (COMEX)", "USD per troy ounce"),
    "copper": ("Copper futures (COMEX)", "USD per pound"),
    "platinum": ("Platinum futures (NYMEX)", "USD per troy ounce"),
    "dxy": ("US Dollar Index", "Index points"),
    "wti": ("WTI crude oil futures", "USD per barrel"),
    "us10y": ("10-year US Treasury yield", "Percent"),
    "cpi": ("US Consumer Price Index", "Index (1982-84 = 100)"),
}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": "#0b0b0b",
    "axes.labelsize": 9, "axes.labelcolor": "#52514e",
    "xtick.color": "#898781", "ytick.color": "#898781",
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.size": 9,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_RAW_DIR / name, dpi=200)
    plt.close(fig)
    print(f"saved report/figures/raw/{name}")


def plot_series_grid(df, cols, title, name):
    """One small chart per series, so series on different scales are never forced onto one axis."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, col in zip(axes.flat, cols):
        series = df[col].dropna()
        ax.plot(series.index, series.values, color=COLORS[col], linewidth=1.1)
        ax.set_title(LABELS[col][0], loc="left")
        ax.set_ylabel(LABELS[col][1])
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color="#0b0b0b")
    save(fig, name)


def plot_missing(df):
    """Bar chart of missing values per column in the raw merged table."""
    missing = df.isna().sum()
    pct = missing / len(df) * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh([LABELS[c][0] for c in missing.index], missing.values, color="#2a78d6", height=0.6)
    for y, (count, share) in enumerate(zip(missing.values, pct.values)):
        ax.text(count, y, f"  {count:,} ({share:.1f}%)", va="center", fontsize=8, color="#52514e")
    ax.invert_yaxis()
    ax.set_xlim(0, missing.max() * 1.25)
    ax.set_xlabel(f"Missing days (out of {len(df):,} rows)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Missing values per column in merged_raw.csv",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color="#0b0b0b")
    ax.set_title("CPI is published monthly, so most daily rows have no CPI value",
                 loc="left", fontsize=9, fontweight="normal", color="#52514e")
    save(fig, "missing_values.png")


def plot_return_distributions(df):
    """Histograms of raw daily returns; the log scale makes the extreme days visible."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, col in zip(axes.flat, METAL_COLS):
        returns = df[col].dropna().pct_change().dropna() * 100
        ax.hist(returns, bins=120, color=COLORS[col], edgecolor="#fcfcfb", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{LABELS[col][0]}: daily returns", loc="left")
        ax.set_xlabel("Daily change (%)")
        ax.set_ylabel("Number of days (log scale)")
        ax.text(0.98, 0.95, f"min {returns.min():.1f}%\nmax {returns.max():.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#52514e")
    fig.suptitle("Raw daily returns of the metals (before outlier handling)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color="#0b0b0b")
    save(fig, "raw_return_distributions.png")


def main():
    FIG_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_DIR / "merged_raw.csv", index_col="date", parse_dates=True)
    plot_series_grid(df, METAL_COLS, "Raw daily closing prices of the metals, 2010 to 2026",
                     "raw_metal_prices.png")
    plot_series_grid(df, ["dxy", "wti", "us10y", "cpi"],
                     "Raw external signals, 2010 to 2026", "raw_external_signals.png")
    plot_missing(df)
    plot_return_distributions(df)


if __name__ == "__main__":
    main()