"""Exploratory Data Analysis (EDA) visualisations for ProcurePulse.

Generates publication-quality charts for Deliverable 1 report:
  1. correlation_heatmap.png: Correlation matrix of daily returns
  2. class_distribution.png: Target direction balance across train, val, and test splits
  3. return_distributions.png: Histograms and KDEs for gold, silver, and copper returns
  4. outliers_before_after.png: Multi-panel boxplots comparing raw vs. capped return distributions
  5. gold_decomposition.png: Multiplicative seasonal/trend decomposition of monthly gold prices
  6. gold_volatility.png: 30-day rolling gold daily volatility time series with peak annotation
  7. gold_vs_dxy.png: Scatter plot and regression trend line of gold returns vs. US Dollar Index
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from config import FIG_EDA_DIR, LOG_DIR, METAL_COLS, PRICE_COLS, PROCESSED_DIR, RAW_DIR

# Ensure output directory and log directory exist
FIG_EDA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging matching clean_data and feature_engineering
logging.basicConfig(
    filename=LOG_DIR / "cleaning.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Unified chart styling for publication quality
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#ebebeb",
    "grid.linewidth": 0.6,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "font.size": 9.5,
})


def plot_correlation_heatmap(df_unscaled: pd.DataFrame) -> None:
    """Chart 1: Correlation heatmap of daily returns across metals and external signals."""
    ret_cols = [f"ret_{c}" for c in PRICE_COLS] + ["ret_us10y"]
    corr = df_unscaled[ret_cols].corr()

    # Clean display labels for axis ticks with explicit percentage point unit for US 10Y
    display_names = [
        "Gold", "Silver", "Copper", "Platinum",
        "USD Index (DXY)", "WTI Crude", "us10y (change, pp)"
    ]

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        xticklabels=display_names,
        yticklabels=display_names,
        cbar_kws={"label": "Pearson Correlation Coefficient (r)", "shrink": 0.8},
        ax=ax
    )
    ax.set_title("Daily Return Correlations: Metals & Macroeconomic Signals", pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    # Note clarifying that us10y is change in percentage points, not relative percent
    ax.text(
        0.5, -0.16,
        "Note: All series are daily % returns except us10y which is daily change in percentage points (pp).",
        ha="center", va="top", transform=ax.transAxes, fontsize=8.5, style="italic", color="#555555"
    )

    fig.tight_layout()
    fig.savefig(FIG_EDA_DIR / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Key metrics for report text
    r_gold_silver = corr.loc["ret_gold", "ret_silver"]
    r_gold_dxy = corr.loc["ret_gold", "ret_dxy"]
    r_gold_us10y = corr.loc["ret_gold", "ret_us10y"]
    msg = (
        f"eda: correlation_heatmap.png generated. "
        f"Gold-Silver correlation: r = {r_gold_silver:.2f} (strongest co-movement); "
        f"Gold-DXY correlation: r = {r_gold_dxy:.2f} (negative currency hedge); "
        f"Gold-US10Y correlation: r = {r_gold_us10y:.2f} (opportunity cost of non-yielding bullion)."
    )
    logging.info(msg)
    print(msg)


def plot_class_distribution(df_unscaled: pd.DataFrame) -> None:
    """Chart 2: Target direction_5d class counts across chronological Train/Val/Test splits."""
    # Partition matching feature_engineering.py
    train = df_unscaled.loc[:"2021-12-31"]
    val = df_unscaled.loc["2022-01-01":"2023-12-31"]
    test = df_unscaled.loc["2024-01-01":]

    splits = [("Train (2010-2021)", train), ("Validation (2022-2023)", val), ("Test (2024-2026)", test)]

    data = []
    summary_stats = {}
    for name, subset in splits:
        n_down = int((subset["direction_5d"] == 0).sum())
        n_up = int((subset["direction_5d"] == 1).sum())
        total = len(subset)
        up_pct = (n_up / total) * 100
        summary_stats[name] = (n_up, n_down, up_pct, total)
        data.append({"Split": name, "Direction": "Down / Flat (0)", "Count": n_down})
        data.append({"Split": name, "Direction": "Up (1)", "Count": n_up})

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {"Down / Flat (0)": "#e34948", "Up (1)": "#1baf7a"}
    sns.barplot(data=plot_df, x="Split", y="Count", hue="Direction", palette=palette, ax=ax)

    # Annotate bar tops with sample counts and %
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2.0, h + 25),
                        ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_ylim(0, max(plot_df["Count"]) * 1.15)
    ax.set_ylabel("Number of Trading Days")
    ax.set_xlabel("Chronological Split")
    ax.set_title("5-Day Ahead Direction Target Class Balance Across Partitions", pad=10)
    ax.legend(title="Target Label (direction_5d)", loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_EDA_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)

    msg = (
        f"eda: class_distribution.png generated. "
        f"Train Up: {summary_stats['Train (2010-2021)'][0]}/{summary_stats['Train (2010-2021)'][3]} ({summary_stats['Train (2010-2021)'][2]:.1f}%); "
        f"Val Up: {summary_stats['Validation (2022-2023)'][0]}/{summary_stats['Validation (2022-2023)'][3]} ({summary_stats['Validation (2022-2023)'][2]:.1f}%); "
        f"Test Up: {summary_stats['Test (2024-2026)'][0]}/{summary_stats['Test (2024-2026)'][3]} ({summary_stats['Test (2024-2026)'][2]:.1f}%)."
    )
    logging.info(msg)
    print(msg)


def plot_return_distributions(df_cleaned: pd.DataFrame, df_unscaled: pd.DataFrame) -> None:
    """Chart 3: Histograms and KDE curves comparing uncapped and 3xIQR capped return distributions."""
    metals = [("gold", "Gold", "#2a78d6"),
              ("silver", "Silver", "#eb6834"),
              ("copper", "Copper", "#1baf7a")]

    # Compute raw uncapped returns from cleaned.csv
    uncapped_returns = df_cleaned[["gold", "silver", "copper"]].pct_change().dropna() * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    metrics_uncapped = {}
    metrics_capped = {}

    for ax, (col, label, color) in zip(axes, metals):
        # 1. Uncapped series statistics
        u_ret = uncapped_returns[col]
        u_skew = stats.skew(u_ret)
        u_kurt = stats.kurtosis(u_ret, fisher=True)  # Excess kurtosis (normal=0)
        metrics_uncapped[label] = (u_skew, u_kurt)

        # 2. Capped series (plotted in chart)
        c_ret = df_unscaled[f"ret_{col}"] * 100
        c_skew = stats.skew(c_ret)
        c_kurt = stats.kurtosis(c_ret, fisher=True)
        metrics_capped[label] = (c_skew, c_kurt)

        # Plot capped distribution
        sns.histplot(c_ret, kde=True, bins=70, color=color, ax=ax,
                     stat="density", line_kws={"linewidth": 1.5})
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)

        # Explicitly present both uncapped and capped metrics in subplot header
        subplot_title = (
            f"{label} Returns (Capped)\n"
            f"Uncapped: Skew={u_skew:.2f}, Kurt={u_kurt:.1f}\n"
            f"Capped (3xIQR): Skew={c_skew:.2f}, Kurt={c_kurt:.1f}"
        )
        ax.set_title(subplot_title, loc="left", fontsize=9.2)
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Probability Density")
        ax.set_xlim(-8, 8)

    fig.suptitle("Daily Return Distributions (Capped at 3xIQR): Leptokurtic Fat Tails in Base and Precious Metals",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_EDA_DIR / "return_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    msg_uncapped = (
        f"eda: return_distributions.png uncapped: "
        f"gold (skew={metrics_uncapped['Gold'][0]:.2f}, kurtosis={metrics_uncapped['Gold'][1]:.1f}); "
        f"silver (skew={metrics_uncapped['Silver'][0]:.2f}, kurtosis={metrics_uncapped['Silver'][1]:.1f}); "
        f"copper (skew={metrics_uncapped['Copper'][0]:.2f}, kurtosis={metrics_uncapped['Copper'][1]:.1f})."
    )
    msg_capped = (
        f"eda: return_distributions.png capped (3xIQR, train bounds): "
        f"gold (skew={metrics_capped['Gold'][0]:.2f}, kurtosis={metrics_capped['Gold'][1]:.1f}); "
        f"silver (skew={metrics_capped['Silver'][0]:.2f}, kurtosis={metrics_capped['Silver'][1]:.1f}); "
        f"copper (skew={metrics_capped['Copper'][0]:.2f}, kurtosis={metrics_capped['Copper'][1]:.1f})."
    )
    logging.info(msg_uncapped)
    logging.info(msg_capped)
    print(msg_uncapped)
    print(msg_capped)


def plot_outliers_before_after(df_raw: pd.DataFrame, df_unscaled: pd.DataFrame) -> None:
    """Chart 4: Multi-panel boxplots comparing raw vs. capped returns with independent scales."""
    # Compute raw returns per column from merged_raw.csv
    raw_returns = df_raw[PRICE_COLS].pct_change().dropna() * 100
    capped_returns = df_unscaled[[f"ret_{c}" for c in PRICE_COLS]] * 100
    capped_returns.columns = PRICE_COLS

    # 2 rows x 3 columns grid: each series gets its own dedicated panel
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    axes = axes.flatten()

    titles = {
        "gold": "Gold (COMEX)", "silver": "Silver (COMEX)", "copper": "Copper (COMEX)",
        "platinum": "Platinum (NYMEX)", "dxy": "US Dollar Index", "wti": "WTI Crude Oil"
    }

    outlier_summary = {}

    for i, col in enumerate(PRICE_COLS):
        ax = axes[i]
        raw_series = raw_returns[col].dropna()
        capped_series = capped_returns[col].dropna()

        # Compute standard 1.5xIQR whisker boundaries
        q1_raw, q3_raw = raw_series.quantile(0.25), raw_series.quantile(0.75)
        iqr_raw = q3_raw - q1_raw
        raw_outliers = int(((raw_series < q1_raw - 1.5 * iqr_raw) |
                            (raw_series > q3_raw + 1.5 * iqr_raw)).sum())

        q1_cap, q3_cap = capped_series.quantile(0.25), capped_series.quantile(0.75)
        iqr_cap = q3_cap - q1_cap
        cap_outliers = int(((capped_series < q1_cap - 1.5 * iqr_cap) |
                            (capped_series > q3_cap + 1.5 * iqr_cap)).sum())

        # Count values resting exactly on the clipping boundaries (min or max)
        c_min, c_max = capped_series.min(), capped_series.max()
        at_bounds = int(((capped_series == c_min) | (capped_series == c_max)).sum())

        outlier_summary[col] = (raw_outliers, cap_outliers, at_bounds)

        # Build local dataframe for seaborn boxplot
        data_to_plot = pd.DataFrame({
            "Return (%)": pd.concat([raw_series, capped_series], ignore_index=True),
            "Dataset": ["Raw"] * len(raw_series) + ["Capped (3xIQR)"] * len(capped_series)
        })

        sns.boxplot(
            data=data_to_plot, x="Dataset", y="Return (%)", hue="Dataset",
            palette={"Raw": "#eb6834", "Capped (3xIQR)": "#2a78d6"},
            legend=False, fliersize=2.5, linewidth=0.9, ax=ax
        )
        ax.set_title(titles[col], loc="left", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Daily Return (%)")

        # Focus y-limits to make the boxes visible while noting extreme outliers
        if col == "wti":
            # WTI raw had an extreme negative spike (-300%+ on 2020-04-20)
            ax.set_ylim(-25, 25)
            ax.text(0.03, 0.05, "WTI raw min <-300% truncated\n(storage exhaustion Apr 2020)",
                    transform=ax.transAxes, fontsize=7.5, color="#990000",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff2f2", edgecolor="#ffcccc"))
        elif col in ["silver", "copper"]:
            ax.set_ylim(-15, 15)
        else:
            ax.set_ylim(-10, 10)

        # Annotate whisker outlier counts
        ax.text(0.97, 0.95, f"Raw >1.5xIQR: {raw_outliers}\nCapped: {cap_outliers}\nAt bounds: {at_bounds}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f5f5", edgecolor="#cccccc"))

    fig.suptitle("Outlier Profiles Before and After 3xIQR Training-Fitted Capping",
                 fontsize=12, fontweight="bold", y=0.98)

    # Note explaining that capping uses 3xIQR while boxplot whiskers use 1.5xIQR
    fig.text(
        0.5, 0.01,
        "Note: Box plot whiskers extend to 1.5xIQR, whereas capping applies 3xIQR training bounds. "
        "Clipped values therefore remain beyond whiskers but with truncated extremity.",
        ha="center", fontsize=8.5, style="italic", color="#444444"
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(FIG_EDA_DIR / "outliers_before_after.png", dpi=200)
    plt.close(fig)

    msg = (
        f"eda: outliers_before_after.png generated. "
        f"Whisker outliers (1.5xIQR raw -> capped): " +
        ", ".join([f"{c}: {outlier_summary[c][0]} -> {outlier_summary[c][1]} (at bounds: {outlier_summary[c][2]})"
                   for c in PRICE_COLS])
    )
    logging.info(msg)
    print(msg)


def plot_gold_decomposition(df_cleaned: pd.DataFrame) -> None:
    """Chart 5: Multiplicative seasonal/trend decomposition of monthly gold prices."""
    # Resample daily cleaned gold prices to monthly start mean
    monthly_gold = df_cleaned["gold"].resample("MS").mean()

    # Multiplicative model: Y_t = Trend_t * Seasonal_t * Residual_t
    decomp = seasonal_decompose(monthly_gold, model="multiplicative", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    palette = ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6"]

    # 1. Observed
    axes[0].plot(decomp.observed.index, decomp.observed, color=palette[0], linewidth=1.2)
    axes[0].set_ylabel("Observed ($)")
    axes[0].set_title("Multiplicative Seasonal Decomposition of Monthly Gold Spot Prices (2010–2026)",
                      fontsize=11, fontweight="bold", loc="left")

    # 2. Trend
    axes[1].plot(decomp.trend.index, decomp.trend, color=palette[1], linewidth=1.4)
    axes[1].set_ylabel("Trend ($)")

    # 3. Seasonal Factor
    axes[2].plot(decomp.seasonal.index, decomp.seasonal, color=palette[2], linewidth=1.1)
    axes[2].axhline(1.0, color="#888888", linestyle="--", linewidth=0.7)
    axes[2].set_ylabel("Seasonal Factor")

    # 4. Residual
    axes[3].scatter(decomp.resid.index, decomp.resid, color=palette[3], s=10, alpha=0.7)
    axes[3].axhline(1.0, color="#888888", linestyle="--", linewidth=0.7)
    axes[3].set_ylabel("Residual (Ratio)")
    axes[3].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(FIG_EDA_DIR / "gold_decomposition.png", dpi=200)
    plt.close(fig)

    # Seasonal metrics: amplitude of seasonality in percentage terms
    seas_min, seas_max = decomp.seasonal.min(), decomp.seasonal.max()
    seasonal_swing_pct = (seas_max - seas_min) * 100

    # Trend component growth vs. raw price growth
    trend_clean = decomp.trend.dropna()
    trend_start, trend_end = trend_clean.iloc[0], trend_clean.iloc[-1]
    trend_growth = (trend_end / trend_start - 1) * 100

    raw_start = df_cleaned["gold"].iloc[0]
    raw_end = df_cleaned["gold"].iloc[-1]
    raw_growth = (raw_end / raw_start - 1) * 100

    msg_decomp = f"eda: decomposition trend component rose {trend_growth:.1f}% (from ${trend_start:.2f} to ${trend_end:.2f})"
    msg_raw = f"eda: raw gold price rose {raw_growth:.1f}% (from ${raw_start:.2f} to ${raw_end:.2f})"
    msg_seas = (
        f"eda: gold_decomposition.png generated. "
        f"Multiplicative seasonal range: [{seas_min:.4f}, {seas_max:.4f}], "
        f"maximum seasonal swing is {seasonal_swing_pct:.2f}%."
    )
    logging.info(msg_decomp)
    logging.info(msg_raw)
    logging.info(msg_seas)
    print(msg_decomp)
    print(msg_raw)
    print(msg_seas)


def plot_gold_volatility(df_unscaled: pd.DataFrame) -> None:
    """Chart 6: 30-day rolling gold daily volatility with annotated peak regime."""
    # Daily return standard deviation in % (not annualized)
    vol_series = df_unscaled["gold_vol30"] * 100

    peak_date = vol_series.idxmax()
    peak_val = vol_series.loc[peak_date]
    median_val = vol_series.median()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(vol_series.index, vol_series.values, color="#6250d6", linewidth=1.2,
            label="30-day Rolling Volatility (Daily Returns)")
    ax.axhline(median_val, color="#888888", linestyle="--", linewidth=0.8,
               label=f"Historical Median ({median_val:.2f}%)")

    # Annotate all-time peak volatility regime (March 2026)
    peak_label = f"All-time Peak: {peak_val:.2f}%\n({peak_date:%d %b %Y})\n2026 Geopolitical Rally"
    ax.annotate(
        peak_label,
        xy=(peak_date, peak_val),
        xytext=(peak_date - pd.Timedelta(days=800), peak_val * 0.95),
        arrowprops=dict(facecolor="#e34948", edgecolor="#e34948", arrowstyle="->", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#e34948", lw=0.8)
    )

    # Annotate secondary visible peak: COVID-19 shock (March 2020)
    covid_date = pd.Timestamp("2020-03-24")
    if covid_date in vol_series.index:
        covid_val = vol_series.loc[covid_date]
    else:
        covid_date = vol_series.loc["2020-03-01":"2020-04-30"].idxmax()
        covid_val = vol_series.loc[covid_date]

    ax.annotate(
        f"COVID-19 Shock: {covid_val:.2f}%\n({covid_date:%b %Y})",
        xy=(covid_date, covid_val),
        xytext=(covid_date - pd.Timedelta(days=700), covid_val * 1.05),
        arrowprops=dict(facecolor="#2a78d6", edgecolor="#2a78d6", arrowstyle="->", lw=1.1),
        fontsize=8.0,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#2a78d6", lw=0.8)
    )

    ax.set_ylabel("30-day rolling std of daily gold returns (%)")
    ax.set_xlabel("Date")
    ax.set_title("Gold Price 30-Day Rolling Daily Volatility (2010–2026)", loc="left", pad=10)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG_EDA_DIR / "gold_volatility.png", dpi=200)
    plt.close(fig)

    msg = (
        f"eda: gold_volatility.png generated. "
        f"All-time peak daily volatility: {peak_val:.2f}% on {peak_date:%Y-%m-%d} (2026 market surge); "
        f"COVID-19 peak: {covid_val:.2f}% on {covid_date:%Y-%m-%d}; "
        f"Baseline median daily volatility: {median_val:.2f}%."
    )
    logging.info(msg)
    print(msg)


def plot_gold_vs_dxy(df_unscaled: pd.DataFrame) -> None:
    """Chart 7: Scatter plot with regression line of Gold returns vs. US Dollar Index returns."""
    x = df_unscaled["ret_dxy"] * 100  # DXY return %
    y = df_unscaled["ret_gold"] * 100  # Gold return %

    # Linear regression fit: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, deg=1)
    r = np.corrcoef(x, y)[0, 1]
    r_squared = r ** 2

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, color="#2a78d6", alpha=0.3, s=12, edgecolors="none", label="Daily Trading Observations")

    # Draw regression line across range
    x_grid = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_grid + intercept
    reg_label = f"Fit: y = {slope:.2f}x + {intercept:.2f} (r = {r:.2f}, R² = {r_squared:.2f})"
    ax.plot(x_grid, y_fit, color="#e34948", linewidth=1.8, label=reg_label)

    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_xlabel("US Dollar Index (DXY) Daily Return (%)")
    ax.set_ylabel("Gold Futures Daily Return (%)")
    ax.set_title("Inverse Relationship: COMEX Gold Returns vs. US Dollar Index Returns", loc="left", pad=10)
    ax.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(FIG_EDA_DIR / "gold_vs_dxy.png", dpi=200)
    plt.close(fig)

    msg = (
        f"eda: gold_vs_dxy.png generated. "
        f"Correlation r = {r:.3f}, regression slope = {slope:.3f}, R-squared = {r_squared:.3f}. "
        f"Strong inverse sensitivity confirms USD strength exerts downward pressure on bullion."
    )
    logging.info(msg)
    print(msg)


def main():
    # Load required data files
    path_unscaled = PROCESSED_DIR / "features_unscaled.csv"
    path_cleaned = PROCESSED_DIR / "cleaned.csv"
    path_raw = RAW_DIR / "merged_raw.csv"

    if not path_unscaled.exists():
        raise FileNotFoundError(f"Missing {path_unscaled}. Run feature_engineering.py first.")
    if not path_cleaned.exists():
        raise FileNotFoundError(f"Missing {path_cleaned}. Run clean_data.py first.")
    if not path_raw.exists():
        raise FileNotFoundError(f"Missing {path_raw}. Run merge_sources.py first.")

    df_unscaled = pd.read_csv(path_unscaled, index_col="date", parse_dates=True)
    df_cleaned = pd.read_csv(path_cleaned, index_col="date", parse_dates=True)
    df_raw = pd.read_csv(path_raw, index_col="date", parse_dates=True)

    print("Running exploratory data analysis (EDA) chart generation...")

    # Generate the 7 charts
    plot_correlation_heatmap(df_unscaled)
    plot_class_distribution(df_unscaled)
    plot_return_distributions(df_cleaned, df_unscaled)
    plot_outliers_before_after(df_raw, df_unscaled)
    plot_gold_decomposition(df_cleaned)
    plot_gold_volatility(df_unscaled)
    plot_gold_vs_dxy(df_unscaled)

    print(f"\nAll 7 EDA charts successfully saved to {FIG_EDA_DIR}.")


if __name__ == "__main__":
    main()
