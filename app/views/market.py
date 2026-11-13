"""Read-only market history and comparisons for Deliverable 3 B2."""
import sys
from pathlib import Path
from unittest.mock import patch

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SERIES = {
    "gold": "Gold (USD/oz)", "silver": "Silver (USD/oz)",
    "copper": "Copper (USD/lb)", "platinum": "Platinum (USD/oz)",
    "dxy": "US dollar index", "wti": "WTI oil (USD/bbl)",
    "us10y": "US 10-year yield (%)", "cpi": "US CPI index",
}
GREEN, AMBER, RED, GREY = "#4CAF50", "#D4A017", "#E5534B", "#969BA3"
MODES = ["Price + 30-day average", "Daily returns"]


@st.cache_data(persist=False, show_spinner=False)
def builtin_prices():
    """Cache frozen prices in memory without initializing cleaning log files."""
    # The shared module imports training-data helpers with file-logger setup.
    with patch("logging.basicConfig"), patch("pathlib.Path.mkdir"):
        from src.inference.features import load_builtin_prices
    return load_builtin_prices()


def styled(chart, height):
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor="#D8DCE2", titleColor="#D8DCE2",
                            gridColor="#30343A", domainColor="#555B63")
            .configure_legend(labelColor="#D8DCE2", titleColor="#D8DCE2"))


def percentage(value):
    return f"{value:+.2%}" if np.isfinite(value) else "n/a"


def comparison(frame, other, label):
    pair = frame[["gold", other]]
    span = pair.max() - pair.min()
    # A constant series has no relative movement; place it at zero.
    scaled = ((pair - pair.min()) / span.replace(0, np.nan) * 100).fillna(0)
    data = (scaled.rename(columns={"gold": "Gold", other: label})
            .reset_index().melt("date", var_name="Series", value_name="Scaled"))
    return alt.Chart(data).mark_line(point=len(frame) == 1).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("Scaled:Q", title="Scaled (0-100)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Gold", label],
                                                   range=[GREEN, AMBER]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                 "Series:N", alt.Tooltip("Scaled:Q", format=".2f")],
    )


try:
    prices = builtin_prices()
    if (prices.empty or not isinstance(prices.index, pd.DatetimeIndex)
            or prices.index.hasnans or not set(SERIES).union({"is_outlier"}) <= set(prices.columns)):
        raise ValueError("The built-in price data is empty or has an invalid schema.")
except Exception as exc:
    st.error(f"Built-in market data is unavailable. Restore data/processed/cleaned.csv. {exc}")
    st.stop()

first, last = prices.index.min().date(), prices.index.max().date()
st.info(f"Real ProcurePulse data. Prices up to {last:%d %b %Y}.")
c1, c2, c3, c4 = st.columns(4)
series = c1.selectbox("Series", list(SERIES), format_func=SERIES.get, key="market_series")
start = c2.date_input("From date", min(max(pd.Timestamp("2024-01-02").date(), first), last),
                      min_value=first, max_value=last, key="market_from")
end = c3.date_input("To date", last, min_value=first, max_value=last, key="market_to")
mode = c4.selectbox("Show", MODES, key="market_mode")
if start is None or end is None:
    st.info("Choose both dates to view market data.")
    st.stop()
if start > end:
    st.warning("From date must be on or before To date.")
    st.stop()

selected = prices.loc[str(start):str(end)]
if selected.empty:
    st.info("No data in this range.")
    st.stop()

# Compute daily returns and averages before filtering to retain past-only context.
values = prices[series]
returns = values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
daily = returns.loc[selected.index]
recent = selected[series]
change, volatility, comparison_help = np.nan, np.nan, "Requires at least 31 rows in the selected range."
if len(selected) >= 31:
    baseline = recent.iloc[-31]
    change = recent.iloc[-1] / baseline - 1 if baseline != 0 else np.nan
    last_returns = daily.iloc[-30:]
    if last_returns.notna().all():
        volatility = last_returns.std() * np.sqrt(252)
    comparison_help = f"vs {selected.index[-31]:%d %b %Y} (30 trading days earlier)."

m1, m2, m3, m4 = st.columns(4)
delta = daily.iloc[-1]
m1.metric("Last price", f"{recent.iloc[-1]:,.2f}",
          delta=percentage(delta) if np.isfinite(delta) else None,
          delta_color="off" if delta == 0 else "normal",
          help=f"{SERIES[series]} on {selected.index[-1]:%d %b %Y}; delta vs the previous trading day.")
m2.metric("30-day change", percentage(change), help=comparison_help)
m3.metric("30-day volatility", f"{volatility:.2%}" if np.isfinite(volatility) else "n/a",
          help="Annualised: sample standard deviation of the last 30 daily returns x sqrt(252). "
               "Requires at least 31 rows in the selected range.")
m4.metric("Outlier days in range", int(selected["is_outlier"].eq(1).sum()))
m4.caption("flagged, not removed")

chart_data = pd.DataFrame({"date": selected.index, "Price": recent.to_numpy(),
                           "Average": values.rolling(30).mean().loc[selected.index].to_numpy(),
                           "Return": daily.to_numpy() * 100,
                           "is_outlier": selected["is_outlier"].to_numpy()})
price_mode = mode == MODES[0]
field = "Price" if price_mode else "Return"
title = f"{SERIES[series]} {'price' if price_mode else 'daily returns'}"
base = alt.Chart(chart_data).encode(
    x=alt.X("date:T", title=None),
    y=alt.Y(f"{field}:Q", title=SERIES[series] if price_mode else "Daily return (%)",
            scale=alt.Scale(zero=False)),
    tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
             alt.Tooltip(f"{field}:Q", title=field if price_mode else "Return (%)", format=",.2f")],
)
line = base.mark_line(color=GREEN, point=len(selected) == 1)
outliers = base.transform_filter(alt.datum.is_outlier == 1).mark_point(color=RED, filled=True, size=45)
if price_mode:
    average = base.mark_line(color=GREY).encode(
        y="Average:Q", tooltip=[alt.Tooltip("date:T", format="%d %b %Y"),
                                alt.Tooltip("Average:Q", title="30-day average", format=",.2f")])
    chart = line + average + outliers
else:
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=GREY).encode(y="zero:Q")
    chart = zero + line + outliers

with st.container(border=True):
    st.subheader(title)
    st.altair_chart(styled(chart, 340), width="stretch", theme=None, alt=title)
    if price_mode:
        st.caption("Green line: price. Grey line: 30-day average. Red dots: days flagged as outliers "
                   "in Deliverable 1 cleaning (a day is flagged if any series had a shock).")
    else:
        st.caption("Green line: daily returns. Grey line: zero. Red dots: days flagged as outliers "
                   "in Deliverable 1 cleaning (a day is flagged if any series had a shock).")
    if series == "cpi":
        st.caption("CPI is monthly and forward-filled, so its daily returns are mostly zero.")

left, right = st.columns(2)
for column, other, label, caption in [
    (left, "dxy", "US dollar index", "A stronger dollar usually means cheaper gold."),
    (right, "wti", "WTI oil", "Oil is an inflation signal used as a model feature."),
]:
    with column:
        st.subheader(f"Gold vs {label}")
        st.altair_chart(styled(comparison(selected, other, label), 220),
                        width="stretch", theme=None, alt=f"Gold vs {label}, scaled over the selected range")
        st.caption(caption)
st.caption("Both lines are scaled 0 to 100 over the selected range so they fit on one chart.")
