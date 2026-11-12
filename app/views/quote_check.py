from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.classical import MODEL_NAMES, predict_classical
from src.inference.features import PRICE_COLUMNS, build_features, load_builtin_prices, read_upload
from src.recommendation.price_range import HORIZON, METALS, UNITS, check_quote

try:
    from src.inference.lstm import predict_lstm
except Exception:
    predict_lstm = None

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "results" / "tables" / "model_comparison.csv"
LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest",
          "hist_gb": "Gradient boosting", "lstm": "LSTM"}
COLORS = {"BUY": "#4caf50", "HOLD": "#d4a017", "NEGOTIATE": "#e5534b"}
SOURCES = ["Built-in data", "Upload CSV"]


@st.cache_data
def builtin_prices():
    return load_builtin_prices()


@st.cache_data
def gold_probabilities(prices):
    features = build_features(prices)
    probs = predict_classical(features)
    if predict_lstm is not None:
        try:
            probs.update(predict_lstm(features))
        except Exception:
            pass
    return probs


@st.cache_data
def test_auc():
    if not COMPARISON.exists():
        return {}
    table = pd.read_csv(COMPARISON)
    table = table[table["split"] == "test"]
    return dict(zip(table["model"], table["roc_auc"]))


def signed(value, fmt):
    text = format(value, fmt)
    return text.lstrip("+-") if text.lstrip("-+").strip("0.%, ") == "" else text


def probability_bar(p):
    color = COLORS["BUY"] if p >= 0.5 else COLORS["NEGOTIATE"]
    return (f"<div style='position:relative;height:12px;margin-top:8px;background:#1f2123;border-radius:3px'>"
            f"<div style='width:{p * 100:.1f}%;height:100%;background:{color};border-radius:3px'></div>"
            f"<div style='position:absolute;left:50%;top:-4px;bottom:-4px;width:1px;background:#8b8d90'></div></div>")


def template_csv(prices):
    return prices[PRICE_COLUMNS].tail(120).reset_index().to_csv(index=False).encode()


def range_bar(r, quote, up_prob=None):
    lo, hi = min(r["low"], quote, r["last_price"]), max(r["high"], quote, r["last_price"])
    pad = (hi - lo) * 0.15
    lo, hi = lo - pad, hi + pad

    def pos(v):
        return (v - lo) / (hi - lo) * 100

    if up_prob is None:
        buy_to, legend = r["mid"], f"Green: up to the typical level {r['mid']:,.2f}"
    elif up_prob >= 0.5:
        buy_to, legend = r["high"], "Green: inside the range, model expects a rise"
    else:
        buy_to, legend = r["low"], f"Green: below the expected low {r['low']:,.2f}"
    green, high, mid = pos(buy_to), pos(r["high"]), pos(r["mid"])
    return f"""
<div style="position:relative;height:28px;margin:34px 0 30px;border-radius:4px;overflow:visible;
background:linear-gradient(90deg,#2e5e30 0 {green}%,#d4a017 {green}% {high}%,#8e3b37 {high}% 100%)">
  <div style="position:absolute;left:{mid}%;top:0;bottom:0;width:2px;background:#0f1011"></div>
  <div style="position:absolute;left:{pos(quote)}%;top:-10px;height:48px;width:3px;background:#fff"></div>
  <div style="position:absolute;left:{pos(quote)}%;top:-30px;transform:translateX(-50%);font-size:13px;white-space:nowrap">Quote {quote:,.2f}</div>
  <div style="position:absolute;left:{pos(r['last_price'])}%;top:-6px;height:40px;width:2px;background:#8b8d90"></div>
  <div style="position:absolute;left:{pos(r['last_price'])}%;top:38px;transform:translateX(-50%);font-size:12px;color:#8b8d90;white-space:nowrap">Market {r['last_price']:,.2f}</div>
</div>
<div style="display:flex;justify-content:space-between;font-size:12px;color:#8b8d90">
  <span>{legend} (BUY)</span>
  <span>Expected {HORIZON}-day range {r['low']:,.2f} to {r['high']:,.2f}</span>
  <span>Red: above {r['high']:,.2f} (NEGOTIATE)</span>
</div>"""


st.title("Check a supplier quote")
left, right = st.columns([1, 2], gap="large")

with left:
    metal = st.selectbox("Metal", METALS, format_func=lambda m: f"{m.title()} ({UNITS[m]})")
    source = st.radio("Market data", SOURCES, horizontal=True)

    prices, base, error = None, None, None
    try:
        base = builtin_prices()
    except FileNotFoundError as exc:
        error = str(exc)

    if base is not None and source == SOURCES[0]:
        prices = base
        st.caption(f"ProcurePulse data, {base.index[0]:%d %b %Y} to {base.index[-1]:%d %b %Y}.")
    elif base is not None:
        st.download_button("Download template", template_csv(base), "procurepulse_template.csv", "text/csv")
        upload = st.file_uploader("CSV with date, " + ", ".join(PRICE_COLUMNS), type="csv")
        if upload is None:
            st.caption("Upload at least 100 trading days of prices.")
        else:
            try:
                prices = read_upload(upload)
                st.caption(f"Uploaded {len(prices)} trading days, {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}.")
            except ValueError as exc:
                error = str(exc)

    if error:
        st.error(error)

    unit = UNITS[metal].split("/")[1]
    market = float(prices[metal].iloc[-1]) if prices is not None else 1.0
    quote = st.number_input(f"Quoted price ({UNITS[metal]})", min_value=0.01, value=round(market, 2),
                            step=round(max(market * 0.001, 0.01), 2), key=f"quote_{metal}_{source}_{market}")
    quantity = st.number_input(f"Quantity ({unit})", min_value=1.0, value=10.0, step=1.0)

    model = None
    if metal == "gold":
        options = MODEL_NAMES + (["lstm"] if predict_lstm is not None else [])
        model = st.selectbox("Model used for the recommendation", options, format_func=LABELS.get)

    clicked = st.button("Check quote", type="primary", use_container_width=True, disabled=prices is None)

