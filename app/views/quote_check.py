from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.classical import MODEL_NAMES, predict_classical
from src.inference.features import PRICE_COLUMNS, build_features, load_builtin_prices, read_upload
from src.inference.live import download_latest
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
SOURCES = ["Built-in data", "Upload CSV", "Latest prices"]


@st.cache_data
def builtin_prices():
    return load_builtin_prices()

@st.cache_data(ttl=3600, show_spinner="Downloading the latest prices...")
def latest_prices():
    return download_latest()


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
  <div style="position:absolute;left:{mid}%;top:0;bottom:0;width:2px;background:#121315"></div>
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
st.caption("Enter a supplier's price. ProcurePulse compares it with how far the market usually moves in "
           f"{HORIZON} trading days and tells you whether to buy, wait or negotiate.")
left, right = st.columns([1, 2], gap="large")

with left:
    metal = st.selectbox("1. Metal", METALS, format_func=lambda m: f"{m.title()} ({UNITS[m]})")
    source = st.radio("2. Market data", SOURCES, horizontal=True)

    prices, base, error = None, None, None
    try:
        base = builtin_prices()
    except FileNotFoundError as exc:
        error = str(exc)

    if base is not None and source == SOURCES[0]:
        prices = base
        st.caption(f"ProcurePulse data, {base.index[0]:%d %b %Y} to {base.index[-1]:%d %b %Y}.")

    elif base is not None and source == SOURCES[2]:
        try:
            prices = latest_prices()
            st.caption(f"Downloaded {len(prices)} trading days, up to {prices.index[-1]:%d %b %Y}. Needs internet.")
        except Exception as exc:
            error = f"Could not download the latest prices ({exc}). Check the internet connection or use built-in data."
            
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
    quote = st.number_input(f"3. Quoted price ({UNITS[metal]})", min_value=0.01, value=round(market, 2),
                            step=round(max(market * 0.001, 0.01), 2), key=f"quote_{metal}_{source}_{market}")
    quantity = st.number_input(f"4. Quantity ({unit})", min_value=1.0, value=10.0, step=1.0)

    model = None
    if metal == "gold":
        options = MODEL_NAMES + (["lstm"] if predict_lstm is not None else [])
        model = st.selectbox("5. Model used for the recommendation", options, format_func=LABELS.get)

    clicked = st.button("Check quote", type="primary", use_container_width=True, disabled=prices is None)

inputs = (metal, source, quote, quantity, model, None if prices is None else prices.index[-1])

if clicked and prices is not None:
    probs, warning = {}, None
    up_prob = None
    if metal == "gold":
        try:
            probs = gold_probabilities(prices)
            up_prob = probs.get(model)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            warning = f"The AI model could not run ({exc}). The recommendation uses the price range only."
    history = base.loc[base.index < prices.index[0], METALS]
    range_prices = pd.concat([history, prices[METALS]])
    result = check_quote(metal, quote, quantity, up_prob, prices=range_prices)
    st.session_state["quote_result"] = {"inputs": inputs, "result": result, "probs": probs,
                                        "model": model, "warning": warning}

with right:
    saved = st.session_state.get("quote_result")
    if saved is None:
        st.info("Choose a metal, enter the supplier's quote and click **Check quote**.")
    else:
        r = saved["result"]
        if saved["inputs"] != inputs:
            st.caption("Inputs changed. Click **Check quote** to update the result.")
        color = COLORS[r["action"]]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:18px;margin-bottom:6px'>"
            f"<div style='font-size:34px;font-weight:700;color:{color};border:2px solid {color};"
            f"border-radius:10px;padding:6px 18px'>{r['action']}</div>"
            f"<div style='color:#8b8d90'>{r['metal'].title()} ({r['unit']})<br>{r['basis']}</div></div>",
            unsafe_allow_html=True)
        if r["basis"] == "Price range only":
            rules = ("**Price range only** (no AI model used)\n\n"
                     f"- BUY: quote at or below the typical level ({r['mid']:,.2f})\n"
                     f"- HOLD: between the typical level and the expected high ({r['high']:,.2f})\n"
                     f"- NEGOTIATE: above the expected high")
        else:
            rules = (f"**AI model + price range** (gold)\n\n"
                     f"- BUY: quote below the expected low ({r['low']:,.2f}), or inside the range "
                     f"and the model expects a rise (50% or more)\n"
                     f"- HOLD: inside the range and the model expects a fall\n"
                     f"- NEGOTIATE: above the expected high ({r['high']:,.2f})")
        st.caption("How is this decided?", help=rules)
        st.markdown("\n".join(f"- {reason}" for reason in r["reasons"]))
        if saved["warning"]:
            st.warning(saved["warning"])

        cost = r["cost_vs_market"]
        unit = r["unit"].split("/")[1]
        gap = f"{abs(r['diff_pct']):.1%} {'below' if cost < 0 else 'above'} today's market price"
        money = f"{abs(cost):,.0f} USD {'saved' if cost < 0 else 'extra'} on {r['quantity']:g} {unit}"
        model_used = r["basis"] != "Price range only"
        summary = {
            "BUY": f"Buy now: the quote is {gap} ({money}), but the model expects gold to rise."
                   if model_used and cost > 0 else f"Good price: the quote is {gap} ({money}).",
            "HOLD": f"Wait: the quote is {gap} ({money}), and the model expects gold to fall."
                    if model_used else f"Fair price, but on the high side: the quote is {gap} ({money}). "
                                       f"Wait, or ask for about {r['mid']:,.2f}.",
            "NEGOTIATE": f"Too expensive: the quote is {gap} ({money}). Ask for about {r['mid']:,.2f}; "
                         f"anything above {r['high']:,.2f} is more than the market usually reaches in {HORIZON} days.",
        }
        st.markdown(f"**{summary[r['action']]}**")
        cost_label = "You save vs market" if cost < -0.5 else "Extra cost vs market" if cost > 0.5 else "Cost vs market"
        market_help = (f"Latest exchange (COMEX futures) price in the data, on {r['last_date']}.\n\n"
                       "Supplier premiums, taxes and delivery are not included.")
        diff_help = ("**Difference = (your quote - market price) / market price x 100**\n\n"
                     f"({r['quote']:,.2f} - {r['last_price']:,.2f}) / {r['last_price']:,.2f} x 100 = "
                     f"{signed(r['diff_pct'], '+.1%')}\n\n"
                     "- Negative: the quote is below the market price.\n"
                     "- Positive: the quote is above the market price.")
        cost_help = ("**Cost vs market = (your quote - market price) x quantity**\n\n"
                     f"({r['quote']:,.2f} - {r['last_price']:,.2f}) x {r['quantity']:g} {unit} = "
                     f"{signed(cost, '+,.2f')} USD\n\n"
                     "- Below market: shown as **You save vs market**.\n"
                     "- Above market: shown as **Extra cost vs market**.\n"
                     "- Within 0.50 USD: shown as **Cost vs market**.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market price", f"{r['last_price']:,.2f}", help=market_help)
        c2.metric("Your quote", f"{r['quote']:,.2f}")
        c3.metric("Difference", signed(r["diff_pct"], "+.1%"), help=diff_help)
        c4.metric(cost_label, f"{abs(cost):,.0f} USD", help=cost_help)
        f1, f2, _, _ = st.columns(4)
        f1.metric("Fair price", f"{r['mid']:,.2f}",
                  help=f"Typical price in {HORIZON} days = market price x (1 + median past {HORIZON}-day change)\n\n"
                       f"{r['last_price']:,.2f} x (1 {r['pct_mid']:+.2%}) = {r['mid']:,.2f}\n\n"
                       "A good target when negotiating.")
        f2.metric("Walk-away price", f"{r['high']:,.2f}",
                  help=f"Expected high = market price x (1 + 90th percentile of past {HORIZON}-day changes)\n\n"
                       f"{r['last_price']:,.2f} x (1 {r['pct_high']:+.2%}) = {r['high']:,.2f}\n\n"
                       "Only 10% of past weeks rose more than this. Above it: NEGOTIATE.")
        st.caption(f"Market prices are from {pd.Timestamp(r['last_date']):%d %b %Y}. If your quote is from a later date, "
                   "the market may have moved; upload recent prices for a fresher check.")

        st.markdown(range_bar(r, r["quote"], saved["probs"].get(saved["model"])), unsafe_allow_html=True)
        st.caption(f"Range from the 10th, 50th and 90th percentile of {r['n_changes']:,} past "
                   f"{HORIZON}-day price changes (2010 to 2021).")

        if saved["probs"]:
            st.subheader(f"Chance gold goes up in the next {HORIZON} trading days")
            auc = test_auc()
            for name, p in saved["probs"].items():
                n1, n2, n3 = st.columns([2, 4, 3])
                marker = " (used)" if name == saved["model"] else ""
                n1.write(f"**{LABELS.get(name, name)}**{marker}")
                n2.markdown(probability_bar(p), unsafe_allow_html=True)
                direction = "Up" if p >= 0.5 else "Down"
                score = f", reliability {auc[name]:.2f}" if name in auc else ""
                n3.write(f"{p:.1%} {direction}{score}")
            ups = sum(p >= 0.5 for p in saved["probs"].values())
            st.caption(f"{ups} of {len(saved['probs'])} models expect gold to rise. Reliability is the test AUC: "
                       "0.50 means guessing and 1.00 is perfect, so treat the recommendation as a guide.",
                       help="AUC measures how well a model ranks rising days above falling days on 701 test days "
                            "(2 Jan 2024 to 14 Oct 2026) that it never saw during training.")
        elif r["metal"] != "gold":
            st.info(f"No AI model for {r['metal']}: the recommendation compares the quote with "
                    f"{r['metal']}'s usual {HORIZON}-day price moves only.")