"""Read-only project context, limitations, team, and policies."""
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMES = {"majority": "Majority baseline", "logreg": "Logistic regression",
         "random_forest": "Random forest", "hist_gb": "Gradient boosting", "lstm": "LSTM"}


def unavailable(name):
    st.info(f"{Path(name).name} is not available.")


@st.cache_data(persist=False, show_spinner=False)
def load_csv(name, columns=None, header=False):
    try:
        frame = pd.read_csv(ROOT / name, usecols=columns, nrows=0 if header else None)
        return frame if header or not frame.empty else None
    except Exception:
        return None


def table(name, columns=None, header=False):
    frame = load_csv(name, columns, header)
    if frame is None:
        unavailable(name)
    return frame


def dates(name):
    frame = table(name, ("date",))
    if frame is not None:
        values = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if not values.empty:
            return values.min(), values.max()
        unavailable(name)
    return None


@st.cache_data(persist=False, show_spinner=False)
def load_team():
    try:
        rows = []
        for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
            match = re.search(r"\*\*(.+?)\s+(24L-\d+)\*\*", line)
            if match and line.startswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                role = "Lead developer" if "(lead)" in cells[0] else "Developer"
                if len(cells) >= 3 and not cells[2].strip(" -\u2013\u2014"):
                    role = "Deliverable 1: " + cells[1]
                rows.append({"Name": match[1], "Roll number": match[2], "Role": role})
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


left, right = st.columns(2, gap="large")
latest = None
with left:
    with st.container(border=True):
        st.subheader("What is ProcurePulse?", anchor=False)
        st.write("ProcurePulse helps small and medium-sized businesses that buy metals, such as jewellers, "
                 "electronics assemblers and metal fabricators, decide whether a supplier's gold quote is fair.")
        st.write("It looks at gold prices and the economic signals that move them, predicts whether gold is likely "
                 "to rise or fall over the next 5 trading days, and turns that into a simple recommendation: "
                 "Buy, Hold or Negotiate.")
        st.subheader("Data", anchor=False)
        st.write("Gold | Silver | Copper | Platinum | US dollar index | WTI oil | US 10-year yield | CPI")
        period = dates("data/processed/cleaned.csv")
        if period:
            first, latest = period
            st.caption(f"Daily data from {first.year} (Yahoo Finance, FRED, BLS). "
                       f"Available dates: {first:%d %b %Y} to {latest:%d %b %Y}.")
        else:
            st.caption("Daily data (Yahoo Finance, FRED, BLS).")
        header = table("data/processed/features_train.csv", header=True)
        if header is not None:
            features = header.columns.difference(["date", "gold_5d_ahead", "direction_5d"])
            st.caption(f"{len(features)} input features. CPI is monthly and forward-filled to trading days.")
        st.subheader("Models", anchor=False)
        st.write("Majority baseline, Logistic regression, Random forest, Gradient boosting, LSTM (PyTorch).")

    with st.container(border=True):
        st.subheader("Limitations", anchor=False)
        comparison = table("results/tables/model_comparison.csv", ("model", "split", "roc_auc"))
        if comparison is not None:
            comparison["roc_auc"] = pd.to_numeric(comparison["roc_auc"], errors="coerce")
            test = comparison.loc[comparison["split"].eq("test") & comparison["roc_auc"].between(0, 1)]
            if not test.empty:
                score = test["roc_auc"].max()
                names = ", ".join(test.loc[test["roc_auc"].eq(score), "model"].map(lambda m: NAMES.get(m, str(m))))
                st.write(f"The price signal is weak: best test AUC is {score:.4f} ({names}); 0.50 is guessing.")
            else:
                unavailable("model_comparison.csv")
        st.write("Recommendations are a guide, not a guarantee. Only gold has AI predictions; silver, copper "
                 "and platinum use the historical 5-day price range only.")
        bias = table("results/analysis/bias_check.csv", ("model", "split", "pred_up_rate"))
        collapsed = pd.DataFrame()
        if bias is not None:
            bias["pred_up_rate"] = pd.to_numeric(bias["pred_up_rate"], errors="coerce")
            collapsed = bias.loc[bias["split"].eq("test") & bias["pred_up_rate"].isin([0, 1])]
            for row in collapsed.itertuples():
                st.write(f"{NAMES.get(row.model, row.model)} predicts {'up' if row.pred_up_rate == 1 else 'down'} "
                         f"on every test day (predicted up-rate: {row.pred_up_rate:.0%}).")
        if not collapsed.empty:
            train_period = dates("data/processed/features_train.csv")
            test_period = dates("data/processed/features_test.csv")
            if train_period and test_period:
                st.write(f"Models were trained on {train_period[0].year} to {train_period[1].year} prices. "
                         f"The test period covers {test_period[0].year} to {test_period[1].year}.")
                prices = table("data/processed/cleaned.csv", ("date", "gold"))
                if prices is not None:
                    price_dates = pd.to_datetime(prices["date"], errors="coerce")
                    gold = pd.to_numeric(prices["gold"], errors="coerce")
                    train_gold = gold.loc[price_dates.between(*train_period)]
                    test_gold = gold.loc[price_dates.between(*test_period)]
                    if test_gold.median() > train_gold.max() and collapsed["model"].isin(["random_forest", "hist_gb"]).any():
                        st.write("Gold's median test-period price is above the training maximum. "
                                 "This shift is a likely reason the tree models predict only down: they cannot extrapolate.")
        else:
            st.write("Market conditions can move beyond the training range, making past relationships less reliable.")

with right:
    with st.container(border=True):
        st.subheader("How a quote check works", anchor=False)
        st.write("Market data -> Clean -> Build features -> Scale -> Models predict -> "
                 "Compare quote with range -> Buy / Hold / Negotiate")
        image_path = ROOT / "report/figures/architecture/system_pipeline.png"
        try:
            st.image(image_path.read_bytes(), width="stretch", alt="ProcurePulse data, prediction and quote-check pipeline")
        except Exception:
            unavailable(image_path)
        built_in = f"prices through {latest:%d %b %Y}" if latest is not None else "the latest available dataset"
        st.markdown(f"- **Built-in data:** {built_in}.\n"
                    "- **Upload CSV:** columns date, gold, silver, copper, platinum, dxy, wti, us10y, cpi; "
                    "at least 100 trading days. A template can be downloaded on the Quote Check page.\n"
                    "- **Latest prices:** downloads about one year of public prices from Yahoo Finance and BLS. "
                    "Needs an internet connection; downloaded prices are used in memory rather than saved as a dataset.")
    with st.container(border=True):
        st.subheader("Team", anchor=False)
        team = load_team()
        if team is None:
            unavailable("README.md")
        else:
            st.table(team, alt="Project team and roles from README")
        st.subheader("Legal", anchor=False)
        st.write("Please read the Terms of Service and Privacy Policy below before using the recommendations.")

with st.expander("Terms of Service", expanded=False):
    st.write("ProcurePulse is a FAST-NUCES university project and does not provide financial advice. "
             "Its outputs are estimates with no guarantee of accuracy or future performance. "
             "The software is provided as is, without warranty. "
             "Do not make purchasing decisions on its recommendations alone. "
             "Check current prices, supplier terms, and your business requirements independently.")
with st.expander("Privacy Policy", expanded=False):
    st.write("Quotes and uploaded files are processed in server memory, not saved as files by the app or sent to price-data providers. "
             "Uploaded CSVs are temporary inputs and are discarded when the corresponding session and in-memory caches are cleared. "
             "Latest prices contacts Yahoo Finance and the US Bureau of Labor Statistics for public price data without sending your quotes or uploads. "
             "Those providers receive ordinary connection information, such as the server's IP address. "
             "The project adds no accounts, tracking cookies, or analytics. "
             "Streamlit may use security cookies needed to run the app.")
