import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGES = [
    ("quote_check.py", "Quote Check"),
    ("market.py", "Market"),
    ("model_insights.py", "Model Insights"),
    ("about.py", "About"),
]


def welcome():
    st.title("ProcurePulse")
    st.info("The app pages are being built. They will appear in the top bar once merged.")


st.set_page_config(page_title="ProcurePulse", layout="wide")

pages = [st.Page(APP_DIR / "views" / name, title=title) for name, title in PAGES
         if (APP_DIR / "views" / name).exists()]
if not pages:
    pages = [st.Page(welcome, title="Home")]

st.navigation(pages, position="top").run()

st.divider()
st.caption("ProcurePulse is a university AI project. Recommendations are estimates, not financial advice.")