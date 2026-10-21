"""Shared paths and the data contract. Owned by Hassan Ashraf 24L3073; read-only for Sharjeel Shahid 24L3055."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LOG_DIR = ROOT / "data_scripts" / "logs"
FIG_RAW_DIR = ROOT / "report" / "figures" / "raw"
FIG_EDA_DIR = ROOT / "report" / "figures" / "eda"

START_DATE = "2010-01-01"
METALS = {"gold": "GC=F", "silver": "SI=F", "copper": "HG=F", "platinum": "PL=F"}
MARKET_SIGNALS = {"dxy": "DX-Y.NYB", "wti": "CL=F"}
FRED_SIGNALS = {"us10y": "DGS10", "cpi": "CPIAUCSL"}

METAL_COLS = list(METALS)
PRICE_COLS = METAL_COLS + list(MARKET_SIGNALS)   # columns with a price (returns make sense)
ALL_COLS = PRICE_COLS + list(FRED_SIGNALS)