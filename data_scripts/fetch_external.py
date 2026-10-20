"""Phase 2 collection: download external signals (US dollar index, crude oil, 10-year yield, CPI).

The yield and CPI come from FRED. If FRED is unreachable (it blocks some networks),
the script switches to backup sources: Yahoo ^TNX for the yield, the BLS API for CPI.
"""
import io
import logging
import time

import pandas as pd
import requests

from config import FRED_SIGNALS, LOG_DIR, MARKET_SIGNALS, RAW_DIR, START_DATE
from fetch_metals import fetch_yahoo

logging.basicConfig(filename=LOG_DIR / "collection.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BACKUPS = {"us10y": ("Yahoo", "^TNX"), "cpi": ("BLS", "CUSR0000SA0")}


def with_retries(download, label, attempts):
    """Run download(), retrying on any error; log each failure."""
    for attempt in range(1, attempts + 1):
        try:
            return download()
        except Exception as e:
            logging.warning(f"{label}: attempt {attempt} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"Could not download {label} after {attempts} attempts")


def fetch_fred(series_id, start):
    """Return one FRED series from the start date."""
    def download():
        response = requests.get(FRED_URL.format(series_id), timeout=45,
                                headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        series = pd.read_csv(io.StringIO(response.text), index_col=0,
                             parse_dates=True, na_values=".").iloc[:, 0]
        return pd.to_numeric(series, errors="coerce")

    series = with_retries(download, f"FRED {series_id}", attempts=2)
    return series[series.index >= start]


def fetch_bls(series_id, start):
    """Return one monthly BLS series; the free API allows 10 years per request."""
    first, last = int(start[:4]), pd.Timestamp.today().year
    values = {}
    for year in range(first, last + 1, 10):
        def download():
            payload = {"seriesid": [series_id], "startyear": str(year),
                       "endyear": str(min(year + 9, last))}
            response = requests.post(BLS_URL, json=payload, timeout=60)
            response.raise_for_status()
            body = response.json()
            if body.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(body.get("message"))
            return body["Results"]["series"][0]["data"]

        for item in with_retries(download, f"BLS {series_id}", attempts=3):
            if item["period"].startswith("M") and item["period"] != "M13":  # M13 = annual average
                month = pd.Timestamp(f"{item['year']}-{item['period'][1:]}-01")
                values[month] = pd.to_numeric(item["value"], errors="coerce")
    return pd.Series(values, dtype=float).sort_index()


def fetch_backup(name, start):
    """Download a FRED series from its backup source."""
    source, series_id = BACKUPS[name]
    if source == "Yahoo":
        series = fetch_yahoo(series_id, start)
        if series.median() > 20:  # older Yahoo data quoted ^TNX as yield x 10
            series = series / 10
    else:
        series = fetch_bls(series_id, start)
    return series, f"{source} {series_id}"


def save(series, name, source):
    """Save one series as data/raw/<name>.csv and log a summary line."""
    series = series.rename(name)
    series.to_csv(RAW_DIR / f"{name}.csv", index_label="date")
    msg = (f"{name} ({source}): {len(series)} rows, "
           f"{series.index.min():%Y-%m-%d} to {series.index.max():%Y-%m-%d}, "
           f"{series.isna().sum()} missing")
    logging.info(msg)
    print(msg)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, ticker in MARKET_SIGNALS.items():
        save(fetch_yahoo(ticker, START_DATE), name, ticker)
    for name, series_id in FRED_SIGNALS.items():
        try:
            series, source = fetch_fred(series_id, START_DATE), f"FRED {series_id}"
        except RuntimeError:
            logging.warning(f"{name}: FRED unavailable, switching to backup source")
            print(f"{name}: FRED unavailable, using backup source...")
            series, source = fetch_backup(name, START_DATE)
        save(series, name, source)


if __name__ == "__main__":
    main()