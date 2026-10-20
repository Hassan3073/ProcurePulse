"""Phase 2 collection: download COMEX metal futures closing prices from Yahoo Finance."""
import logging
import time

import yfinance as yf

from config import LOG_DIR, METALS, RAW_DIR, START_DATE

logging.basicConfig(filename=LOG_DIR / "collection.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def fetch_yahoo(ticker, start):
    """Return the daily closing price for one ticker, retrying up to 3 times."""
    for attempt in range(1, 4):
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
            if not df.empty:
                close = df["Close"].squeeze()
                close.index = close.index.tz_localize(None)
                return close
            logging.warning(f"{ticker}: empty response (attempt {attempt})")
        except Exception as e:
            logging.warning(f"{ticker}: attempt {attempt} failed: {e}")
        time.sleep(5)
    raise RuntimeError(f"Could not download {ticker} after 3 attempts")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, ticker in METALS.items():
        close = fetch_yahoo(ticker, START_DATE).rename(name)
        close.to_csv(RAW_DIR / f"{name}.csv", index_label="date")
        msg = (f"{name} ({ticker}): {len(close)} rows, "
               f"{close.index.min():%Y-%m-%d} to {close.index.max():%Y-%m-%d}, "
               f"{close.isna().sum()} missing")
        logging.info(msg)
        print(msg)


if __name__ == "__main__":
    main()