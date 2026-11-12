import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.csv"

METALS = ["gold", "silver", "copper", "platinum"]
UNITS = {"gold": "USD/oz", "silver": "USD/oz", "copper": "USD/lb", "platinum": "USD/oz"}
HORIZON = 5
TRAIN_END = "2021-12-31"
PERCENTILES = (0.10, 0.50, 0.90)


def load_prices(path=CLEANED_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Restore the frozen data into data/processed/")
    df = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    return df[METALS]


def expected_range(prices, train_end=TRAIN_END, horizon=HORIZON):
    prices = prices.dropna()
    hist = prices.loc[:train_end]
    change = (hist.shift(-horizon) / hist - 1).dropna()
    if len(change) < 100:
        raise ValueError(f"Need at least 100 past {horizon}-day changes, got {len(change)}")
    p_low, p_mid, p_high = change.quantile(list(PERCENTILES))
    last_price = float(prices.iloc[-1])
    return {
        "last_date": prices.index[-1].date().isoformat(),
        "last_price": last_price,
        "low": last_price * (1 + p_low),
        "mid": last_price * (1 + p_mid),
        "high": last_price * (1 + p_high),
        "pct_low": float(p_low),
        "pct_mid": float(p_mid),
        "pct_high": float(p_high),
        "n_changes": int(len(change)),
    }

def recommend(quote, rng, up_prob=None):
    if quote <= 0:
        raise ValueError("quote must be a positive price")
    if up_prob is not None and not 0 <= up_prob <= 1:
        raise ValueError("up_prob must be between 0 and 1")

    low, mid, high = rng["low"], rng["mid"], rng["high"]
    basis = "Price range only" if up_prob is None else "AI model + price range"

    if quote > high:
        return "NEGOTIATE", basis, [
            f"Quote {quote:,.2f} is above the expected {HORIZON}-day high ({high:,.2f}).",
            f"Ask for a price near the typical level ({mid:,.2f}).",
        ]
    if quote < low:
        return "BUY", basis, [
            f"Quote {quote:,.2f} is below the expected {HORIZON}-day low ({low:,.2f}).",
            "This is cheaper than the market usually reaches.",
        ]
    if up_prob is None:
        if quote <= mid:
            return "BUY", basis, [
                f"Quote {quote:,.2f} is inside the expected range, at or below the typical level ({mid:,.2f}).",
            ]
        return "HOLD", basis, [
            f"Quote {quote:,.2f} is inside the expected range but above the typical level ({mid:,.2f}).",
            "Wait, or ask for a small discount.",
        ]
    if up_prob >= 0.5:
        return "BUY", basis, [
            f"Quote {quote:,.2f} is inside the expected range ({low:,.2f} to {high:,.2f}).",
            f"The model gives a {up_prob:.1%} chance that the price rises, so waiting may cost more.",
        ]
    return "HOLD", basis, [
        f"Quote {quote:,.2f} is inside the expected range ({low:,.2f} to {high:,.2f}).",
        f"The model gives a {1 - up_prob:.1%} chance that the price falls, so waiting may be cheaper.",
    ]


def check_quote(metal, quote, quantity=1.0, up_prob=None, prices=None):
    if metal not in METALS:
        raise ValueError(f"metal must be one of {METALS}, got {metal!r}")
    if prices is None:
        prices = load_prices()
    rng = expected_range(prices[metal])
    action, basis, reasons = recommend(quote, rng, up_prob)
    market = rng["last_price"]
    return {
        "metal": metal,
        "unit": UNITS[metal],
        "quote": quote,
        "quantity": quantity,
        "action": action,
        "basis": basis,
        "reasons": reasons,
        "diff_pct": (quote - market) / market,
        "cost_vs_market": (quote - market) * quantity,
        **rng,
    }


def main():
    parser = argparse.ArgumentParser(description="Check a supplier quote against the expected price range.")
    parser.add_argument("--metal", choices=METALS, required=True)
    parser.add_argument("--quote", type=float, required=True)
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument("--up-prob", type=float, default=None, help="model probability of a rise (gold only)")
    args = parser.parse_args()

    r = check_quote(args.metal, args.quote, args.quantity, args.up_prob)
    print(f"{r['metal'].upper()} ({r['unit']}), prices up to {r['last_date']}")
    print(f"  Market price : {r['last_price']:,.2f}")
    print(f"  Expected range ({HORIZON} days): {r['low']:,.2f} / {r['mid']:,.2f} / {r['high']:,.2f}"
          f"  ({r['pct_low']:+.1%} / {r['pct_mid']:+.1%} / {r['pct_high']:+.1%}, from {r['n_changes']} past changes)")
    print(f"  Quote        : {r['quote']:,.2f}  ({r['diff_pct']:+.1%} vs market)")
    print(f"  Cost vs market for {r['quantity']:g} units: {r['cost_vs_market']:+,.2f} USD")
    print(f"  RECOMMENDATION: {r['action']}  [{r['basis']}]")
    for reason in r["reasons"]:
        print(f"   - {reason}")


if __name__ == "__main__":
    main()