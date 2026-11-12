import numpy as np
import pandas as pd
import pytest

from src.recommendation.price_range import check_quote, expected_range, recommend


def fake_prices(n=2000, start="2015-01-01", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)


def test_range_is_ordered_and_centred_on_last_price():
    r = expected_range(fake_prices())
    assert r["low"] < r["mid"] < r["high"]
    assert r["low"] < r["last_price"] < r["high"]
    assert r["pct_low"] < 0 < r["pct_high"]


def test_range_uses_only_training_years():
    prices = fake_prices()
    changed = prices.copy()
    changed.loc["2022-01-01":] *= 3
    a, b = expected_range(prices), expected_range(changed)
    assert a["pct_low"] == b["pct_low"] and a["pct_high"] == b["pct_high"]
    assert b["last_price"] == pytest.approx(3 * a["last_price"])


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        expected_range(fake_prices(n=50))


RNG = {"low": 90.0, "mid": 100.0, "high": 110.0}


@pytest.mark.parametrize("quote, action", [(85, "BUY"), (95, "BUY"), (100, "BUY"), (105, "HOLD"), (115, "NEGOTIATE")])
def test_range_only_rule(quote, action):
    assert recommend(quote, RNG)[0] == action
    assert recommend(quote, RNG)[1] == "Price range only"


@pytest.mark.parametrize("quote, up_prob, action", [(105, 0.6, "BUY"), (95, 0.4, "HOLD"), (115, 0.9, "NEGOTIATE"), (85, 0.1, "BUY")])
def test_model_rule_for_gold(quote, up_prob, action):
    assert recommend(quote, RNG, up_prob)[0] == action
    assert recommend(quote, RNG, up_prob)[1] == "AI model + price range"


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        recommend(0, RNG)
    with pytest.raises(ValueError):
        recommend(100, RNG, up_prob=1.5)
    with pytest.raises(ValueError):
        check_quote("zinc", 100, prices=pd.DataFrame())


def test_check_quote_cost():
    s = fake_prices()
    prices = pd.DataFrame({"gold": s, "silver": s / 50, "copper": s / 500, "platinum": s / 2})
    r = check_quote("silver", 3.0, quantity=500, prices=prices)
    assert r["unit"] == "USD/oz"
    assert r["cost_vs_market"] == pytest.approx((3.0 - r["last_price"]) * 500)
    assert r["action"] in ("BUY", "HOLD", "NEGOTIATE")