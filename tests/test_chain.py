import math
from datetime import datetime, timedelta
import pytest

from vol_greeks.blackscholes import BlackScholesInputs, calculate_black_scholes_price
from vol_greeks.chain import (
    ChainRow,
    getCurrentPrice,
    time_till_expiry,
    getIVPerRow,
)


# ---------------------------
# Helpers
# ---------------------------

def _years_from_now(years: float) -> datetime:
    # deterministic enough for these tests
    return datetime.now() + timedelta(seconds=years * 365.25 * 24 * 3600)

def _fair_price(option_type, S, K, T, r, q, sigma):
    return calculate_black_scholes_price(
        BlackScholesInputs(
            option_type=option_type, S=S, K=K, T=T, r=r, q=q, sigma=sigma
        )
    )


# ---------------------------
# getCurrentPrice behavior
# ---------------------------

def test_getCurrentPrice_precedence_mid_override():
    expiry = _years_from_now(0.25)

    # mid_option_market_price has highest precedence
    row = ChainRow(
        option_type="call", S=100, K=100, T=0.25, r=0.02, q=0.0,
        expiry=expiry, mid_option_market_price=1.23,
        bid=1.0, ask=3.0, last=2.9, mid=2.1
    )
    assert getCurrentPrice(row) == 1.23

def test_getCurrentPrice_uses_last_when_no_mid_override():
    expiry = _years_from_now(0.25)

    row = ChainRow(
        option_type="put", S=100, K=100, T=0.25, r=0.02, q=0.0,
        expiry=expiry, mid_option_market_price=None,
        last=2.8, bid=1.0, ask=3.0, mid=2.1
    )
    # NOTE: your current implementation ignores `mid` field entirely
    assert getCurrentPrice(row) == 2.8

def test_getCurrentPrice_uses_bid_ask_mid_when_last_missing():
    expiry = _years_from_now(0.25)

    row = ChainRow(
        option_type="call", S=100, K=100, T=0.25, r=0.02, q=0.0,
        expiry=expiry, mid_option_market_price=None,
        bid=1.50, ask=2.50, last=None
    )
    assert math.isclose(getCurrentPrice(row), 2.00, abs_tol=1e-12)

def test_getCurrentPrice_falls_back_to_ask_then_bid():
    expiry = _years_from_now(0.25)

    row = ChainRow(
        option_type="call", S=100, K=100, T=0.25, r=0.02, q=0.0,
        expiry=expiry, mid_option_market_price=None,
        bid=None, ask=2.70, last=None
    )
    assert getCurrentPrice(row) == 2.70

    row = ChainRow(
        option_type="put", S=100, K=100, T=0.25, r=0.02, q=0.0,
        expiry=expiry, mid_option_market_price=None,
        bid=2.60, ask=None, last=None
    )
    assert getCurrentPrice(row) == 2.60

def test_getCurrentPrice_raises_when_no_prices_present():
    expiry = _years_from_now(0.25)

    row = ChainRow(
        option_type="call", S=100, K=100, T=0.25, r=0.02, q=0.0,
        expiry=expiry, mid_option_market_price=None
    )
    with pytest.raises(ValueError):
        getCurrentPrice(row)


# ---------------------------
# time_till_expiry
# ---------------------------

def test_time_till_expiry_future_is_positive():
    now = datetime.now()
    expiry = now + timedelta(days=365)
    t = time_till_expiry(expiry, now)
    assert math.isclose(t, 1.0, rel_tol=0, abs_tol=2e-3)  # ~1 year allowing small float wiggle

def test_time_till_expiry_past_is_zero():
    now = datetime.now()
    expiry = now - timedelta(days=1)
    assert time_till_expiry(expiry, now) == 0.0


# ---------------------------
# getIVPerRow happy paths
# ---------------------------

@pytest.mark.parametrize("option_type,S,K,r,q,T,sigma", [
    ("call", 100, 100, 0.02, 0.00, 0.50, 0.20),
    ("put",  100, 120, 0.03, 0.01, 0.75, 0.35),
    ("call", 420, 400, 0.00, 0.02, 0.25, 0.10),
    ("put",  250, 200, 0.01, 0.00, 1.00, 0.50),
])
def test_getIVPerRow_roundtrip(option_type, S, K, r, q, T, sigma):
    expiry = _years_from_now(T)
    fair = _fair_price(option_type, S, K, T, r, q, sigma)

    row = ChainRow(
        option_type=option_type, S=S, K=K, T=0.0,  # T will be recomputed
        r=r, q=q, expiry=expiry, mid_option_market_price=fair
    )

    solved = getIVPerRow([row])[0]
    assert solved.iv is not None
    assert math.isclose(solved.iv, sigma, abs_tol=5e-6)

    assert solved.greeks is not None
    assert 'delta' in solved.greeks and 'gamma' in solved.greeks and 'vega' in solved.greeks

def test_getIVPerRow_uses_last_when_mid_override_missing():
    # Build price via last only
    option_type, S, K, r, q, T, sigma = "call", 150, 140, 0.01, 0.00, 0.3, 0.25
    expiry = _years_from_now(T)
    fair = _fair_price(option_type, S, K, T, r, q, sigma)

    row = ChainRow(
        option_type=option_type, S=S, K=K, T=0.0,
        r=r, q=q, expiry=expiry,
        last=fair, mid_option_market_price=None
    )

    solved = getIVPerRow([row])[0]
    assert solved.iv is not None
    assert math.isclose(solved.iv, sigma, abs_tol=5e-6)

def test_getIVPerRow_missing_all_prices_sets_iv_none_and_still_sets_greeks():
    expiry = _years_from_now(0.5)

    row = ChainRow(
        option_type="call", S=100, K=100, T=0.0,
        r=0.02, q=0.00, expiry=expiry, mid_option_market_price=None
    )

    solved = getIVPerRow([row])[0]
    # getCurrentPrice raises -> caught in try/except -> iv becomes None
    assert solved.iv is None
    assert solved.greeks is not None  # computed with fallback sigma=0.2


# ---------------------------
# Bounds & bad quotes
# ---------------------------

@pytest.mark.parametrize("option_type,S,K,r,q,T", [
    ("call", 100,  90, 0.02, 0.00, 0.50),  # ITM call -> positive intrinsic
    ("put",  100, 110, 0.02, 0.00, 0.50),  # ITM put  -> positive intrinsic
    ("call", 150, 120, 0.01, 0.02, 1.00),  # changed: ITM, not OTM
])
def test_getIVPerRow_rejects_price_below_intrinsic(option_type, S, K, r, q, T):
    expiry = _years_from_now(T)
    disc_r = math.exp(-r*T); disc_q = math.exp(-q*T)
    if option_type == "call":
        intrinsic = max(S*disc_q - K*disc_r, 0.0)
    else:
        intrinsic = max(K*disc_r - S*disc_q, 0.0)

    assert intrinsic > 0.0  # sanity: we truly test "below intrinsic"

    bad_price = intrinsic - 1e-4  # slightly below the hard bound
    row = ChainRow(option_type=option_type, S=S, K=K, T=0.0, r=r, q=q,
                   expiry=expiry, mid_option_market_price=bad_price)

    solved = getIVPerRow([row])[0]
    assert solved.iv is None
    assert solved.greeks is not None


# ---------------------------
# Expired rows
# ---------------------------

def test_getIVPerRow_expired_row_iv_none_and_greeks_defined():
    # expiry in the past
    expiry = datetime.now() - timedelta(days=1)
    row = ChainRow(
        option_type="put", S=100, K=105, T=0.0,
        r=0.02, q=0.00, expiry=expiry, mid_option_market_price=2.0
    )
    solved = getIVPerRow([row])[0]
    # Time gets clamped to 0.0 in your time_till_expiry
    assert solved.T == 0.0
    # IV solver should fail (near-expiry handling) -> iv None
    assert solved.iv is None
    # Greeks still present (your greeks handle near-zero T by short-circuiting)
    assert solved.greeks is not None
    # Some quick sanity on ranges
    assert -1.0 <= solved.greeks["delta"] <= 1.0


# ---------------------------
# Regression: Greeks reflect solved IV
# ---------------------------

def test_greeks_use_solved_iv_not_guess():
    option_type, S, K, r, q, T_true, sigma_true = "call", 200, 180, 0.01, 0.02, 0.4, 0.42
    expiry = _years_from_now(T_true)
    fair = _fair_price(option_type, S, K, T_true, r, q, sigma_true)

    row = ChainRow(
        option_type=option_type, S=S, K=K, T=0.0,
        r=r, q=q, expiry=expiry, mid_option_market_price=fair
    )

    solved = getIVPerRow([row])[0]
    assert solved.iv is not None and solved.greeks is not None
    assert math.isclose(solved.iv, sigma_true, abs_tol=5e-6)

    # Reprice with solved IV to confirm consistency
    back_price = _fair_price(option_type, S, K, solved.T, r, q, solved.iv)
    assert math.isclose(back_price, fair, abs_tol=7e-4)
