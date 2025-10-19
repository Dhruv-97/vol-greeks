import math
import pytest
from vol_greeks.blackscholes import BlackScholesInputs, calculate_black_scholes_d1_d2, calculate_black_scholes_price
from vol_greeks.implied_vol import implied_volatility, calculate_vega

def test_calculate_implied_volatility():
    inputs = BlackScholesInputs(
        option_type='call',
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2  # initial guess, not used in implied vol calculation
    )
    market_price = calculate_black_scholes_price(inputs)
    implied_vol = implied_volatility(market_price, inputs)
    assert abs(implied_vol - inputs.sigma) < 1e-4



def test_iv_roundtrip_call():
    S,K,T,r,sig = 100,100,1.0,0.05,0.20
    price = calculate_black_scholes_price(BlackScholesInputs("call", S,K,T,r,sig))
    ihat = implied_volatility(price, BlackScholesInputs("call", S,K,T,r, 0.3))
    assert abs(ihat - sig) < 1e-6

def test_iv_roundtrip_put():
    S,K,T,r,sig = 100,120,0.5,0.02,0.35
    price = calculate_black_scholes_price(BlackScholesInputs("put", S,K,T,r,sig))
    ihat = implied_volatility(price, BlackScholesInputs("put", S,K,T,r, 0.1))
    assert abs(ihat - sig) < 1e-5

def test_iv_rejects_bad_price_below_intrinsic_call():
    S,K,T,r = 100,100,0.5,0.02
    # intrinsic for call is S - K e^{-rT} > 0 here
    bad = 0.0
    with pytest.raises(ValueError):
        implied_volatility(bad, BlackScholesInputs("call", S,K,T,r, 0.2))

def test_iv_widens_bracket_when_needed(monkeypatch):
    # Force a weird price so initial hi=5 fails; widening to 10 should bracket
    S,K,T,r,sig = 100,80,0.05,0.01, 1.5
    price = calculate_black_scholes_price(BlackScholesInputs("call", S,K,T,r,sig))
    # call with a distant K/T yields large IV; our widening should handle it
    ihat = implied_volatility(price, BlackScholesInputs("call", S,K,T,r, 0.2))
    assert ihat == pytest.approx(sig, rel=1e-3, abs=1e-3)

@pytest.mark.parametrize("sig", [0.05, 0.10, 0.30, 0.80])
def test_iv_roundtrip_parametric(sig):
    S,K,T,r = 120,100,0.75,0.00
    price = calculate_black_scholes_price(BlackScholesInputs("put", S,K,T,r,sig))
    vega = calculate_vega(BlackScholesInputs("put", S,K,T,r,sig))
    sigma_tol = max(2e-6, 10 * 1e-5 / max(vega, 1e-12))  # 10× epsilon_price / vega
    ihat = implied_volatility(price, BlackScholesInputs("put", S,K,T,r, 0.2))
    assert abs(ihat - sig) < sigma_tol*1.05

def test_vega_increases_with_T():
    x_short = BlackScholesInputs("call", 100, 100, 0.1, 0.01, 0.2)
    x_long = BlackScholesInputs("call", 100, 100, 1.0, 0.01, 0.2)
    vega_short = calculate_vega(x_short)
    vega_long = calculate_vega(x_long)
    assert vega_long > vega_short

def test_vega_positive_for_T_gt_0():
    x = BlackScholesInputs("call", 100, 100, 0.5, 0.01, 0.2)
    assert calculate_vega(x) > 0.0

def test_price_increases_with_small_sigma_step_proportional_to_vega():
    x = BlackScholesInputs("call", 100, 100, 0.5, 0.01, 0.2)
    base = calculate_black_scholes_price(x)
    eps = 1e-4
    x_eps = BlackScholesInputs("call", x.S, x.K, x.T, x.r, x.sigma + eps)
    bumped = calculate_black_scholes_price(x_eps)
    # Finite-difference dPrice/dSigma ~ Vega
    fd = (bumped - base) / eps
    vega = calculate_vega(x)
    assert fd == pytest.approx(vega, rel=5e-3)  # allow small numerical tolerance

def test_vega_nan_on_bad_inputs():
    bad = BlackScholesInputs("call", 0.0, 100, 1.0, 0.0, 0.2)
    with pytest.raises(ValueError):
        calculate_vega(bad)

def test_near_expiry_behaves_sane():
    S,K,T,r,sig = 100, 100, 1/365, 0.01, 0.50  # ~1 day
    price = calculate_black_scholes_price(BlackScholesInputs("call", S,K,T,r,sig))
    ihat = implied_volatility(price, BlackScholesInputs("call", S,K,T,r, 0.2))
    assert abs(ihat - sig) < 5e-4  # looser tolerance near T~0

def test_deep_otm_put_roundtrip():
    S,K,T,r,sig = 100, 200, 1.0, 0.02, 0.55
    price = calculate_black_scholes_price(BlackScholesInputs("put", S,K,T,r,sig))
    ihat = implied_volatility(price, BlackScholesInputs("put", S,K,T,r, 0.1))
    assert abs(ihat - sig) < 1e-4

def test_upper_lower_bounds_are_hard_limits():
    S,K,T,r = 100, 100, 0.5, 0.02
    disc = math.exp(-r*T)
    call_lower = max(S - disc*K, 0.0)
    put_upper  = disc*K
    with pytest.raises(ValueError):
        implied_volatility(call_lower - 1e-3, BlackScholesInputs("call", S,K,T,r, 0.2))
    with pytest.raises(ValueError):
        implied_volatility(put_upper + 1e-3,  BlackScholesInputs("put",  S,K,T,r, 0.2))
