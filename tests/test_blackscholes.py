import math
import pytest
from vol_greeks.blackscholes import BlackScholesInputs, calculate_black_scholes_d1_d2, calculate_black_scholes_price


def test_calculate_black_scholes_d1_d2():
    call_inputs = BlackScholesInputs(
        option_type='call',
        S = 100,
        K = 100,
        T = 1,
        r = 0.05,
        sigma = 0.2
    )
    put_inputs = BlackScholesInputs(
        option_type='put',
        S = 100,
        K = 100,
        T = 1,
        r = 0.05,
        sigma = 0.2
    )

    call_price = calculate_black_scholes_price(call_inputs)
    put_price = calculate_black_scholes_price(put_inputs)


    #test prices
    rhs = call_inputs.S - call_inputs.K * math.exp(-call_inputs.r * call_inputs.T)
    assert abs((call_price - put_price) - rhs) < 1e-6

def test_call_vs_put_sanity():
    inputs = BlackScholesInputs(
        option_type='call',
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2
    )
    call_price = calculate_black_scholes_price(inputs)
    inputs.option_type = 'put'
    put_price = calculate_black_scholes_price(inputs)
    assert put_price < call_price

def test_increasing_vol_increases_price():
    call_inputs = BlackScholesInputs(
        option_type='call',
        S = 100,
        K = 100,
        T = 1,
        r = 0.05,
        sigma = 0.1
    )
    prev = calculate_black_scholes_price(call_inputs)
    cur = 0
    for i in range(2,20):
        call_inputs.sigma = 0.1 * i
        cur = calculate_black_scholes_price(call_inputs)
        assert cur > prev
        prev = cur
def test_invalid_inputs():
    invalid_inputs = [
        BlackScholesInputs(option_type='call', S=-100, K=100, T=1, r=0.05, sigma=0.2),
        BlackScholesInputs(option_type='put', S=100, K=-100, T=1, r=0.05, sigma=0.2),
        BlackScholesInputs(option_type='call', S=100, K=100, T=-1, r=0.05, sigma=0.2),
        BlackScholesInputs(option_type='put', S=100, K=100, T=1, r=0.05, sigma=-0.2),
        BlackScholesInputs(option_type='call', S=0, K=100, T=1, r=0.05, sigma=0.2),
    ]
    for inputs in invalid_inputs:
        with pytest.raises(ValueError):
            calculate_black_scholes_price(inputs)

def test_put_call_parity_no_divs():
    xC = BlackScholesInputs("call", 100, 100, 0.5, 0.02, 0.20)
    xP = BlackScholesInputs("put",  100, 100, 0.5, 0.02, 0.20)
    C = calculate_black_scholes_price(xC)
    P = calculate_black_scholes_price(xP)
    rhs = xC.S - xC.K * math.exp(-xC.r * xC.T)
    assert abs((C - P) - rhs) < 1e-8

@pytest.mark.parametrize("sigma1,sigma2", [(0.10, 0.30), (0.30, 0.60)])
def test_price_increases_with_vol(sigma1, sigma2):
    x1 = BlackScholesInputs("call", 100, 95, 1.0, 0.01, sigma1)
    x2 = BlackScholesInputs("call", 100, 95, 1.0, 0.01, sigma2)
    assert calculate_black_scholes_price(x2) > calculate_black_scholes_price(x1)

def test_sigma_to_zero_limit_matches_discounted_intrinsic_call():
    S, K, T, r = 105.0, 100.0, 0.5, 0.02
    tiny = 1e-8
    x = BlackScholesInputs("call", S, K, T, r, tiny)
    C = calculate_black_scholes_price(x)
    intrinsic = max(S - K*math.exp(-r*T), 0.0)
    assert abs(C - intrinsic) < 1e-4

def test_bounds_no_arbitrage():
    S, K, T, r, sig = 100, 100, 0.75, 0.03, 0.25
    xC = BlackScholesInputs("call", S, K, T, r, sig)
    xP = BlackScholesInputs("put",  S, K, T, r, sig)
    C = calculate_black_scholes_price(xC)
    P = calculate_black_scholes_price(xP)
    disc = math.exp(-r*T)
    assert max(S - disc*K, 0.0) - 1e-12 <= C <= S + 1e-12
    assert max(disc*K - S, 0.0) - 1e-12 <= P <= disc*K + 1e-12

def test_d1_d2_basic_sanity():
    x = BlackScholesInputs("call", 100, 100, 1.0, 0.0, 0.2)
    d1, d2 = calculate_black_scholes_d1_d2(x)
    assert d1 - d2 == pytest.approx(x.sigma*math.sqrt(x.T))

