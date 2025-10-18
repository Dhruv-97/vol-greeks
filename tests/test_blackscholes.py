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

