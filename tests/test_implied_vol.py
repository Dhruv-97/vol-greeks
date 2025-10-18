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