import math
from vol_greeks.blackscholes import BlackScholesInputs, calculate_black_scholes_price, calculate_black_scholes_d1_d2, _T_EXP



def calculate_vega(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Vega of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): The inputs required for Black-Scholes calculation.

    Returns:
    float: The Vega of the option.
    """
  # Vega is typically expressed per 1% change in volatility
    if inputs.T <= 0 or inputs.sigma <= 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    d1, _ = calculate_black_scholes_d1_d2(inputs)
    vega = inputs.S * math.exp(-inputs.q * inputs.T) * math.sqrt(inputs.T) * 1/(math.sqrt(2*math.pi)) * math.exp(-d1**2 / 2)
    return vega
