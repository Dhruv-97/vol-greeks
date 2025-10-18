from __future__ import annotations

from math import log, sqrt, exp
from dataclasses import dataclass
from typing import Literal
import math
from statistics import NormalDist

N = NormalDist()

@dataclass
class BlackScholesInputs:
    option_type: Literal['call', 'put']
    S: float          # Current stock price
    K: float          # Strike price
    T: float          # Time to expiration in years
    r: float          # Risk-free interest rate (annualized)
    sigma: float      # Volatility of the underlying stock (annualized)
def calculate_black_scholes_d1_d2(inputs: BlackScholesInputs) -> tuple[float, float]:
    """
    Calculate the Black-Scholes option price.

    Parameters:
    option_type (str): 'call' or 'put'
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)

    Returns:
    float: The Black-Scholes option price
    """
    epsilon = 1e-6
    if inputs.T <= 0 or inputs.sigma <= 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and T, sigma must be greater than zero.")
    d1 = (log(inputs.S/inputs.K) + (inputs.r + .5 * inputs.sigma ** 2)*inputs.T) / (inputs.sigma * sqrt(inputs.T))
    d2 = d1 - inputs.sigma * sqrt(inputs.T)
    return d1, d2

def calculate_black_scholes_price(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Black-Scholes option price.

    Parameters:
    option_type (str): 'call' or 'put'
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying stock (annualized)

    Returns:
    float: The Black-Scholes option price
    """
    d1, d2 = calculate_black_scholes_d1_d2(inputs)
    if math.isnan(d2) or math.isnan(d1):
        return float('nan')
    if inputs.option_type == 'call':
        return inputs.S * N.cdf(d1) - inputs.K * exp(-inputs.r * inputs.T) * N.cdf(d2)
    if inputs.option_type == 'put':
        return inputs.K * exp(-inputs.r * inputs.T) * N.cdf(-d2) - inputs.S * N.cdf(-d1)