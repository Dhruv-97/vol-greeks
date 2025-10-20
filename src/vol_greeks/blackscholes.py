from __future__ import annotations

from math import log, sqrt, exp
from dataclasses import dataclass
from typing import Literal
import math
from statistics import NormalDist

N = NormalDist()

_T_EXP = 1e-4
_SIGMA_EPS = 1e-12

@dataclass
class BlackScholesInputs:
    option_type: Literal['call', 'put']
    S: float          # Current stock price
    K: float          # Strike price
    T: float          # Time to expiration in years
    r: float          # Risk-free interest rate (annualized)
    sigma: float      # Volatility of the underlying stock (annualized)
    q: float = 0.0    # Dividend yield (annualized)
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
    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and T, sigma must be greater than zero.")
    
    if inputs.T <= _T_EXP or inputs.sigma <= _SIGMA_EPS:
        T_eff = max(inputs.T, _T_EXP)
        sigma_eff = max(inputs.sigma, _SIGMA_EPS)
    else:
        T_eff = inputs.T
        sigma_eff = inputs.sigma

    d1 = (log(inputs.S/inputs.K) + (inputs.r - inputs.q + .5 * sigma_eff ** 2)*T_eff) / (sigma_eff * sqrt(T_eff))
    d2 = d1 - sigma_eff * sqrt(T_eff)
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
    if inputs.T < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and T must be greater than zero.")
    

    if inputs.T <= _T_EXP:
        if inputs.option_type == 'call':
            return max(0.0, inputs.S - inputs.K)
        if inputs.option_type == 'put':
            return max(0.0, inputs.K - inputs.S)
    
    if inputs.sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    d1, d2 = calculate_black_scholes_d1_d2(inputs)

    if inputs.option_type == 'call':
        return inputs.S * exp(-inputs.q * inputs.T) * N.cdf(d1) - inputs.K * exp(-inputs.r * inputs.T) * N.cdf(d2)
    if inputs.option_type == 'put':
        return inputs.K * exp(-inputs.r * inputs.T) * N.cdf(-d2) - inputs.S * exp(-inputs.q * inputs.T) * N.cdf(-d1)