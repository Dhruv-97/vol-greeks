from __future__ import annotations
from typing import Literal

import math
from math import log, exp, sqrt
from vol_greeks.blackscholes import BlackScholesInputs, calculate_black_scholes_price, calculate_black_scholes_d1_d2, _T_EXP, _SIGMA_EPS
from statistics import NormalDist

N = NormalDist()


def calculate_vega(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Vega of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): The inputs required for Black-Scholes calculation.

    Returns:
    float: The Vega of the option.
    """
    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    if inputs.T <= _T_EXP:
        return 0.0
    d1, _ = calculate_black_scholes_d1_d2(inputs)
    vega = inputs.S * math.exp(-inputs.q * inputs.T) * math.sqrt(inputs.T) * 1/(math.sqrt(2*math.pi)) * math.exp(-d1**2 / 2)
    return vega


def calculate_delta(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Delta of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): S,T,r,sigma,K,option_type

    Returns:
    float: The Delta of the option.
    """

    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    
    if inputs.T <= _T_EXP:
        if inputs.option_type == 'call':
            return 1.0 if inputs.S > inputs.K else (0.5 if inputs.S == inputs.K else 0.0)
        if inputs.option_type == 'put':
            return -1.0 if inputs.S < inputs.K else (-0.5 if inputs.S == inputs.K else 0.0)

    d1, d2 = calculate_black_scholes_d1_d2(inputs)
    delta = 0.0

    if inputs.option_type == 'call':
        delta = exp(-inputs.q*inputs.T) * N.cdf(d1)
    elif inputs.option_type == 'put':
        delta = -exp(-inputs.q*inputs.T) * (N.cdf(-d1))

    return delta

def calculate_gamma(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Gamma of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): S,T,r,sigma,K,option_type

    Returns:
    float: The Gamma of the option.
    """
    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    if inputs.T <= _T_EXP:
        return 0.0

    d1, d2 = calculate_black_scholes_d1_d2(inputs)

    gamma = exp(-inputs.q*inputs.T)/(inputs.S*inputs.sigma*sqrt(inputs.T)) * 1/(math.sqrt(2*math.pi)) * math.exp(-d1**2 / 2)
    return gamma

def calculate_theta(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Theta of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): S,T,r,sigma,K,option_type

    Returns:
    float: The Theta of the option.
    """
    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    if inputs.T <= _T_EXP:
        return 0.0

    d1, d2 = calculate_black_scholes_d1_d2(inputs)

    theta = 0.0

    if inputs.option_type == 'call':
        term1 = - (inputs.S * inputs.sigma * exp(-inputs.q*inputs.T))/(2*sqrt(inputs.T)) * 1/(math.sqrt(2*math.pi)) * math.exp(-d1**2 / 2)
        theta = (term1 - inputs.r * inputs.K*exp(-inputs.r*inputs.T)*N.cdf(d2) + inputs.q*inputs.S*exp(-inputs.q*inputs.T)*N.cdf(d1)) / 365.0
    elif inputs.option_type == 'put':
        term1 = - (inputs.S * inputs.sigma * exp(-inputs.q*inputs.T))/(2*sqrt(inputs.T)) * 1/(math.sqrt(2*math.pi)) * math.exp(-d1**2 / 2)
        theta = (term1 + inputs.r * inputs.K*exp(-inputs.r*inputs.T)*N.cdf(-d2) - inputs.q*inputs.S*exp(-inputs.q*inputs.T)*N.cdf(-d1)) / 365.0
    return theta

def calculate_rho(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Rho of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): S,T,r,sigma,K,option_type

    Returns:
    float: The Rho of the option.
    """
    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    if inputs.T <= _T_EXP:
        return 0.0
    d1, d2 = calculate_black_scholes_d1_d2(inputs)

    rho = 0.0

    if inputs.option_type == 'call':
        rho = inputs.K * inputs.T * exp(-inputs.r*inputs.T) * N.cdf(d2)
    elif inputs.option_type == 'put':
        rho = -inputs.K * inputs.T * exp(-inputs.r*inputs.T) * N.cdf(-d2)

    return rho

def calculate_charm(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Charm of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): S,T,r,sigma,K,option_type

    Returns:
    float: The Charm of the option.
    """
    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    if inputs.T <= _T_EXP:
        return 0.0


    d1, d2 = calculate_black_scholes_d1_d2(inputs)
    disc_q = math.exp(-inputs.q * inputs.T)
    phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2*math.pi)

    numer = 2*(inputs.r - inputs.q)*inputs.T - d2*inputs.sigma*math.sqrt(inputs.T)
    denom = 2*inputs.T*inputs.sigma*math.sqrt(inputs.T)
    slope = disc_q * phi * (numer / denom)

    if inputs.option_type == 'call':
        return -inputs.q * disc_q * N.cdf(d1) + slope
    else:
        return +inputs.q * disc_q * N.cdf(-d1) - slope

def calculate_vanna(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Vanna of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): S,T,r,sigma,K,option_type

    Returns:
    float: The Vanna of the option.
    """
    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    if inputs.T <= _T_EXP:
        return 0.0

    d1, d2 = calculate_black_scholes_d1_d2(inputs)
    vega = calculate_vega(inputs)             # per unit sigma
    return - (d2 / inputs.sigma) * vega


def calculate_volga(inputs: BlackScholesInputs) -> float:
    """
    Calculate the Volga of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): S,T,r,sigma,K,option_type

    Returns:
    float: The Volga of the option.
    """
    if inputs.T < 0 or inputs.sigma < 0 or inputs.S <= 0 or inputs.K <= 0:
        raise ValueError("All inputs must be positive and greater than zero")
    if inputs.T <= _T_EXP:
        return 0.0
    d1, d2 = calculate_black_scholes_d1_d2(inputs)

    volga = inputs.S * exp(-inputs.q*inputs.T) * 1/(math.sqrt(2*math.pi)) * math.exp(-d1**2 / 2) * sqrt(inputs.T) * d1 * d2 / inputs.sigma
    return volga

def calculate_dividend_rho(inputs: BlackScholesInputs) -> float:
    if inputs.S <= 0 or inputs.K <= 0 or inputs.T < 0 or inputs.sigma < 0:
        raise ValueError("Inputs must satisfy S>0, K>0, T>=0, sigma>=0.")
    if inputs.T <= _T_EXP:
        return 0.0
    d1, _ = calculate_black_scholes_d1_d2(inputs)
    disc_q = math.exp(-inputs.q*inputs.T)
    if inputs.option_type == 'call':
        return -inputs.S * inputs.T * disc_q * N.cdf(d1)
    else:
        return  inputs.S * inputs.T * disc_q * N.cdf(-d1)



def BlackScholesGreeks(inputs: BlackScholesInputs) -> dict[str, float]:
    """
    Calculate all the primary Greeks of a European option using the Black-Scholes model.

    Parameters:
    inputs (BlackScholesInputs): S,T,r, sigma,K, option_type

    Returns:
    dict[str, float]: A dict containing all the Greeks of the option.
    """
    greeks = {
        'delta': calculate_delta(inputs),
        'gamma': calculate_gamma(inputs),
        'vega': calculate_vega(inputs)/100,
        'theta': calculate_theta(inputs),
        'rho': calculate_rho(inputs)/100,
        'charm': calculate_charm(inputs),
        'vanna': calculate_vanna(inputs),
        'volga': calculate_volga(inputs),
        'dividend_rho': calculate_dividend_rho(inputs)/100
    }
    return greeks