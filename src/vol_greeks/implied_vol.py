from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Tuple
from vol_greeks.blackscholes import BlackScholesInputs, calculate_black_scholes_d1_d2, calculate_black_scholes_price


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
    vega = inputs.S * math.sqrt(inputs.T) * 1/(math.sqrt(2*math.pi)) * math.exp(-d1**2 / 2)
    return vega

def implied_volatility(
    option_market_price: float, 
    inputs: BlackScholesInputs, 
    max_iterations: int = 200, 
    epsilon_price: float = 1e-5, 
    epsilon_vol: float = 1e-5,
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0,
    ) -> float:
    """
    Calculate the implied volatility of a European option using the Black-Scholes model.
    Parameters:
    option_market_price (float): The market price of the option.
    inputs (BlackScholesInputs): The inputs required for Black-Scholes calculation, except sigma.
    max_iterations (int): Maximum number of iterations for the root-finding algorithm.
    epsilon_price (float): Acceptable error in option price.
    epsilon_vol (float): Acceptable error in volatility.
    sigma_low (float): Lower bound for volatility search.
    sigma_high (float): Upper bound for volatility search.
    Returns:
    float: The implied volatility of the option.
    """
    if inputs.T <=0 or inputs.S <= 0 or inputs.K <=0 or option_market_price <=0:
        raise ValueError("All inputs must be positive and greater than zero")
    discount_factor = math.exp(-inputs.r * inputs.T)
    intrinsic_value = max(0.0, inputs.S - inputs.K*discount_factor) if inputs.option_type == 'call' else max(0.0, inputs.K*discount_factor -inputs.S)
    upper = inputs.S if inputs.option_type == 'call' else discount_factor*inputs.K
    if not (intrinsic_value - 1e-12 <= option_market_price <= upper + 1e-12): #marketPrice of the option is not inside the intrinsic value of the call or is worth more than the stock itself 
        raise ValueError("Market price is out of bounds given the other inputs.")
    return bisection(inputs, option_market_price, sigma_low, sigma_high, epsilon_price, epsilon_vol, max_iterations)
    
def sigma_adjustment_option_price_comparison(sigma: float, inputs: BlackScholesInputs, option_market_price: float) -> float:
    new_inputs = BlackScholesInputs(
        option_type=inputs.option_type,
        S=inputs.S,
        K=inputs.K,
        T=inputs.T,
        r=inputs.r,
        sigma=sigma
    )
    return calculate_black_scholes_price(new_inputs) - option_market_price

def bisection(
    inputs:BlackScholesInputs,
    option_market_price: float, 
    sigma_low: float, sigma_high: float, 
    epsilon_price: float = 1e-5, epsilon_vol: float = 1e-5, 
    max_iterations: int=200):

    #check if you need to widen the bounds
    temp_low_price_diff = sigma_adjustment_option_price_comparison(sigma_low, inputs, option_market_price)
    if temp_low_price_diff == 0:
        return sigma_low
    temp_high_price_diff = sigma_adjustment_option_price_comparison(sigma_high, inputs, option_market_price)
    if temp_high_price_diff == 0:
        return sigma_high
    if temp_low_price_diff * temp_high_price_diff > 0:
        for new_bound in (10.0,20.0):
            f_new = sigma_adjustment_option_price_comparison(new_bound, inputs, option_market_price)
            if temp_low_price_diff * f_new <= 0:
                sigma_high = new_bound
                temp_high_price_diff = f_new
                break
        else:
            return float('nan') #could not find valid bounds
    for i in range(max_iterations):    
        mid_sigma = (sigma_high+sigma_low)/2
        price_diff = sigma_adjustment_option_price_comparison(mid_sigma, inputs, option_market_price)
        if abs(price_diff) < epsilon_price:
            return mid_sigma
        elif price_diff < 0:
            sigma_low = mid_sigma
        else:
            sigma_high = mid_sigma
    return (sigma_high + sigma_low)/2