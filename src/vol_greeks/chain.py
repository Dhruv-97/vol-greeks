from dataclasses import dataclass
from datetime import datetime
from .blackscholes import BlackScholesInputs
from .greeks import BlackScholesGreeks
from math import log, sqrt, exp
from .implied_vol import implied_volatility
from typing import List, Iterable, Literal, Optional

@dataclass
class ChainRow:
    option_type: Literal['call', 'put']
    S: float  # Underlying price
    K: float  # Strike price
    T: float  # Time to expiration in years
    r: float  # Risk-free interest rate
    q: float  # Dividend yield
    # oi: int  # Open interest
    # volume: int  # Volume
    expiry: datetime  # Expiration date of the option
    mid_option_market_price: float  # Market price of the option

    #optional fields
    bid: Optional[float] = None  # Bid price
    ask: Optional[float] = None  # Ask price
    last: Optional[float] = None  # Last traded price
    mid: Optional[float] = None  # Mid price


    #calculated fields
    iv: Optional[float] = None  # Implied volatility
    greeks: Optional[dict[str, float]] = None  # Greeks dictionary

def getCurrentPrice(chain_row: ChainRow) -> float:
    """
    Get the current price of the option from the ChainRow.
    """

    if chain_row.mid_option_market_price is not None:
        return chain_row.mid_option_market_price
    elif chain_row.last is not None:
        return chain_row.last
    elif chain_row.ask is not None and chain_row.bid is not None:
        return (chain_row.ask + chain_row.bid) / 2
    elif chain_row.ask is not None:
        return chain_row.ask
    elif chain_row.bid is not None:
        return chain_row.bid
    else:
        raise ValueError("No price information available in ChainRow")

def time_till_expiry(expiry: datetime, current_date: datetime = datetime.now()) -> float:
    """
    Calculate the time till expiry in years.
    """
    return max(0.0, (expiry - current_date).total_seconds() / (365.25 * 24 * 3600))

def getIVPerRow(chain_rows: Iterable[ChainRow]) -> List[ChainRow]:
    """
    Calculate and set the implied volatility for each ChainRow in the iterable.
    """
    res = []
    for row in chain_rows:
        row.T = time_till_expiry(row.expiry)
        inputs = BlackScholesInputs(
            S=row.S,
            K=row.K,
            T=row.T,
            r=row.r,
            q=row.q,
            option_type=row.option_type,
            sigma=0.2  # Initial guess for volatility
        )
        try: 
            iv = implied_volatility(
                option_market_price=getCurrentPrice(row),
                inputs=inputs
            )
        except Exception as e:
            iv = None
        row.iv = iv
        inputs.sigma = iv if iv is not None else 0.2
        row.greeks = BlackScholesGreeks(inputs)
        res.append(row)
    return res