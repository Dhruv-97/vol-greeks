from __future__ import annotations

import os
import math
from typing import Dict, Any, List, Optional, Iterable
from datetime import datetime, timezone
import requests

from dotenv import load_dotenv

# --- your library imports ---
from vol_greeks.chain import ChainRow, time_till_expiry, getIVPerRow

load_dotenv()
API_KEY = os.getenv("MARKETDATA_API_KEY")

OPTIONS_CHAIN_URL = "https://api.marketdata.app/v1/options/chain/{symbol}"
STOCK_QUOTE_URL   = "https://api.marketdata.app/v1/stocks/quotes/{symbol}"


# ------------------------------ HTTP Helpers ------------------------------ #

def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        print("STATUS:", resp.status_code)
        print("RESPONSE:", resp.text)
        resp.raise_for_status()
    data = resp.json()
    return data


# ------------------------------ Fetchers ------------------------------ #

def fetch_underlying_price(symbol: str) -> float:
    """
    Fetch a single underlying quote. Returns a reasonable price proxy.
    """
    if not API_KEY:
        raise RuntimeError("MARKETDATA_API_KEY not set")

    url = STOCK_QUOTE_URL.format(symbol=symbol)
    data = _get(url, {"token": API_KEY})

    # Try a few likely fields in order
    for key in ("mid", "last", "close", "price", "ask", "bid"):
        if key in data and data[key] is not None:
            return float(data[key])

    # Some responses use arrays; try first element
    for key in ("mid", "last", "close", "price", "ask", "bid"):
        v = data.get(key)
        if isinstance(v, list) and v:
            return float(v[0])

    raise KeyError(f"No usable price field in quote response: {data}")


def fetch_chain(symbol: str, expiration: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch the raw options chain 'arrays' payload.
    - symbol: 'AAPL', 'SPX', etc.
    - expiration: optional 'YYYY-MM-DD'. If None, API returns a default (often nearest) expiry.
    """
    if not API_KEY:
        raise RuntimeError("MARKETDATA_API_KEY not set")

    url = OPTIONS_CHAIN_URL.format(symbol=symbol)
    params = {"token": API_KEY}
    if expiration:
        params["expiration"] = expiration  # API expects YYYY-MM-DD

    data = _get(url, params)
    if data.get("s") != "ok":
        raise RuntimeError(f"Chain fetch error: {data.get('errmsg', 'Unknown error')}")

    return data


# ------------------------------ Parsing & q Inference ------------------------------ #

def _epoch_to_dt(ts: int) -> datetime:
    # MarketData sends epoch seconds; convert to naive UTC datetime
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)

def _infer_q_via_parity(
    S: float, T: float, r: float, call_mid_by_k: Dict[float, float], put_mid_by_k: Dict[float, float]
) -> float:
    """
    Put–call parity with continuous dividend yield q:
      C - P = S e^{-qT} - K e^{-rT}
      => q = -(1/T) * ln((C - P + K e^{-rT}) / S)
    Choose strike closest to S for stability.
    """
    common = sorted(set(call_mid_by_k).intersection(put_mid_by_k))
    if not common or T <= 0 or S <= 0:
        raise ValueError("Not enough info to infer q")

    K = min(common, key=lambda k: abs(k - S))
    C = call_mid_by_k[K]
    P = put_mid_by_k[K]
    num = C - P + K * math.exp(-r * T)
    if num <= 0:
        raise ValueError("Parity numerator non-positive; cannot infer q")
    return -(1.0 / T) * math.log(num / S)

def rows_from_marketdata_chain(
    data: Dict[str, Any],
    r: float = 0.04,
    q: Optional[float] = None,
) -> List[ChainRow]:
    """
    Convert MarketData 'arrays' chain (your pasted shape) → List[ChainRow].
    - Uses data['underlyingPrice'][0] for S
    - Uses data['expiration'][0] for expiry → T
    - Infers q via parity if not provided; else uses provided q
    """
    required = ["optionSymbol", "side", "strike", "mid", "expiration", "underlyingPrice"]
    for k in required:
        if k not in data or not isinstance(data[k], list) or not data[k]:
            raise ValueError(f"Chain JSON missing/empty list field '{k}'")

    n = len(data["optionSymbol"])
    # Sanity check length consistency for key arrays
    for k in ["side", "strike", "mid", "expiration"]:
        if len(data[k]) != n:
            raise ValueError(f"Inconsistent array length for '{k}'")

    expiry_ts = data["expiration"][0]
    expiry_dt = _epoch_to_dt(expiry_ts)
    T = time_till_expiry(expiry_dt)

    S = float(data["underlyingPrice"][0])

    bids = data.get("bid", [None] * n)
    asks = data.get("ask", [None] * n)
    lasts = data.get("last", [None] * n)

    call_mid_by_k: Dict[float, float] = {}
    put_mid_by_k: Dict[float, float] = {}
    for i in range(n):
        side = data["side"][i]
        k = float(data["strike"][i])
        mid_i = data["mid"][i]
        if mid_i is None:
            continue
        mid_f = float(mid_i)
        if side == "call":
            call_mid_by_k[k] = mid_f
        elif side == "put":
            put_mid_by_k[k] = mid_f

    if q is None:
        try:
            q = _infer_q_via_parity(S, T, r, call_mid_by_k, put_mid_by_k)
        except Exception:
            q = 0.0  # fallback

    rows: List[ChainRow] = []
    for i in range(n):
        side = data["side"][i]
        k = float(data["strike"][i])

        mid = data["mid"][i]
        mid_f = float(mid) if mid is not None else None

        bid_f = float(bids[i]) if (isinstance(bids, list) and i < len(bids) and bids[i] is not None) else None
        ask_f = float(asks[i]) if (isinstance(asks, list) and i < len(asks) and asks[i] is not None) else None
        last_f = float(lasts[i]) if (isinstance(lasts, list) and i < len(lasts) and lasts[i] is not None) else None

        row = ChainRow(
            option_type=side,
            S=S,
            K=k,
            T=T,  # your pipeline recomputes from expiry anyway, but this keeps ChainRow consistent
            r=r,
            q=q,
            expiry=expiry_dt,
            mid_option_market_price=mid_f,
            bid=bid_f,
            ask=ask_f,
            last=last_f,
        )
        rows.append(row)

    return rows


# ------------------------------ End-to-end helper ------------------------------ #

def fetch_solve_chain(
    symbol: str,
    expiration: Optional[str] = None,  # 'YYYY-MM-DD'
    r: float = 0.04,
    q: Optional[float] = None,
) -> List[ChainRow]:
    """
    1) Fetch chain from API
    2) Parse to rows (inferring q if needed)
    3) Solve IV and Greeks for each row
    """
    raw = fetch_chain(symbol, expiration)
    rows = rows_from_marketdata_chain(raw, r=r, q=q)
    solved = getIVPerRow(rows)
    return solved


# ------------------------------ CLI demo ------------------------------ #

if __name__ == "__main__":
    # Example: fetch AAPL for a specific expiry (the “14 DTE” sample you pasted is 2025-11-21)
    symbol = "AAPL"
    expiration = "2025-11-21"  # or None for API default/nearest

    solved_rows = fetch_solve_chain(symbol, expiration, r=0.04, q=None)

    # Print a few lines
    for rrow in solved_rows[:10]:
        print(
            f"{symbol} {rrow.option_type.upper()} {rrow.K:.2f} "
            f"T={rrow.T:.4f} r={rrow.r:.3f} q={rrow.q:.3f} "
            f"mid={rrow.mid_option_market_price} iv={rrow.iv} "
            f"Δ={rrow.greeks.get('delta') if rrow.greeks else None}"
        )
