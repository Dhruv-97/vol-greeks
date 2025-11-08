# src/scripts/fetch_chain.py
from __future__ import annotations

import os
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from vol_greeks.chain import ChainRow, time_till_expiry, getIVPerRow  # noqa: F401

load_dotenv()

API_KEY = os.getenv("MARKETDATA_API_KEY")

OPTIONS_CHAIN_URL = "https://api.marketdata.app/v1/options/chain/{symbol}"
STOCK_QUOTE_URL   = "https://api.marketdata.app/v1/stocks/quotes/{symbol}"

DEFAULT_R = 0.04


# ------------------------------ HTTP helpers ------------------------------ #

def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=20)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from {url}: {e}")


# ------------------------------ Utilities ------------------------------ #

def _epoch_to_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)

def _moneyness(S: float, K: float) -> float:
    return abs(K / S - 1.0)

def _intrinsic(option_type: str, S: float, K: float) -> float:
    if option_type == "call":
        return max(S - K, 0.0)
    else:
        return max(K - S, 0.0)


# ------------------------------ Public fetchers ------------------------------ #

def fetch_underlying_price(symbol: str) -> float:
    if not API_KEY:
        raise RuntimeError("MARKETDATA_API_KEY not set")

    url = STOCK_QUOTE_URL.format(symbol=symbol)
    data = _get(url, {"token": API_KEY})

    for key in ("mid", "last", "close", "price", "ask", "bid"):
        v = data.get(key)
        if v is not None and not isinstance(v, list):
            return float(v)

    for key in ("mid", "last", "close", "price", "ask", "bid"):
        v = data.get(key)
        if isinstance(v, list) and v and v[0] is not None:
            return float(v[0])

    raise KeyError(f"No usable price field in quote response: {list(data.keys())}")


def fetch_chain(symbol: str, expiration: Optional[str] = None) -> Dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("MARKETDATA_API_KEY not set")

    url = OPTIONS_CHAIN_URL.format(symbol=symbol)
    params = {"token": API_KEY}
    if expiration:
        params["expiration"] = expiration

    data = _get(url, params)
    if data.get("s") != "ok":
        raise RuntimeError(f"Chain fetch error: {data.get('errmsg') or 'unknown'}")
    return data


# ------------------------------ Carry inference ------------------------------ #

def _infer_carry_r_minus_q_for_expiry(
    S: float,
    T: float,
    r: float,
    call_mid_by_k: Dict[float, float],
    put_mid_by_k: Dict[float, float],
) -> Optional[float]:
    """
    Median-of-window parity estimator for (r - q):

      C - P = S e^{-qT} - K e^{-rT}
      => r - q = r + (1/T) * ln((C - P + K e^{-rT}) / S)

    Restrict to |K/S - 1| <= 5% or the 5 closest-to-ATM strikes to reduce noise.
    """
    if T <= 0 or S <= 0 or not call_mid_by_k or not put_mid_by_k:
        return None

    common = sorted(set(call_mid_by_k).intersection(put_mid_by_k), key=lambda k: abs(k - S))
    if not common:
        return None

    window = [k for k in common if _moneyness(S, k) <= 0.05] or common[:5]

    vals: List[float] = []
    for K in window:
        C = call_mid_by_k.get(K)
        P = put_mid_by_k.get(K)
        if C is None or P is None or C <= 0 or P < 0:
            continue
        num = C - P + K * math.exp(-r * T)
        if num <= 0:
            continue
        carry = r + (1.0 / T) * math.log(num / S)
        vals.append(carry)

    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]  # median


# ------------------------------ Chain conversion ------------------------------ #

def rows_from_marketdata_chain(
    data: Dict[str, Any],
    r: float = DEFAULT_R,
    q: Optional[float] = None,
    *,
    intrinsic_floor: bool = True,
    use_shared_spot: bool = False,
    shared_spot: Optional[float] = None,
) -> List[ChainRow]:
    """
    Convert MarketData 'arrays' into List[ChainRow].

    Features:
      • Per-row T from each row’s expiration epoch
      • Per-expiry q via parity → q = max(0, min(0.08, r - carry))
      • Optional intrinsic-floor on mid to stabilize IV near parity edge
      • Spot selection:
          - default: per-row underlyingPrice[i] if available, else underlyingPrice[0]
          - use_shared_spot=True: force a single spot (passed via `shared_spot` or quotes endpoint by caller)
    """
    required = ["optionSymbol", "side", "strike", "mid", "expiration", "underlyingPrice"]
    for k in required:
        if k not in data or not isinstance(data[k], list) or not data[k]:
            raise ValueError(f"Chain JSON missing/empty list field '{k}'")

    n = len(data["optionSymbol"])
    for k in ("side", "strike", "mid", "expiration"):
        if len(data[k]) != n:
            raise ValueError(f"Inconsistent array length for '{k}'")

    # Expiry buckets
    by_expiry: Dict[int, List[int]] = {}
    for i in range(n):
        exp_ts = int(data["expiration"][i])
        by_expiry.setdefault(exp_ts, []).append(i)

    # Spot selection
    def spot_for(i: int) -> float:
        if use_shared_spot and shared_spot is not None:
            return float(shared_spot)
        up = data.get("underlyingPrice")
        if isinstance(up, list):
            if i < len(up) and up[i] is not None:
                return float(up[i])
            if up and up[0] is not None:
                return float(up[0])
        raise KeyError("No usable underlying price in payload")

    # Per-expiry q via parity
    q_per_expiry: Dict[int, float] = {}
    if q is not None:
        for exp_ts in by_expiry:
            q_per_expiry[exp_ts] = float(q)
    else:
        for exp_ts, idxs in by_expiry.items():
            S_vals = sorted(spot_for(i) for i in idxs)
            S_rep = S_vals[len(S_vals) // 2]
            T = time_till_expiry(_epoch_to_dt(exp_ts))

            call_mid_by_k: Dict[float, float] = {}
            put_mid_by_k: Dict[float, float] = {}
            for i in idxs:
                side = data["side"][i]
                K = float(data["strike"][i])
                mid = data["mid"][i]
                if mid is None or mid <= 0:
                    continue
                if side == "call":
                    call_mid_by_k[K] = float(mid)
                else:
                    put_mid_by_k[K] = float(mid)

            carry = _infer_carry_r_minus_q_for_expiry(S_rep, T, r, call_mid_by_k, put_mid_by_k)
            q_hat = 0.0 if carry is None else (r - carry)
            q_hat = max(0.0, min(0.08, q_hat))  # clamp for equities/ETFs
            q_per_expiry[exp_ts] = q_hat

    bids = data.get("bid", [None] * n)
    asks = data.get("ask", [None] * n)
    lasts = data.get("last", [None] * n)

    rows: List[ChainRow] = []
    for i in range(n):
        side = data["side"][i]
        K = float(data["strike"][i])

        expiry_ts = int(data["expiration"][i])
        expiry_dt = _epoch_to_dt(expiry_ts)
        T_i = time_till_expiry(expiry_dt)

        S_i = spot_for(i)

        mid = data["mid"][i]
        mid_f = float(mid) if mid is not None else None

        # Stabilize near intrinsic: small floor to avoid pathological IV inversion
        if intrinsic_floor and mid_f is not None:
            floor = _intrinsic(side, S_i, K)
            if mid_f < floor + 0.01:
                mid_f = floor + 0.01

        bid_f = float(bids[i]) if (isinstance(bids, list) and i < len(bids) and bids[i] is not None) else None
        ask_f = float(asks[i]) if (isinstance(asks, list) and i < len(asks) and asks[i] is not None) else None
        last_f = float(lasts[i]) if (isinstance(lasts, list) and i < len(lasts) and lasts[i] is not None) else None

        rows.append(
            ChainRow(
                option_type=side,
                S=S_i,
                K=K,
                T=T_i,
                r=r,
                q=q_per_expiry.get(expiry_ts, 0.0),
                expiry=expiry_dt,
                mid_option_market_price=mid_f,
                bid=bid_f,
                ask=ask_f,
                last=last_f,
            )
        )
    return rows


# ------------------------------ Convenience ------------------------------ #

def fetch_solve_chain(
    symbol: str,
    expiration: Optional[str] = None,
    r: float = DEFAULT_R,
    q: Optional[float] = None,
    *,
    use_shared_spot: bool = False,
    shared_spot: Optional[float] = None,
) -> List[ChainRow]:
    raw = fetch_chain(symbol, expiration)
    rows = rows_from_marketdata_chain(
        raw, r=r, q=q, use_shared_spot=use_shared_spot, shared_spot=shared_spot
    )
    return getIVPerRow(rows)
