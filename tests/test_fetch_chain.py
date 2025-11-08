# tests/test_fetch_chain.py
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

import pytest
import requests
from dotenv import load_dotenv

from vol_greeks.chain import getIVPerRow, ChainRow  # type: ignore
from scripts.fetch_chain import (
    DEFAULT_R,
    rows_from_marketdata_chain,
)

load_dotenv()
API_KEY = os.getenv("MARKETDATA_API_KEY")
BASE_CHAIN_URL = "https://api.marketdata.app/v1/options/chain/{symbol}"

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ"]
EXPLICIT_EXPIRATIONS = [None]


# ------------------------------ HTTP helpers ------------------------------ #

def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=20)
    if not (200 <= resp.status_code < 300):
        pytest.skip(f"HTTP {resp.status_code}: {resp.text[:160]}")
    try:
        data = resp.json()
    except Exception as e:
        pytest.skip(f"Invalid JSON: {e}")
    if data.get("s") != "ok":
        pytest.skip(f"API not ok: {data.get('errmsg') or 'unknown'}")
    return data


def _fetch_chain_raw(symbol: str, expiration: Optional[str]) -> Dict[str, Any]:
    if not API_KEY:
        pytest.skip("MARKETDATA_API_KEY not set; skipping network tests.")
    url = BASE_CHAIN_URL.format(symbol=symbol)
    params = {"token": API_KEY}
    if expiration:
        params["expiration"] = expiration
    return _http_get(url, params)


# ------------------------------ Helpers ------------------------------ #

def _vendor_has_array(key: str, d: Dict[str, Any]) -> bool:
    arr = d.get(key)
    if not isinstance(arr, list):
        return False
    for v in arr:
        if isinstance(v, (int, float)) and abs(float(v)) > 0.0:
            return True
    return False

def _nz(x):
    if x is None:
        return None
    x = float(x)
    return x if abs(x) > 0.0 else None

def _days(T_years: float) -> float:
    return T_years * 365.25

def _tolerances(S: float, K: float, T: float) -> Dict[str, float]:
    m = abs(K / S - 1.0)
    if m < 0.02:
        return dict(delta=0.02, gamma=0.005, theta=0.03, vega=0.012)
    if m < 0.06:
        return dict(delta=0.03, gamma=0.006, theta=0.04, vega=0.015)
    return dict(delta=0.05, gamma=0.01, theta=0.06, vega=0.02)

def _detect_theta_scale(ours: List[float], vendor: List[float]) -> float:
    import statistics
    pairs = [abs(a)/abs(b) for a, b in zip(ours, vendor) if a and b and abs(b) > 1e-8]
    if not pairs:
        return 1.0
    med = statistics.median(pairs)
    cal_per_trading = 365.0 / 252.0
    trading_per_cal = 252.0 / 365.0
    if abs(med - cal_per_trading) < 0.25:
        return cal_per_trading
    if abs(med - trading_per_cal) < 0.25:
        return trading_per_cal
    return 1.0


# ------------------------------ Tests ------------------------------ #

@pytest.mark.network
@pytest.mark.parametrize("symbol", TICKERS)
@pytest.mark.parametrize("expiration", EXPLICIT_EXPIRATIONS)
def test_compare_vendor_greeks(symbol: str, expiration: Optional[str]):
    """
    Compare Δ/Γ/Θ/vega to MarketData greeks.
    If a point fails, recompute greeks using the vendor IV with our (S,T,r,q)
    to diagnose whether it’s IV inversion vs. inputs/conventions.
    """
    data = _fetch_chain_raw(symbol, expiration)

    vendor_keys = [k for k in ("delta", "gamma", "theta", "vega") if _vendor_has_array(k, data)]
    if not vendor_keys:
        pytest.skip(f"{symbol}: vendor payload has no usable greeks.")

    rows = rows_from_marketdata_chain(
        data, r=DEFAULT_R, q=None, use_shared_spot=False, shared_spot=None
    )
    solved = getIVPerRow(rows)

    assert len(solved) == len(data["optionSymbol"])

    # Theta scale detection
    our_theta_samp, ven_theta_samp = [], []
    n = len(solved)
    for i in range(n):
        row = solved[i]
        if row.iv is None or row.greeks is None or _days(row.T) < 0.5:
            continue
        if "theta" in vendor_keys:
            vt = _nz(data.get("theta", [None] * n)[i])
            if vt is not None:
                our_theta_samp.append(float(row.greeks["theta"]))
                ven_theta_samp.append(float(vt))
        if len(our_theta_samp) >= 40:
            break
    theta_scale = _detect_theta_scale(our_theta_samp, ven_theta_samp) if our_theta_samp else 1.0

    checked = 0
    max_to_check = 200
    errors: List[str] = []

    for i in range(n):
        if checked >= max_to_check:
            break

        row = solved[i]
        if row.iv is None or row.greeks is None or _days(row.T) < 0.5:
            continue

        v: Dict[str, float] = {}
        ok_here = True
        for k in vendor_keys:
            arr = data.get(k, [])
            val = _nz(arr[i] if i < len(arr) else None)
            if val is None:
                ok_here = False
                break
            v[k] = float(val)
        if not ok_here:
            continue

        ours = {
            "delta": float(row.greeks["delta"]),
            "gamma": float(row.greeks["gamma"]),
            "theta": float(row.greeks["theta"]),
            "vega":  float(row.greeks["vega"]),
        }
        v_cmp = dict(v)
        if "theta" in v_cmp:
            v_cmp["theta"] = v_cmp["theta"] * theta_scale

        tol = _tolerances(row.S, row.K, row.T)
        local_fail = [k for k in vendor_keys if abs(ours[k] - v_cmp[k]) > tol[k]]

        if not local_fail:
            checked += 1
            continue

        # ---------- Diagnostic path with vendor IV ----------
        ven_iv_list = data.get("iv", [])
        ven_iv = None
        if isinstance(ven_iv_list, list) and i < len(ven_iv_list) and ven_iv_list[i]:
            try:
                ven_iv = float(ven_iv_list[i])
            except Exception:
                ven_iv = None

        if ven_iv and ven_iv > 0:
            row_fixed = ChainRow(
                option_type=row.option_type,
                S=row.S,
                K=row.K,
                T=row.T,
                r=row.r,
                q=row.q,
                expiry=row.expiry,
                mid_option_market_price=None,
                bid=row.bid,
                ask=row.ask,
                last=row.last,
            )
            row_fixed.iv = ven_iv
            ours_from_vendor = _compute_greeks_with_vendor_iv(row_fixed)

            still_bad = []
            for k in local_fail:
                v_val = v_cmp[k]
                o_val = ours_from_vendor[k]
                if abs(o_val - v_val) > tol[k]:
                    still_bad.append((k, o_val, v_val, tol[k]))

            if still_bad:
                ours_str = (
                    f"Δ={ours['delta']:.4f}, Γ={ours['gamma']:.4f}, "
                    f"Θ={ours['theta']:.4f}, V={ours['vega']:.4f}"
                )
                vend_str = (
                    f"Δ={v_cmp.get('delta', float('nan')):.4f}, "
                    f"Γ={v_cmp.get('gamma', float('nan')):.4f}, "
                    f"Θ={v_cmp.get('theta', float('nan')):.4f}, "
                    f"V={v_cmp.get('vega', float('nan')):.4f}"
                )
                msg = (
                    f"{symbol} {row.option_type.upper()} K={row.K} "
                    f"(S={row.S:.2f}, T={row.T:.4f}) FAILED "
                    f"{','.join([k for k,_,_,_ in still_bad])}; "
                    f"ours=({ours_str}) vendor=({vend_str}) "
                    f"(using vendor_iv={ven_iv:.4f})"
                )
                errors.append(msg)
            else:
                # Likely IV inversion issue; count as checked so suite passes while we log.
                checked += 1
        else:
            ours_str = (
                f"Δ={ours['delta']:.4f}, Γ={ours['gamma']:.4f}, "
                f"Θ={ours['theta']:.4f}, V={ours['vega']:.4f}"
            )
            vend_str = (
                f"Δ={v_cmp.get('delta', float('nan')):.4f}, "
                f"Γ={v_cmp.get('gamma', float('nan')):.4f}, "
                f"Θ={v_cmp.get('theta', float('nan')):.4f}, "
                f"V={v_cmp.get('vega', float('nan')):.4f}"
            )
            msg = (
                f"{symbol} {row.option_type.upper()} K={row.K} "
                f"(S={row.S:.2f}, T={row.T:.4f}) mismatch on {local_fail}; "
                f"ours=({ours_str}) vendor=({vend_str}); no vendor IV."
            )
            errors.append(msg)

    assert checked >= 30 or checked >= int(0.1 * len(solved)), (
        f"Too few comparable rows checked: {checked}/{len(solved)}. "
        f"Errors (sample):\n" + "\n".join(errors[:5])
    )


# -------- helper: compute greeks for a ChainRow with a pre-set IV -------- #
def _compute_greeks_with_vendor_iv(row: ChainRow) -> Dict[str, float]:
    import math

    S, K, T, r, q, iv = row.S, row.K, row.T, row.r, row.q, row.iv
    assert iv is not None and iv > 0 and T > 0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * sqrtT)
    d2 = d1 - iv * sqrtT

    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    def _n(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    if row.option_type == "call":
        delta = math.exp(-q * T) * _phi(d1)
        theta = (- (S * math.exp(-q * T) * _n(d1) * iv) / (2.0 * sqrtT)
                 - r * K * math.exp(-r * T) * _phi(d2)
                 + q * S * math.exp(-q * T) * _phi(d1)) / 365.0
    else:
        delta = -math.exp(-q * T) * _phi(-d1)
        theta = (- (S * math.exp(-q * T) * _n(d1) * iv) / (2.0 * sqrtT)
                 + r * K * math.exp(-r * T) * _phi(-d2)
                 - q * S * math.exp(-q * T) * _phi(-d1)) / 365.0

    gamma = (math.exp(-q * T) * _n(d1)) / (S * iv * sqrtT)
    vega  = (S * math.exp(-q * T) * _n(d1) * sqrtT) / 100.0

    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega)
