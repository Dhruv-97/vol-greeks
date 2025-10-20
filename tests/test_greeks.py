import math
import itertools
import pytest

from vol_greeks.blackscholes import (
    BlackScholesInputs,
    calculate_black_scholes_price,
    calculate_black_scholes_d1_d2,
    _T_EXP,
)
from vol_greeks.greeks import (
    BlackScholesGreeks,
    calculate_delta,
    calculate_gamma,
    calculate_theta,
    calculate_vega,
    calculate_rho,
    calculate_dividend_rho,
)

# ---------- helpers ----------

def central_fd(f_plus, f_minus, h):
    return (f_plus - f_minus) / (2.0 * h)

def second_fd(f0, f_plus, f_minus, h):
    return (f_plus - 2.0*f0 + f_minus) / (h*h)

def price(x: BlackScholesInputs) -> float:
    return calculate_black_scholes_price(x)

# Reasonable finite-diff steps (tuned not to flicker but still tight)
HS   = 1e-2      # $0.01 spot step
HSIG = 1e-4      # 0.0001 vol (10 bps) step
HR   = 1e-5      # 1 bps rate step
HQ   = 1e-5      # 1 bps dividend step
HT   = 1/365000.0  # ~0.001 day in years for theta FD


# ---------- 0DTE behavior ----------

def test_0dte_delta_step_and_others_zero():
    S,K,r,q,sig = 100, 100, 0.02, 0.01, 0.25
    T = _T_EXP / 2

    at = BlackScholesInputs("call", S,K,T,r,sig,q)
    assert calculate_delta(at) == pytest.approx(0.5, abs=0)

    call_itm = BlackScholesInputs("call", 105,K,T,r,sig,q)
    call_otm = BlackScholesInputs("call",  95,K,T,r,sig,q)
    put_itm  = BlackScholesInputs("put",   95,K,T,r,sig,q)
    put_otm  = BlackScholesInputs("put",  105,K,T,r,sig,q)

    assert calculate_delta(call_itm) == 1.0
    assert calculate_delta(call_otm) == 0.0
    assert calculate_delta(put_itm)  == -1.0
    assert calculate_delta(put_otm)  == 0.0

    # Others zero at 0DTE
    for x in [at, call_itm, call_otm, put_itm, put_otm]:
        assert calculate_gamma(x) == 0.0
        assert calculate_vega(x)  == 0.0
        assert calculate_theta(x) == 0.0
        assert calculate_rho(x)   == 0.0
        assert calculate_dividend_rho(x) == 0.0


# ---------- identities that should always hold ----------

@pytest.mark.parametrize("opt", ["call", "put"])
def test_put_call_parity(opt):
    # C - P = S e^{-qT} - K e^{-rT}
    S,K,T,r,q,sig = 123, 120, 0.7, 0.03, 0.01, 0.22
    C = price(BlackScholesInputs("call", S,K,T,r,sig,q))
    P = price(BlackScholesInputs("put",  S,K,T,r,sig,q))
    rhs = S*math.exp(-q*T) - K*math.exp(-r*T)
    assert (C - P) == pytest.approx(rhs, rel=1e-12, abs=1e-10)


def test_gamma_and_vega_equal_for_call_put():
    # In Black–Scholes, gamma and (per-unit) vega are identical for calls and puts
    S,K,T,r,q,sig = 150, 140, 0.9, 0.02, 0.015, 0.35
    call = BlackScholesInputs("call", S,K,T,r,sig,q)
    put  = BlackScholesInputs("put",  S,K,T,r,sig,q)

    assert calculate_gamma(call) == pytest.approx(calculate_gamma(put), rel=1e-12, abs=1e-10)
    assert calculate_vega(call)  == pytest.approx(calculate_vega(put),  rel=1e-12, abs=1e-10)


# ---------- finite difference checks (core Greeks) ----------

@pytest.mark.parametrize("opt", ["call", "put"])
def test_delta_fd(opt):
    S,K,T,r,q,sig = 100, 105, 0.6, 0.01, 0.02, 0.30
    x0  = BlackScholesInputs(opt, S,K,T,r,sig,q)
    xp  = BlackScholesInputs(opt, S+HS,K,T,r,sig,q)
    xm  = BlackScholesInputs(opt, S-HS,K,T,r,sig,q)
    dVdS = central_fd(price(xp), price(xm), HS)
    assert calculate_delta(x0) == pytest.approx(dVdS, rel=2e-3, abs=2e-5)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_gamma_fd(opt):
    S,K,T,r,q,sig = 120, 100, 0.75, 0.00, 0.01, 0.20
    x0  = BlackScholesInputs(opt, S,K,T,r,sig,q)
    xp  = BlackScholesInputs(opt, S+HS,K,T,r,sig,q)
    xm  = BlackScholesInputs(opt, S-HS,K,T,r,sig,q)
    g_fd = second_fd(price(x0), price(xp), price(xm), HS)
    assert calculate_gamma(x0) == pytest.approx(g_fd, rel=3e-3, abs=3e-6)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_vega_fd(opt):
    # calculate_vega returns per-unit-σ; aggregator scales to per 1% separately
    S,K,T,r,q,sig = 90, 100, 0.9, -0.005, 0.00, 0.40  # include negative rate case
    x0  = BlackScholesInputs(opt, S,K,T,r,sig,q)
    xp  = BlackScholesInputs(opt, S,K,T,r,sig+HSIG,q)
    xm  = BlackScholesInputs(opt, S,K,T,r,sig-HSIG,q)
    dVdsig = central_fd(price(xp), price(xm), HSIG)
    assert calculate_vega(x0) == pytest.approx(dVdsig, rel=2e-3, abs=2e-5)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_rho_fd(opt):
    # calculate_rho should be per-unit r; aggregator scales to per 1%
    S,K,T,r,q,sig = 105, 100, 0.5, 0.015, 0.01, 0.18
    x0  = BlackScholesInputs(opt, S,K,T,r,sig,q)
    xp  = BlackScholesInputs(opt, S,K,T,r+HR,sig,q)
    xm  = BlackScholesInputs(opt, S,K,T,r-HR,sig,q)
    dVdr = central_fd(price(xp), price(xm), HR)
    assert calculate_rho(x0) == pytest.approx(dVdr, rel=2e-3, abs=2e-5)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_dividend_rho_fd(opt):
    # calculate_dividend_rho should be per-unit q; aggregator scales to per 1%
    S,K,T,r,q,sig = 100, 100, 0.8, 0.03, 0.02, 0.22
    x0  = BlackScholesInputs(opt, S,K,T,r,sig,q)
    xp  = BlackScholesInputs(opt, S,K,T,r,sig,q+HQ)
    xm  = BlackScholesInputs(opt, S,K,T,r,sig,q-HQ)
    dVdq = central_fd(price(xp), price(xm), HQ)
    assert calculate_dividend_rho(x0) == pytest.approx(dVdq, rel=2e-3, abs=2e-5)


def test_theta_day_convention_via_fd_call():
    # Aggregator returns theta/day. Here we verify raw theta/day matches FD.
    S,K,T,r,q,sig = 100, 100, 1.0, 0.02, 0.01, 0.25
    x0  = BlackScholesInputs("call", S,K,T,r,sig,q)
    xp  = BlackScholesInputs("call", S,K,T+HT,r,sig,q)
    xm  = BlackScholesInputs("call", S,K,T-HT,r,sig,q)
    theta_fd_day = (price(xm) - price(xp)) / (2.0 * HT * 365.0)
    assert calculate_theta(x0) == pytest.approx(theta_fd_day, rel=5e-3, abs=5e-5)


# ---------- aggregator scaling & identities ----------

def test_aggregator_scaling_and_identities():
    S,K,T,r,q,sig = 110, 100, 0.6, 0.01, 0.02, 0.33
    x = BlackScholesInputs("put", S,K,T,r,sig,q)
    g = BlackScholesGreeks(x)

    # check vega per 1% vs FD
    xp = BlackScholesInputs("put", S,K,T,r,sig+HSIG,q)
    xm = BlackScholesInputs("put", S,K,T,r,sig-HSIG,q)
    dVdsig = central_fd(price(xp), price(xm), HSIG)
    assert g["vega"] == pytest.approx(dVdsig * 0.01, rel=3e-3, abs=3e-5)

    # check rho per 1%
    xp = BlackScholesInputs("put", S,K,T,r+HR,sig,q)
    xm = BlackScholesInputs("put", S,K,T,r-HR,sig,q)
    dVdr = central_fd(price(xp), price(xm), HR)
    assert g["rho"] == pytest.approx(dVdr * 0.01, rel=3e-3, abs=3e-5)

    # check dividend_rho per 1%
    xp = BlackScholesInputs("put", S,K,T,r,sig,q+HQ)
    xm = BlackScholesInputs("put", S,K,T,r,sig,q-HQ)
    dVdq = central_fd(price(xp), price(xm), HQ)
    assert g["dividend_rho"] == pytest.approx(dVdq * 0.01, rel=3e-3, abs=3e-5)

    # vanna identity via d2 and vega per-unit
    d1, d2 = calculate_black_scholes_d1_d2(x)
    vega_per_unit = g["vega"] * 100.0
    expected_vanna = -(d2 / sig) * vega_per_unit
    assert g["vanna"] == pytest.approx(expected_vanna, rel=3e-3, abs=3e-5)

    # volga identity via d1*d2/σ
    expected_volga = vega_per_unit * (d1 * d2) / sig
    assert g["volga"] == pytest.approx(expected_volga, rel=3e-3, abs=3e-5)


# ---------- continuity near cutoff ----------

@pytest.mark.parametrize("side", ["call", "put"])
def test_continuity_near_TEXP(side):
    S,K,r,q,sig = 100, 100, 0.02, 0.01, 0.20
    # Just-above and just-below the cutoff
    T_above = _T_EXP * 1.01
    T_below = _T_EXP * 0.99

    xa = BlackScholesInputs(side, S,K,T_above,r,sig,q)
    xb = BlackScholesInputs(side, S,K,T_below,r,sig,q)

    # vega, gamma -> 0 as T->0
    assert calculate_vega(xa) >= calculate_vega(xb)
    assert calculate_gamma(xa) >= calculate_gamma(xb)
    # theta bounded (not asserting sign, just finite and approaching 0)
    assert abs(calculate_theta(xb)) <= abs(calculate_theta(xa)) + 1e-6


# ---------- grid-based smoke (robustness) ----------

PARAMS = {
    "S":   [80, 100, 150],
    "K":   [90, 100, 120],
    "T":   [0.2, 0.75, 1.5],
    "r":   [-0.01, 0.0, 0.03],
    "q":   [0.0, 0.02],
    "sig": [0.10, 0.25, 0.60],
}
OPT_SIDES = ["call", "put"]

@pytest.mark.parametrize("opt,S,K,T,r,q,sig",
    [(opt,S,K,T,r,q,sig)
     for opt in OPT_SIDES
     for S in PARAMS["S"]
     for K in PARAMS["K"]
     for T in PARAMS["T"]
     for r in PARAMS["r"]
     for q in PARAMS["q"]
     for sig in PARAMS["sig"]]
)
def test_grid_no_explosions_and_basic_properties(opt,S,K,T,r,q,sig):
    x = BlackScholesInputs(opt, S,K,T,r,sig,q)
    g = BlackScholesGreeks(x)
    # Finite and real
    for k, v in g.items():
        assert math.isfinite(v), f"{k} not finite at {x}"

    # Gamma should be >= 0 (convexity)
    assert g["gamma"] >= -1e-10

    # Vega (per 1%) should be >= 0
    assert g["vega"] >= -1e-10



def test_delta_at_expiry_step():
    S, K, r, q, sig = 100, 100, 0.02, 0.01, 0.20
    # Force 0DTE via tiny T (<= _T_EXP)
    T = _T_EXP / 2

    call = BlackScholesInputs("call", S, K, T, r, sig, q)
    put  = BlackScholesInputs("put",  S, K, T, r, sig, q)

    assert calculate_delta(call) == pytest.approx(0.5, abs=0)  # ATM
    assert calculate_delta(put)  == pytest.approx(-0.5, abs=0)

    # ITM/OTM steps:
    call_itm = BlackScholesInputs("call", 105, 100, T, r, sig, q)
    call_otm = BlackScholesInputs("call",  95, 100, T, r, sig, q)
    put_itm  = BlackScholesInputs("put",   95, 100, T, r, sig, q)
    put_otm  = BlackScholesInputs("put",  105, 100, T, r, sig, q)
    assert calculate_delta(call_itm) == 1.0
    assert calculate_delta(call_otm) == 0.0
    assert calculate_delta(put_itm)  == -1.0
    assert calculate_delta(put_otm)  == 0.0


def test_0dte_other_greeks_zero():
    S, K, r, q, sig = 100, 100, 0.02, 0.01, 0.20
    T = _T_EXP / 2
    x = BlackScholesInputs("call", S, K, T, r, sig, q)

    assert calculate_gamma(x) == 0.0
    assert calculate_vega(x)  == 0.0
    assert calculate_theta(x) == 0.0
    assert calculate_rho(x)   == 0.0
    assert calculate_dividend_rho(x) == 0.0


def test_greeks_scaling_in_aggregator():
    # Aggregator returns: vega per 1%σ, rho per 1% r, dividend_rho per 1% q, theta per day.
    S, K, T, r, q, sig = 100, 100, 0.75, 0.03, 0.01, 0.25
    x = BlackScholesInputs("call", S, K, T, r, sig, q)
    g = BlackScholesGreeks(x)

    # Finite-difference settings
    hs = 1e-3     # for S
    hsig = 1e-4   # for sigma
    hr = 1e-5     # for rate
    hq = 1e-5     # for dividend yield
    hT = 1/3650   # ~0.1 day (in years) for theta FD

    # --- Vega per 1% σ ---
    x_hi = BlackScholesInputs("call", S, K, T, r, sig + hsig, q)
    x_lo = BlackScholesInputs("call", S, K, T, r, sig - hsig, q)
    dV_dsigma_per_unit = (calculate_black_scholes_price(x_hi) - calculate_black_scholes_price(x_lo)) / (2*hsig)
    expected_vega_per_percent = dV_dsigma_per_unit * 0.01
    assert g["vega"] == pytest.approx(expected_vega_per_percent, rel=2e-3, abs=2e-5)

    # --- Rho per 1% r ---
    x_hi = BlackScholesInputs("call", S, K, T, r + hr, sig, q)
    x_lo = BlackScholesInputs("call", S, K, T, r - hr, sig, q)
    dV_dr_per_unit = (calculate_black_scholes_price(x_hi) - calculate_black_scholes_price(x_lo)) / (2*hr)
    expected_rho_per_percent = dV_dr_per_unit * 0.01
    assert g["rho"] == pytest.approx(expected_rho_per_percent, rel=2e-3, abs=2e-5)

    # --- Dividend rho (psi) per 1% q ---
    x_hi = BlackScholesInputs("call", S, K, T, r, sig, q + hq)
    x_lo = BlackScholesInputs("call", S, K, T, r, sig, q - hq)
    dV_dq_per_unit = (calculate_black_scholes_price(x_hi) - calculate_black_scholes_price(x_lo)) / (2*hq)
    expected_psi_per_percent = dV_dq_per_unit * 0.01
    assert g["dividend_rho"] == pytest.approx(expected_psi_per_percent, rel=2e-3, abs=2e-5)

    # --- Theta per day ---
    # theta ≈ (V(T - dT) - V(T + dT)) / (2*dT*365)
    x_hi = BlackScholesInputs("call", S, K, T + hT, r, sig, q)
    x_lo = BlackScholesInputs("call", S, K, T - hT, r, sig, q)
    theta_fd = (calculate_black_scholes_price(x_lo) - calculate_black_scholes_price(x_hi)) / (2*hT*365.0)
    assert g["theta"] == pytest.approx(theta_fd, rel=5e-3, abs=5e-5)


def test_gamma_second_derivative_fd():
    S, K, T, r, q, sig = 130, 120, 0.6, 0.01, 0.02, 0.35
    hs = 1e-2  # $0.01 step on spot
    x0 = BlackScholesInputs("put", S, K, T, r, sig, q)
    xp = BlackScholesInputs("put", S + hs, K, T, r, sig, q)
    xm = BlackScholesInputs("put", S - hs, K, T, r, sig, q)

    V0 = calculate_black_scholes_price(x0)
    Vp = calculate_black_scholes_price(xp)
    Vm = calculate_black_scholes_price(xm)
    gamma_fd = (Vp - 2*V0 + Vm) / (hs*hs)

    assert calculate_gamma(x0) == pytest.approx(gamma_fd, rel=3e-3, abs=3e-6)


def test_vanna_identity():
    # vanna ≈ -(d2/σ) * vega_per_unit  ; aggregator returns vega per 1% -> multiply by 100 to get per-unit
    S, K, T, r, q, sig = 95, 100, 0.9, 0.015, 0.00, 0.40
    x = BlackScholesInputs("call", S, K, T, r, sig, q)
    d1, d2 = calculate_black_scholes_d1_d2(x)
    g = BlackScholesGreeks(x)
    vega_per_unit = g["vega"] * 100.0
    lhs = g["vanna"]
    rhs = - (d2 / sig) * vega_per_unit
    assert lhs == pytest.approx(rhs, rel=2e-3, abs=2e-5)
