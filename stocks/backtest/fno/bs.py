"""Black-Scholes pricer for the F&O validation rig (self-contained math)."""

from __future__ import annotations

import math


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t_years: float, rate: float,
             iv: float, kind: str) -> float:
    """European Black-Scholes call ('CE') / put ('PE')."""
    if t_years <= 0:
        return max(spot - strike, 0.0) if kind == "CE" else max(strike - spot, 0.0)
    iv = max(iv, 0.01)
    sqt = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t_years) / (iv * sqt)
    d2 = d1 - iv * sqt
    disc = math.exp(-rate * t_years)
    if kind == "CE":
        return spot * _ncdf(d1) - strike * disc * _ncdf(d2)
    return strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)
