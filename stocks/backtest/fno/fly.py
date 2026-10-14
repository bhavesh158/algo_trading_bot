"""Thursday iron fly replay on modelled NIFTY weekly options.

Structure per Thursday expiry:
  entry 09:30 -> sell ATM CE + ATM PE, buy wings +/- WING_WIDTH
  exit  15:15 -> buy back shorts, sell wings (flat before exercise; no
                 exercise STT, no overnight risk)

Costs per leg-fill: brokerage Rs.20 + STT 0.15% on SELL premium turnover
+ exchange/SEBI/GST/stamp ~Rs.12 estimate (documented). Slippage: 1% of
premium adverse on each fill (options spreads are wide; conservative).

Margin (informational): iron fly spread margin ~Rs.45-60k/lot < Rs.1.6L.
"""

from __future__ import annotations

import logging
from datetime import datetime, time

import pandas as pd

from .bs import bs_price

logger = logging.getLogger(__name__)

LOT = 65                    # NIFTY 2026
STRIKE_STEP = 50
RATE = 0.065
ENTRY_T = time(9, 30)
EXIT_T = time(15, 15)
BROKER_PER_FILL = 20.0
STT_RATE = 0.0015           # 0.15% on sell-side premium turnover
OTHER_PER_FILL = 12.0       # exchange + SEBI + GST + stamp estimate
SLIP = 0.01                 # 1% adverse per fill


def _years_to_expiry(now: datetime, day) -> float:
    expiry = datetime(day.year, day.month, day.day, 15, 30)
    return max((expiry - now).total_seconds() / (365 * 24 * 3600), 1e-6)


def run_fly(nsei_15m: pd.DataFrame, vix_daily: pd.DataFrame,
            wing_width: int = 200, iv_mult: float = 1.0,
            entry_t: time = ENTRY_T, exit_t: time = EXIT_T) -> list[dict]:
    """Replay every Thursday in the data. Returns per-expiry results."""
    out: list[dict] = []
    days = sorted({ts.date() for ts in nsei_15m.index})
    for day in days:
        if day.weekday() != 3:  # Thursday only
            continue
        bars = nsei_15m[nsei_15m.index.date == day]
        bars = bars[(bars.index.time >= time(9, 15)) & (bars.index.time <= time(15, 30))]
        if len(bars) < 10:
            continue
        # IV: previous session's India VIX close
        vix_hist = vix_daily[vix_daily.index.date < day]
        if vix_hist.empty:
            continue
        iv = float(vix_hist["close"].iloc[-1]) / 100.0 * iv_mult

        open_px = float(bars.iloc[0]["open"])
        atm = round(open_px / STRIKE_STEP) * STRIKE_STEP
        legs = [  # (kind, strike, qty_sign): +1 buy, -1 sell
            ("CE", atm, -1), ("PE", atm, -1),
            ("CE", atm + wing_width, +1), ("PE", atm - wing_width, +1),
        ]
        ebar = bars[bars.index.time >= entry_t]
        xbar = bars[bars.index.time >= exit_t]
        if ebar.empty or xbar.empty:
            continue
        e_ts, x_ts = ebar.index[0], xbar.index[0]
        e_spot = float(ebar.iloc[0]["close"])
        x_spot = float(xbar.iloc[0]["close"])
        day_range = float(bars["high"].max() - bars["low"].min())

        # Per-leg P&L: sells profit when premium falls (px < pe),
        # buys profit when premium rises. sgn=+1 buy, -1 sell.
        gross, cost = 0.0, 0.0
        max_loss_bound = wing_width * LOT  # worst case before credit
        for kind, strike, sgn in legs:
            pe = bs_price(e_spot, strike, _years_to_expiry(e_ts, day), RATE, iv, kind)
            px = bs_price(x_spot, strike, _years_to_expiry(x_ts, day), RATE, iv, kind)
            pe *= (1 - SLIP) if sgn < 0 else (1 + SLIP)   # entry fill
            px *= (1 + SLIP) if sgn < 0 else (1 - SLIP)   # exit fill
            gross += sgn * (px - pe) * LOT
            cost += BROKER_PER_FILL + OTHER_PER_FILL
            if sgn < 0:  # sold at entry -> STT on entry premium
                cost += STT_RATE * pe * LOT
            else:        # wings sold at exit -> STT on exit premium
                cost += STT_RATE * px * LOT
        net = gross - cost
        out.append({
            "expiry": str(day), "atm": atm, "iv": round(iv, 4),
            "entry_spot": round(e_spot, 2), "exit_spot": round(x_spot, 2),
            "day_range": round(day_range, 2),
            "gross": round(gross, 2), "cost": round(cost, 2), "net": round(net, 2),
            "max_loss_bound": max_loss_bound,
        })
    return out
