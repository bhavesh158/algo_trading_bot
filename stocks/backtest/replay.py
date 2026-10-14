"""Event-replay engine: runs the REAL strategy classes on truncated history.

No live module is modified. The only runtime patch is the `datetime`
reference inside the ORB strategy module *in this process*, pointed at a
controllable clock (restored afterwards).
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
SQUARE_OFF = time(15, 7)          # mirrors live pre_close_start
WARMUP_DAYS = 2                   # let rolling indicators stabilize


class _SimClock:
    """Process-local fake clock for the ORB module's `datetime.now()`."""
    current: datetime | None = None


def _patch_orb_clock():
    import stocks.strategy.strategies.opening_range_breakout as orb_mod
    real_dt = orb_mod.datetime

    class FakeDT(real_dt):  # type: ignore[valid-type, misc]
        @classmethod
        def now(cls, tz=None):
            if _SimClock.current is not None:
                return _SimClock.current if tz is None else _SimClock.current.replace(tzinfo=tz)
            return real_dt.now(tz)

    orb_mod.datetime = FakeDT
    return orb_mod, real_dt


def daily_regime(index_daily: pd.DataFrame, day: pd.Timestamp.date) -> str:
    """Approximate live regime from ^NSEI daily bars strictly before `day`."""
    df = index_daily[index_daily.index.date < day]
    if len(df) < 30:
        return "UNKNOWN"
    close, high, low = df["close"], df["high"], df["low"]
    # ADX(14), same math family as live
    up, dn = high.diff(), -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    dx = 100 * ((100 * plus_dm.rolling(14).mean() / atr.replace(0, np.nan)
                 - 100 * minus_dm.rolling(14).mean() / atr.replace(0, np.nan)).abs()
                / (100 * (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean())
                   / atr.replace(0, np.nan)).replace(0, np.nan))
    adx = float(dx.rolling(14).mean().iloc[-1])
    if np.isnan(adx):
        return "UNKNOWN"
    if adx >= 25:
        sma20 = float(close.rolling(20).mean().iloc[-1])
        return "TRENDING_UP" if float(close.iloc[-1]) >= sma20 else "TRENDING_DOWN"
    day_range = float((high.iloc[-1] - low.iloc[-1]))
    atr_last = float(atr.iloc[-1])
    if atr_last > 0 and day_range / atr_last > 1.5:
        return "HIGH_VOLATILITY"
    return "SIDEWAYS"


def strategy_allowed(strategy_id: str, regime: str) -> bool:
    """Mirror of strategy_engine._is_strategy_suitable (documented copy)."""
    if strategy_id == "mean_reversion":
        return regime in ("SIDEWAYS", "LOW_VOLATILITY", "UNKNOWN")
    if strategy_id == "momentum_breakout":
        return regime in ("TRENDING_UP", "TRENDING_DOWN", "HIGH_VOLATILITY", "UNKNOWN")
    if strategy_id == "vwap_reversion":
        return regime in ("SIDEWAYS", "LOW_VOLATILITY", "TRENDING_UP",
                          "TRENDING_DOWN", "UNKNOWN")
    return True  # ORB and others: all regimes


def run_backtest(data_15m: dict[str, pd.DataFrame],
                 index_daily: pd.DataFrame,
                 strategies: list[tuple[str, Any]],
                 engine: Any,
                 conf_threshold: float,
                 min_rr: float,
                 max_concurrent: int = 5,
                 notional: float = 30000.0,
                 commission: float = 25.0,
                 slippage_pct: float = 0.05,
                 entry_start: time | None = None,
                 regime_allow: dict[str, set[str]] | None = None,
                 index_frames: dict[str, pd.DataFrame] | None = None,
                 round_the_clock: bool = False,
                 fee_pct: float = 0.0,
                 tds_pct: float = 0.0,
                 fractional_qty: bool = False) -> list[dict]:
    """Replay session bars; returns closed-trade dicts.

    Phase 2 options (all default to Phase 1 behavior):
    - entry_start: skip entries on bars before this session time.
    - regime_allow: {strategy_id: {regimes}} overriding suitability for
      listed strategies (others use the live mapping).
    - index_frames: extra symbol->frame map (e.g. ^NSEI 15m) truncated in
      lockstep for index-aware candidates.
    Crypto options (default off):
    - round_the_clock: no session filter, no square-off, positions may be
      held across days (remaining opens marked at final bar, reason "open").
    - fee_pct: per-side fraction of fill value, added to `commission`.
    - tds_pct: withheld fraction on SELL-side fills (working-capital lock).
    """
    from stocks.core.enums import OrderSide
    from stocks.core.models import Position
    from stocks.core.enums import Timeframe

    orb_mod, real_dt = _patch_orb_clock()
    try:
        return _run(data_15m, index_daily, strategies, engine, conf_threshold,
                    min_rr, max_concurrent, notional, commission, slippage_pct,
                    OrderSide, Position, Timeframe, entry_start,
                    regime_allow or {}, index_frames or {},
                    round_the_clock, fee_pct, tds_pct, fractional_qty)
    finally:
        orb_mod.datetime = real_dt
        _SimClock.current = None


def _run(data_15m, index_daily, strategies, engine, conf_threshold, min_rr,
         max_concurrent, notional, commission, slippage_pct,
         OrderSide, Position, Timeframe, entry_start, regime_allow,
         index_frames, round_the_clock, fee_pct, tds_pct,
         fractional_qty) -> list[dict]:
    # All session dates present in the data
    if round_the_clock:
        all_dates = sorted({ts.date() for df in data_15m.values() for ts in df.index})
    else:
        all_dates = sorted({ts.date() for df in data_15m.values() for ts in df.index
                            if SESSION_OPEN <= ts.time() <= SESSION_CLOSE})
    sim_dates = all_dates[WARMUP_DAYS:]
    logger.info("Replaying %d sessions (%s .. %s) rtc=%s", len(sim_dates),
                sim_dates[0] if sim_dates else None, sim_dates[-1] if sim_dates else None,
                round_the_clock)

    slip = slippage_pct / 100.0
    trades: list[dict] = []
    open_pos: dict[str, dict] = {}  # symbol -> {pos, strategy, opened_at, max_hold}

    def side_cost(fill: float, qty: int, is_sell_fill: bool) -> float:
        c = commission + fill * qty * fee_pct
        if is_sell_fill:
            c += fill * qty * tds_pct
        return c

    for day in sim_dates:
        regime = daily_regime(index_daily, day)
        # Union of bar stamps in session for that day
        if round_the_clock:
            stamps = sorted({ts for df in data_15m.values() for ts in df.index
                             if ts.date() == day})
        else:
            stamps = sorted({ts for df in data_15m.values() for ts in df.index
                             if ts.date() == day and SESSION_OPEN <= ts.time() <= SESSION_CLOSE})
        for ts in stamps:
            _SimClock.current = ts
            # --- point every symbol's frames at <= ts and refresh indicators
            for sym, full in data_15m.items():
                part = full.loc[:ts]
                if part.empty:
                    continue
                engine._data[sym][Timeframe.M15] = part
                engine._compute_indicators(sym, Timeframe.M15)
            for sym, full in index_frames.items():
                part = full.loc[:ts]
                if part.empty:
                    continue
                engine._data[sym][Timeframe.M15] = part
                engine._compute_indicators(sym, Timeframe.M15)

            # --- exits first (frees slots, mirrors live ordering)
            for sym in list(open_pos):
                bar = _bar_at(data_15m.get(sym), ts)
                if bar is None:
                    continue
                st = open_pos[sym]
                pos: Position = st["pos"]
                strategy = st["strategy"]
                is_buy = pos.side == OrderSide.BUY
                exited = None
                # 1. stop-loss on extremes (conservative: stop wins ties)
                if is_buy and bar["low"] <= pos.stop_loss:
                    exited = (pos.stop_loss, "stop_loss")
                elif not is_buy and bar["high"] >= pos.stop_loss:
                    exited = (pos.stop_loss, "stop_loss")
                # 2. target touch
                elif is_buy and bar["high"] >= pos.target_price > 0:
                    exited = (pos.target_price, "target")
                elif not is_buy and bar["low"] <= pos.target_price and pos.target_price > 0:
                    exited = (pos.target_price, "target")
                else:
                    close = float(bar["close"])
                    pos.current_price = close
                    pos.update_extremes(close)
                    # 3. strategy exit at close
                    if strategy.should_exit(sym, pos.entry_price, close):
                        exited = (close, "strategy")
                    # 4. trailing stops at close
                    elif strategy.check_trailing_stop(sym, pos) or \
                            strategy.check_trailing_stop_pct(pos):
                        exited = (close, "trailing_stop")
                    # 5. time exit
                    elif (ts - st["opened_at"]).total_seconds() / 60 >= st["max_hold"]:
                        exited = (close, "time_exit")
                    # 6. square-off (NSE sessions only)
                    elif not round_the_clock and ts.time() >= SQUARE_OFF:
                        exited = (close, "squareoff")
                if exited:
                    px, reason = exited
                    qty = pos.quantity
                    gross = ((px - pos.entry_price) if is_buy else (pos.entry_price - px)) * qty
                    c_in = side_cost(pos.entry_price, qty, not is_buy)
                    c_out = side_cost(px, qty, is_buy)
                    tds = (0.0 if is_buy else pos.entry_price * qty * tds_pct) + \
                          (px * qty * tds_pct if is_buy else 0.0)
                    friction = c_in + c_out
                    net = gross - friction
                    trades.append({
                        "symbol": sym, "strategy": strategy.strategy_id,
                        "side": "BUY" if is_buy else "SELL",
                        "qty": qty, "entry": pos.entry_price, "exit": px,
                        "stop": pos.stop_loss, "target": pos.target_price,
                        "entry_time": st["opened_at"], "exit_time": ts,
                        "reason": reason, "gross": gross, "net": net,
                        "commission": friction, "tds": tds,
                        "regime": st["regime"],
                        "confidence": st["confidence"],
                    })
                    del open_pos[sym]

            # --- entries (skip the square-off bar and beyond; NSE only)
            if not round_the_clock and ts.time() >= SQUARE_OFF:
                continue
            if entry_start is not None and ts.time() < entry_start:
                continue
            if len(open_pos) >= max_concurrent:
                continue
            for sym in data_15m:
                if sym in open_pos or _bar_at(data_15m[sym], ts) is None:
                    continue
                if len(open_pos) >= max_concurrent:
                    break
                for sid, strategy in strategies:
                    if sid in regime_allow:
                        if regime not in regime_allow[sid]:
                            continue
                    elif not strategy_allowed(sid, regime):
                        continue
                    try:
                        sig = strategy.analyze(sym)
                    except Exception:
                        logger.exception("analyze failed %s %s", sid, sym)
                        continue
                    if sig is None:
                        continue
                    # validation gates (mirror of strategy_engine, documented)
                    if sig.confidence < conf_threshold:
                        continue
                    rr = sig.risk_reward_ratio
                    if rr > 0 and rr < min_rr:
                        continue
                    # fill at this bar's close + adverse slippage
                    close = float(data_15m[sym].loc[ts, "close"])
                    is_buy = sig.side == OrderSide.BUY
                    fill = close * (1 + slip if is_buy else 1 - slip)
                    qty = (notional / fill) if fractional_qty else int(notional // fill)
                    if qty <= 0:
                        continue
                    pos = Position(symbol=sym, side=sig.side, quantity=qty,
                                   entry_price=fill, current_price=fill,
                                   stop_loss=sig.stop_loss, target_price=sig.target_price,
                                   strategy_id=sid, opened_at=ts)
                    pos.update_extremes(fill)
                    max_hold = (sig.metadata or {}).get("max_hold_minutes") \
                        or getattr(strategy, "_max_hold_minutes", 0) or 10**9
                    open_pos[sym] = {"pos": pos, "strategy": strategy,
                                     "opened_at": ts, "max_hold": max_hold,
                                     "regime": regime, "confidence": sig.confidence}
                    break  # one signal per symbol per bar

        # end of day: force-close anything left (NSE safety net; skipped 24/7)
        if round_the_clock:
            continue
        for sym, st in list(open_pos.items()):
            bar = data_15m[sym][data_15m[sym].index.date == day].iloc[-1]
            pos = st["pos"]
            is_buy = pos.side == OrderSide.BUY
            px = float(bar["close"])
            gross = ((px - pos.entry_price) if is_buy else (pos.entry_price - px)) * pos.quantity
            friction = (side_cost(pos.entry_price, pos.quantity, not is_buy)
                        + side_cost(px, pos.quantity, is_buy))
            trades.append({
                "symbol": sym, "strategy": pos.strategy_id,
                "side": "BUY" if is_buy else "SELL",
                "qty": pos.quantity, "entry": pos.entry_price, "exit": px,
                "stop": pos.stop_loss, "target": pos.target_price,
                "entry_time": st["opened_at"],
                "exit_time": data_15m[sym][data_15m[sym].index.date == day].index[-1],
                "reason": "eod", "gross": gross, "net": gross - friction,
                "commission": friction, "tds": 0.0, "regime": st["regime"],
                "confidence": st["confidence"],
            })
            del open_pos[sym]

    # 24/7: mark any still-open positions at the final bar (reason "open")
    if round_the_clock:
        for sym, st in list(open_pos.items()):
            pos = st["pos"]
            is_buy = pos.side == OrderSide.BUY
            px = float(data_15m[sym].iloc[-1]["close"])
            px_ts = data_15m[sym].index[-1]
            gross = ((px - pos.entry_price) if is_buy else (pos.entry_price - px)) * pos.quantity
            friction = (side_cost(pos.entry_price, pos.quantity, not is_buy)
                        + side_cost(px, pos.quantity, is_buy))
            trades.append({
                "symbol": sym, "strategy": pos.strategy_id,
                "side": "BUY" if is_buy else "SELL",
                "qty": pos.quantity, "entry": pos.entry_price, "exit": px,
                "stop": pos.stop_loss, "target": pos.target_price,
                "entry_time": st["opened_at"], "exit_time": px_ts,
                "reason": "open", "gross": gross, "net": gross - friction,
                "commission": friction, "tds": 0.0, "regime": st["regime"],
                "confidence": st["confidence"],
            })
            del open_pos[sym]

    logger.info("Replay done: %d closed trades", len(trades))
    return trades


def _bar_at(df: pd.DataFrame | None, ts) -> pd.Series | None:
    if df is None:
        return None
    try:
        return df.loc[ts]
    except KeyError:
        return None
