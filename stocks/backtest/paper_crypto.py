"""Daily paper-trading runner for crypto trend-daily (NO real orders, ever).

Reads live daily bars via CCXT public klines, runs the REAL
TrendFollowStrategy, tracks paper positions in paper_state.json.
Run once daily after 00:05 UTC:
    venv/bin/python -m stocks.backtest.paper_crypto

Costs modeled (not charged): 0.1%/side, 1% TDS flag, 0.05% slip.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("paper_crypto")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
NOTIONAL = 1000.0
STATE = Path(__file__).resolve().parent / "paper_state.json"
FEE_PCT, TDS_PCT, SLIP_PCT = 0.001, 0.01, 0.0005


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"open": {}, "closed": [], "started": datetime.now(timezone.utc).isoformat()}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2, default=str))


def daily_frames() -> dict[str, pd.DataFrame]:
    import ccxt
    ex = ccxt.binance({"enableRateLimit": True})
    out = {}
    for s in SYMBOLS:
        rows = ex.fetch_ohlcv(s, "15m", limit=1000)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts").sort_index()
        d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "volume": "sum"}).dropna()
        d.index = pd.to_datetime(d.index.date)
        out[s] = d.iloc[:-1]  # drop forming bar: only CLOSED days trade
    return out


def main() -> int:
    from stocks.core.enums import OrderSide
    from stocks.core.event_bus import EventBus
    from stocks.data.market_data_engine import MarketDataEngine
    from stocks.backtest.candidates.trend_follow import TrendFollowStrategy

    cfg = yaml.safe_load(open("stocks/config/default_config.yaml"))
    c2 = copy.deepcopy(cfg)
    # Promoted cell from 3y sensitivity: lb20, stop 2.5xATR / target 5xATR
    # (train +$24.73, n=74 | test +$44.38, n=5). Live code untouched.
    c2.setdefault("backtest_candidates", {}).setdefault("trend_follow", {}).update(
        {"index_symbol": "BTC/USDT", "max_hold_minutes": 43200,
         "lookback_bars": 20, "atr_stop": 2.5, "atr_target": 5.0})

    frames = daily_frames()
    last_day = max(df.index[-1] for df in frames.values())
    print(f"Last closed day: {last_day.date()}")

    eng = MarketDataEngine(
        {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}},
        EventBus())
    from stocks.core.enums import Timeframe
    for sym, full in {**frames, "BTC/USDT": frames["BTC/USDT"]}.items():
        eng._data[sym][Timeframe.M15] = full
        eng._compute_indicators(sym, Timeframe.M15)
    strat = TrendFollowStrategy(c2, eng)
    state = load_state()

    # --- exits on open paper positions (at today's close = last bar)
    for sym in list(state["open"]):
        p = state["open"][sym]
        bar = frames[sym].iloc[-1]
        is_buy = p["side"] == "BUY"
        px = float(bar["close"])
        exited, reason = None, ""
        if is_buy and bar["low"] <= p["stop"]:
            exited, reason = p["stop"], "stop_loss"
        elif not is_buy and bar["high"] >= p["stop"]:
            exited, reason = p["stop"], "stop_loss"
        elif is_buy and bar["high"] >= p["target"]:
            exited, reason = p["target"], "target"
        elif not is_buy and bar["low"] <= p["target"]:
            exited, reason = p["target"], "target"
        elif strat.should_exit(sym, p["entry"], px):
            exited, reason = px, "strategy"
        elif (datetime.now(timezone.utc).replace(tzinfo=None) -
              pd.Timestamp(p["opened_at"]).to_pydatetime()).total_seconds() / 60 >= p["max_hold"]:
            exited, reason = px, "time_exit"
        if exited is not None:
            gross = ((exited - p["entry"]) if is_buy else (p["entry"] - exited)) * p["qty"]
            fee = (p["entry"] + exited) * p["qty"] * FEE_PCT + abs(exited - p["entry"]) * 0
            tds = (exited * p["qty"] * TDS_PCT) if is_buy else (p["entry"] * p["qty"] * TDS_PCT)
            slip = (p["entry"] + exited) * p["qty"] * SLIP_PCT
            net = gross - fee - slip
            state["closed"].append({**p, "exit": exited, "exit_day": str(last_day.date()),
                                    "reason": reason, "gross": round(gross, 2),
                                    "net": round(net - tds, 2), "tds_locked": round(tds, 2)})
            print(f"EXIT {sym} {p['side']} @{exited:.2f} reason={reason} net=${net - tds:+.2f}")
            del state["open"][sym]

    # --- entries (max 5 concurrent, one per symbol)
    if len(state["open"]) < 5:
        for sym in SYMBOLS:
            if sym in state["open"] or len(state["open"]) >= 5:
                continue
            try:
                sig = strat.analyze(sym)
            except Exception:
                logger.exception("analyze failed %s", sym)
                continue
            if sig is None or sig.confidence < 0.5:
                continue
            rr = sig.risk_reward_ratio
            if rr > 0 and rr < 2.0:
                continue
            close = float(frames[sym].iloc[-1]["close"])
            is_buy = sig.side == OrderSide.BUY
            fill = close * (1 + SLIP_PCT if is_buy else 1 - SLIP_PCT)
            qty = NOTIONAL / fill
            state["open"][sym] = {
                "side": "BUY" if is_buy else "SELL", "qty": qty,
                "entry": fill, "stop": sig.stop_loss, "target": sig.target_price,
                "opened_at": str(last_day.date()), "confidence": round(sig.confidence, 2),
                "max_hold": (sig.metadata or {}).get("max_hold_minutes", 43200),
            }
            print(f"ENTER {sym} {'BUY' if is_buy else 'SELL'} {qty:.5f} @{fill:.2f} "
                  f"stop={sig.stop_loss:.2f} tgt={sig.target_price:.2f}")

    save_state(state)
    n = len(state["closed"])
    net = sum(t["net"] for t in state["closed"])
    print(f"Paper book: {len(state['open'])} open | {n} closed, total net=${net:+.2f} "
          f"(post-30%-tax=${net - 0.30 * max(net, 0):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
