"""Daily-timeframe crypto trend (+vol) validation.

Same breakout logic, daily bars, multi-day holds: costs stay ~0.2-1.2%
while targets become 10-30% moves. The cost-ratio experiment.
Usage: venv/bin/python -m stocks.backtest.run_crypto_daily [--only trend,vol]
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")

SPLIT = date(2026, 5, 15)
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna()
    d.index = pd.to_datetime(d.index.date)
    return d


def report(name, trades):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"{name}: no trades"); return
    df["day"] = pd.to_datetime(df["entry_time"]).dt.date
    tds = df["tds"].sum() if "tds" in df else 0.0
    print(f"\n### {name} ({len(df)} trades, TDS locked: ${tds:,.2f})")
    for label, sub in (("ALL", df), ("TRAIN", df[df.day <= SPLIT]), ("TEST", df[df.day > SPLIT])):
        n = len(sub)
        w = (sub.net > 0).mean() if n else 0
        net = sub.net.sum()
        taxed = net - 0.30 * max(net, 0)
        print(f"  {label:5} n={n:>4} win={w:.0%} exp_net=${sub.net.mean() if n else 0:+8.2f} "
              f"net=${net:+10.2f} post-tax(30%)=${taxed:+10.2f} "
              f"targets={(sub.reason == 'target').sum()} exits={dict(sub.reason.value_counts())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    only = {v.strip() for v in ap.parse_args().only.split(",") if v.strip()} or {"trend", "vol"}
    from stocks.backtest import replay as R
    from stocks.core.event_bus import EventBus
    from stocks.data.market_data_engine import MarketDataEngine
    from stocks.backtest.candidates.trend_follow import TrendFollowStrategy
    from stocks.backtest.candidates.vol_expansion import VolatilityExpansionStrategy

    cfg = yaml.safe_load(open("stocks/config/default_config.yaml"))
    frames, idx = {}, None
    for s in SYMBOLS:
        p = Path("stocks/backtest/data") / f"CRYPTO_{s.replace('/', '_')}_15m.csv"
        df15 = pd.read_csv(p, index_col=0, parse_dates=True)
        frames[s] = to_daily(df15)
    print("daily bars:", {k: len(v) for k, v in frames.items()})
    index_daily = frames["BTC/USDT"]  # regime input (len>30: real regimes)

    outdir = Path("stocks/backtest/results")
    outdir.mkdir(parents=True, exist_ok=True)
    kw = dict(round_the_clock=True, commission=0.0, fee_pct=0.001,
              tds_pct=0.01, slippage_pct=0.05, fractional_qty=True,
              notional=1000.0, max_concurrent=5)

    c2 = copy.deepcopy(cfg)
    c2.setdefault("backtest_candidates", {}).setdefault("trend_follow", {}).update(
        {"index_symbol": "BTC/USDT", "max_hold_minutes": 43200})  # 30-day backstop
    c3 = copy.deepcopy(cfg)
    c3.setdefault("backtest_candidates", {}).setdefault("vol_expansion", {})["max_hold_minutes"] = 43200

    jobs = []
    if "trend" in only:
        jobs.append(("dtrend", TrendFollowStrategy, c2, 0.5, 2.0))
    if "vol" in only:
        jobs.append(("dvol", VolatilityExpansionStrategy, c3, 0.5, 2.0))
    for name, cls, c, ct, rr in jobs:
        eng = MarketDataEngine(
            {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}}, EventBus())
        tr = R.run_backtest(frames, index_daily, [(name, cls(c, eng))], eng,
                            conf_threshold=ct, min_rr=rr,
                            index_frames={"BTC/USDT": frames["BTC/USDT"]}, **kw)
        pd.DataFrame(tr).to_csv(outdir / f"crypto_{name}.csv", index=False)
        report(f"crypto_{name}", tr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
