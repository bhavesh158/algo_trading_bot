"""Sensitivity sweep: daily crypto trend/vol over parameter grids.

Each cell: full-span + train/test compact line. Plateaus ship, peaks don't.
Usage: venv/bin/python -m stocks.backtest.run_crypto_sens [--only trend,vol]
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")

SPLIT = date(2026, 5, 15)
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

TREND_GRID = [
    {"lookback_bars": 15, "atr_stop": 2.0, "atr_target": 4.0},
    {"lookback_bars": 20, "atr_stop": 2.0, "atr_target": 4.0},  # base
    {"lookback_bars": 30, "atr_stop": 2.0, "atr_target": 4.0},
    {"lookback_bars": 20, "atr_stop": 1.5, "atr_target": 3.0},
    {"lookback_bars": 20, "atr_stop": 2.5, "atr_target": 5.0},
]
VOL_GRID = [
    {"range_atr_mult": 1.2, "extreme_frac": 0.75},
    {"range_atr_mult": 1.5, "extreme_frac": 0.75},  # base
    {"range_atr_mult": 1.8, "extreme_frac": 0.75},
    {"range_atr_mult": 1.5, "extreme_frac": 0.70},
    {"range_atr_mult": 1.5, "extreme_frac": 0.80},
]


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna()
    d.index = pd.to_datetime(d.index.date)
    return d


def cell(name, trades):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"{name:28} no trades"); return
    df["day"] = pd.to_datetime(df["entry_time"]).dt.date
    tr = df[df.day <= SPLIT]; te = df[df.day > SPLIT]
    print(f"{name:28} ALL n={len(df):>4} exp=${df.net.mean():+7.2f} net=${df.net.sum():+9.2f} | "
          f"TRN n={len(tr):>4} exp=${tr.net.mean() if len(tr) else 0:+7.2f} | "
          f"TST n={len(te):>4} exp=${te.net.mean() if len(te) else 0:+7.2f} net=${te.net.sum() if len(te) else 0:+9.2f}")


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
    frames = {}
    for s in SYMBOLS:
        p = Path("stocks/backtest/data") / f"CRYPTO_{s.replace('/', '_')}_15m.csv"
        df15 = pd.read_csv(p, index_col=0, parse_dates=True)
        frames[s] = to_daily(df15)
    print("daily bars:", {k: (str(v.index.min().date()), str(v.index.max().date()), len(v))
                          for k, v in frames.items()})
    index_daily = frames["BTC/USDT"]

    outdir = Path("stocks/backtest/results")
    outdir.mkdir(parents=True, exist_ok=True)
    kw = dict(round_the_clock=True, commission=0.0, fee_pct=0.001,
              tds_pct=0.01, slippage_pct=0.05, fractional_qty=True,
              notional=1000.0, max_concurrent=5)

    if "trend" in only:
        for i, g in enumerate(TREND_GRID):
            c = copy.deepcopy(cfg)
            c.setdefault("backtest_candidates", {}).setdefault("trend_follow", {}).update(
                {"index_symbol": "BTC/USDT", "max_hold_minutes": 43200, **g})
            eng = MarketDataEngine(
                {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}}, EventBus())
            tr = R.run_backtest(frames, index_daily, [("trend_follow", TrendFollowStrategy(c, eng))],
                                eng, conf_threshold=0.5, min_rr=2.0,
                                index_frames={"BTC/USDT": frames["BTC/USDT"]}, **kw)
            pd.DataFrame(tr).to_csv(outdir / f"sens_trend{i}.csv", index=False)
            cell(f"trend lb={g['lookback_bars']} {g['atr_stop']}R/{g['atr_target']}R", tr)
    if "vol" in only:
        for i, g in enumerate(VOL_GRID):
            c = copy.deepcopy(cfg)
            c.setdefault("backtest_candidates", {}).setdefault("vol_expansion", {}).update(
                {"max_hold_minutes": 43200, **g})
            eng = MarketDataEngine(
                {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}}, EventBus())
            tr = R.run_backtest(frames, index_daily, [("vol_expansion", VolatilityExpansionStrategy(c, eng))],
                                eng, conf_threshold=0.5, min_rr=2.0, **kw)
            pd.DataFrame(tr).to_csv(outdir / f"sens_vol{i}.csv", index=False)
            cell(f"vol rng={g['range_atr_mult']}x ext={g['extreme_frac']}", tr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
