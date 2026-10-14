"""Crypto validation: trend/vol candidates on BTC/ETH/SOL 15m, tax-aware costs.

Costs: 0.1%/side fee + 1% TDS withheld on SELL fills (working-capital lock,
creditable) + 0.05% slippage. Report shows pre-tax net and 30%-tax haircut.
Usage: venv/bin/python -m stocks.backtest.run_crypto [--only trend,vol]
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
NOTIONAL_USDT = 1000.0


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()


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
    from stocks.backtest import replay as R, crypto_data as C
    from stocks.core.event_bus import EventBus
    from stocks.data.market_data_engine import MarketDataEngine
    from stocks.backtest.candidates.trend_follow import TrendFollowStrategy
    from stocks.backtest.candidates.vol_expansion import VolatilityExpansionStrategy

    cfg = yaml.safe_load(open("stocks/config/default_config.yaml"))
    frames = {s: C.fetch_klines(s, "15m", 365) for s in SYMBOLS}
    frames = {k: v for k, v in frames.items() if not v.empty}
    print("span:", {k: (str(v.index.min()), str(v.index.max()), len(v)) for k, v in frames.items()})
    index_daily = to_daily(frames["BTC/USDT"])

    outdir = Path("stocks/backtest/results")
    outdir.mkdir(parents=True, exist_ok=True)
    kw = dict(round_the_clock=True, commission=0.0, fee_pct=0.001,
              tds_pct=0.01, slippage_pct=0.05, fractional_qty=True,
              notional=NOTIONAL_USDT, max_concurrent=5)

    if "trend" in only:
        c2 = copy.deepcopy(cfg)
        c2.setdefault("backtest_candidates", {}).setdefault("trend_follow", {})["index_symbol"] = "BTC/USDT"
        eng = MarketDataEngine(
            {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}}, EventBus())
        tr = R.run_backtest(frames, index_daily, [("trend_follow", TrendFollowStrategy(c2, eng))],
                            eng, conf_threshold=0.5, min_rr=2.0,
                            index_frames={"BTC/USDT": frames["BTC/USDT"]}, **kw)
        pd.DataFrame(tr).to_csv(outdir / "crypto_trend.csv", index=False)
        report("crypto_trend", tr)
    if "vol" in only:
        eng = MarketDataEngine(
            {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}}, EventBus())
        tr = R.run_backtest(frames, index_daily, [("vol_expansion", VolatilityExpansionStrategy(cfg, eng))],
                            eng, conf_threshold=0.5, min_rr=2.0, **kw)
        pd.DataFrame(tr).to_csv(outdir / "crypto_vol.csv", index=False)
        report("crypto_vol", tr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
