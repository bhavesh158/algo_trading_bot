"""Candidate validation runs: one replay per candidate, train/test split.

Usage (from repo root):
    venv/bin/python -m stocks.backtest.run_candidates [--only vol,trend,orb]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")

SPLIT = date(2026, 8, 14)

CANDIDATES = {
    # name: (module attr, strategy_id, conf_gate, rr_gate)
    "vol": ("vol_expansion", "vol_expansion", 0.50, 2.0),
    "trend": ("trend_follow", "trend_follow", 0.50, 2.0),
    "orb": ("strict_orb", "strict_orb", 0.60, 1.5),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma list among vol,trend,orb (default: all)")
    only = {v.strip() for v in ap.parse_args().only.split(",") if v.strip()} or set(CANDIDATES)
    from stocks.backtest import data as D, replay as R
    from stocks.backtest.run_phase2 import split_report
    from stocks.core.event_bus import EventBus
    from stocks.data.market_data_engine import MarketDataEngine
    from stocks.selection.stock_selector import NIFTY50_SYMBOLS
    from stocks.backtest.candidates import vol_expansion, trend_follow, strict_orb

    mods = {"vol": vol_expansion, "trend": trend_follow, "orb": strict_orb}
    classes = {"vol": vol_expansion.VolatilityExpansionStrategy,
               "trend": trend_follow.TrendFollowStrategy,
               "orb": strict_orb.StrictORBStrategy}

    cfg = yaml.safe_load(open("stocks/config/default_config.yaml"))
    commission = cfg["paper_trading"].get("commission_per_trade", 25.0)
    slippage = cfg["paper_trading"].get("slippage_pct", 0.05)

    universe = D.tradable_universe(list(NIFTY50_SYMBOLS))
    frames = {}
    for s in universe:
        p = D._cache_path(s, "15m")
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if not df.empty:
                frames[s] = df
    idx15 = D.fetch_symbol("^NSEI", "15m", "60d")
    index_daily = D.fetch_daily_index()
    print(f"Loaded {len(frames)} symbols + NSEI 15m ({len(idx15)} rows).")

    outdir = Path("stocks/backtest/results")
    outdir.mkdir(parents=True, exist_ok=True)

    for name in ("vol", "trend", "orb"):
        if name not in only:
            continue
        modname, sid, conf_th, min_rr = CANDIDATES[name]
        engine = MarketDataEngine(
            {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}},
            EventBus(),
        )
        strat = classes[name](cfg, engine)
        trades = R.run_backtest(
            frames, index_daily, [(sid, strat)], engine,
            conf_threshold=conf_th, min_rr=min_rr,
            commission=commission, slippage_pct=slippage,
            index_frames={"^NSEI": idx15},
        )
        pd.DataFrame(trades).to_csv(outdir / f"cand_{name}.csv", index=False)
        split_report(f"cand_{name}", trades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
