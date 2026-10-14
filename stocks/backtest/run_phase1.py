"""Phase 1 entry point: 60d x 15m x NSE universe x 3 real strategies.

Usage (from repo root):
    venv/bin/python -m stocks.backtest.run_phase1 [--force-fetch]

Reads stocks/config/default_config.yaml (gates + strategy params).
Writes nothing outside stocks/backtest/ (data cache + results CSV).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("backtest")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-fetch", action="store_true", help="ignore data cache")
    ap.add_argument("--notional", type=float, default=30000.0)
    args = ap.parse_args()

    from stocks.backtest import data as bt_data
    from stocks.backtest import replay as bt_replay
    from stocks.backtest import metrics as bt_metrics
    from stocks.core.event_bus import EventBus
    from stocks.data.market_data_engine import MarketDataEngine
    from stocks.selection.stock_selector import NIFTY50_SYMBOLS
    from stocks.strategy.strategies.momentum_breakout import MomentumBreakoutStrategy
    from stocks.strategy.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
    from stocks.strategy.strategies.vwap_reversion import VWAPReversionStrategy

    cfg = yaml.safe_load(open("stocks/config/default_config.yaml"))
    conf_th = cfg["strategies"].get("default_confidence_threshold", 0.5)
    min_rr = cfg["risk"].get("min_risk_reward_ratio", 2.0)
    commission = cfg["paper_trading"].get("commission_per_trade", 25.0)
    slippage = cfg["paper_trading"].get("slippage_pct", 0.05)
    logger.warning("Gates: conf>=%.2f rr>=%.2f comm=%.2f slip=%.3f%%",
                   conf_th, min_rr, commission, slippage)

    universe = bt_data.tradable_universe(list(NIFTY50_SYMBOLS))
    frames = bt_data.fetch_all(universe, "15m", "60d", force=args.force_fetch)
    index_daily = bt_data.fetch_daily_index()
    if not frames or index_daily.empty:
        logger.warning("No data fetched; aborting.")
        return 1

    engine = MarketDataEngine(
        {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}},
        EventBus(),
    )
    strategies = [
        ("vwap_reversion", VWAPReversionStrategy(cfg, engine)),
        ("momentum_breakout", MomentumBreakoutStrategy(cfg, engine)),
        ("opening_range_breakout", OpeningRangeBreakoutStrategy(cfg, engine)),
    ]

    trades = bt_replay.run_backtest(
        frames, index_daily, strategies, engine,
        conf_threshold=conf_th, min_rr=min_rr,
        notional=args.notional, commission=commission, slippage_pct=slippage,
    )
    summary = bt_metrics.summarize(trades)
    bt_metrics.print_report(summary)

    outdir = Path("stocks/backtest/results")
    outdir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(trades).to_csv(outdir / "phase1_trades.csv", index=False)
    print(f"Saved {len(trades)} trades -> {outdir / 'phase1_trades.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
