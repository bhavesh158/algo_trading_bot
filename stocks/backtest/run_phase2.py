"""Phase 2: hypothesis tests A/B/C + sensitivity, train/test split.

Usage (from repo root):
    venv/bin/python -m stocks.backtest.run_phase2

Reads cache from stocks/backtest/data/ (no refetch).
Writes per-variant CSVs to stocks/backtest/results/.
"""

from __future__ import annotations

import copy
import logging
import sys
from datetime import date, time
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("backtest2")

# Train/test cutoff: variants are judged on TEST (unseen) span.
SPLIT = date(2026, 8, 14)


def build_all(cfg):
    from stocks.core.event_bus import EventBus
    from stocks.data.market_data_engine import MarketDataEngine
    from stocks.strategy.strategies.momentum_breakout import MomentumBreakoutStrategy
    from stocks.strategy.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
    from stocks.strategy.strategies.vwap_reversion import VWAPReversionStrategy
    engine = MarketDataEngine(
        {"market_data": {"provider": "yahoo", "timeframes": ["15m"], "history_days": 30}},
        EventBus(),
    )
    return engine, [
        ("vwap_reversion", VWAPReversionStrategy(cfg, engine)),
        ("momentum_breakout", MomentumBreakoutStrategy(cfg, engine)),
        ("opening_range_breakout", OpeningRangeBreakoutStrategy(cfg, engine)),
    ]


def split_report(name, trades):
    from stocks.backtest import metrics as M
    tr = [t for t in trades if pd.Timestamp(t["entry_time"]).date() <= SPLIT]
    te = [t for t in trades if pd.Timestamp(t["entry_time"]).date() > SPLIT]
    print(f"\n### {name} (n={len(trades)} | train<={SPLIT}: {len(tr)} | test: {len(te)})")
    for label, subset in (("TRAIN", tr), ("TEST ", te)):
        s = M.summarize(subset)
        print(f"  {label} n={s['trades']:>4} win={s['win_rate']:.0%} "
              f"exp_net=Rs.{s['expectancy_net']:+7.2f} net=Rs.{s['net']:+10.2f} "
              f"targets={s['target_hits']} exits={s['exit_mix']}")
    return tr, te


def trailing_autopsy(trades, frames):
    """For trail-exited trades: what happened between exit and square-off?"""
    rec = {"rescued": 0, "cut_winner": 0, "neutral": 0, "rescued_rs": 0.0, "lost_rs": 0.0}
    for t in trades:
        if t["reason"] != "trailing_stop":
            continue
        df = frames.get(t["symbol"])
        if df is None:
            continue
        day = pd.Timestamp(t["exit_time"]).date()
        bars = df[df.index.date == day]
        bars = bars[bars.index > pd.Timestamp(t["exit_time"])]
        if bars.empty:
            continue
        buy = t["side"] == "BUY"
        hit_target = ((bars["high"] >= t["target"]).any() if buy
                      else (bars["low"] <= t["target"]).any())
        sq = float(bars["close"].iloc[-1])
        would = ((sq - t["entry"]) if buy else (t["entry"] - sq)) * t["qty"]
        got = t["gross"]
        if hit_target:
            rec["cut_winner"] += 1
            rec["lost_rs"] += ((t["target"] - t["entry"]) if buy else (t["entry"] - t["target"])) * t["qty"] - got
        elif would < got - 1e-9:
            rec["rescued"] += 1
            rec["rescued_rs"] += got - would
        else:
            rec["neutral"] += 1
    print(f"\n### B-autopsy on baseline trail exits: {rec}")
    return rec


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma list of variant names to run (default: all)")
    only = {v.strip() for v in ap.parse_args().only.split(",") if v.strip()}
    from stocks.backtest import data as D, replay as R
    from stocks.selection.stock_selector import NIFTY50_SYMBOLS

    cfg = yaml.safe_load(open("stocks/config/default_config.yaml"))
    conf_th = cfg["strategies"].get("default_confidence_threshold", 0.5)
    min_rr = cfg["risk"].get("min_risk_reward_ratio", 2.0)
    commission = cfg["paper_trading"].get("commission_per_trade", 25.0)
    slippage = cfg["paper_trading"].get("slippage_pct", 0.05)

    universe = D.tradable_universe(list(NIFTY50_SYMBOLS))
    frames, index_daily = {}, D.fetch_daily_index()
    for s in universe:
        p = D._cache_path(s, "15m")
        if p.exists():
            frames[s] = pd.read_csv(p, index_col=0, parse_dates=True)
    frames = {k: v for k, v in frames.items() if not v.empty}
    print(f"Loaded {len(frames)} symbols from cache.")

    outdir = Path("stocks/backtest/results")
    outdir.mkdir(parents=True, exist_ok=True)
    kw = dict(conf_threshold=conf_th, min_rr=min_rr,
              commission=commission, slippage_pct=slippage)

    def run(name, cfg_override=None, **rkw):
        if only and name not in only and name != "baseline":
            return [], []
        c = cfg_override or cfg
        engine, strats = build_all(c)
        trades = R.run_backtest(frames, index_daily, strats, engine, **kw, **rkw)
        pd.DataFrame(trades).to_csv(outdir / f"phase2_{name}.csv", index=False)
        return split_report(name, trades)

    # Baseline (replica with stop/target fields)
    base, _ = run("baseline")

    # A: regime gate on vwap (+ sensitivity neighbours)
    run("A_up_only", regime_allow={"vwap_reversion": {"TRENDING_UP"}})
    run("A_up_side", regime_allow={"vwap_reversion": {"TRENDING_UP", "SIDEWAYS"}})
    run("A_up_down", regime_allow={"vwap_reversion": {"TRENDING_UP", "TRENDING_DOWN"}})

    # B-autopsy on baseline, then trail-width sweep
    if not only or "autopsy" in only:
        base_df = pd.read_csv(outdir / "phase2_baseline.csv").to_dict("records")
        trailing_autopsy(base_df, frames)
    for tag, pct in (("B_trail04", 0.4), ("B_trail08", 0.8), ("B_trail12", 1.2)):
        c2 = copy.deepcopy(cfg)
        c2["vwap_reversion"]["trailing_stop_pct"] = pct
        run(tag, cfg_override=c2)

    # C: entry timing cutoffs
    run("C_from1000", entry_start=time(10, 0))
    run("C_from1030", entry_start=time(10, 30))
    return 0


if __name__ == "__main__":
    sys.exit(main())
