"""Isolated historical backtest harness for NSE intraday strategies (Phase 1).

READ-ONLY w.r.t. live code: this package IMPORTS the real strategy classes
and the real MarketDataEngine indicator math, but NEVER modifies live files,
live configs, live state, or live logs. All replay state lives on a private
MarketDataEngine instance fed with truncated cached history (no look-ahead).

Fidelity notes (documented approximations):
- Entries fill at 15m bar close + slippage; exits evaluated on LATER bars.
- Stop-loss checked against bar extremes first (conservative), then
  target/strategy exits, then trailing (on close), then time exits.
- VWAP here is session-cumulative over the stored frame, exactly like live
  (live engine cumsums over its whole multi-day frame too).
- ORB opening range = first 15m bar high/low == max/min of its fifteen 1m
  bars, so no M1 data is needed for an exact range.
- Regime is approximated from ^NSEI daily (ADX>=25 -> trending by SMA20
  slope, else HIGH_VOLATILITY if daily range > 1.5*ATR14, else SIDEWAYS).
- Validation gates mirror strategy_engine: confidence >= threshold and
  risk_reward_ratio >= min_risk_reward_ratio. Macro/MTF/AI gates are OFF
  (live: mood filter disabled, no pair biases observed).
"""
