"""Strategy Engine — orchestrates multiple strategies (PRD §8).

Loads configured strategies, runs them against active pairs,
and aggregates/filters trading signals.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from crypto.core.enums import MarketRegime, OrderSide
from crypto.core.event_bus import EventBus
from crypto.core.events import SignalEvent
from crypto.core.models import Position, Signal
from crypto.data.market_data_engine import MarketDataEngine
from crypto.strategy.base_strategy import BaseStrategy

_BTC_FILTER_LOG_INTERVAL = 1800  # log at most once per 30 min

logger = logging.getLogger(__name__)

# Regimes where mean reversion BUY is dangerous (fighting the trend)
_REGIME_BLOCK_MR_BUY = {MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOLATILITY}
# Regimes where mean reversion SELL is dangerous
_REGIME_BLOCK_MR_SELL = {MarketRegime.TRENDING_UP, MarketRegime.HIGH_VOLATILITY}
# Regimes where trend_following is dangerous (whipsaw / unreliable EMA crossovers)
_REGIME_BLOCK_TF = {MarketRegime.HIGH_VOLATILITY}


class StrategyEngine:
    """Runs all enabled strategies and publishes trading signals."""

    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        risk_manager: Any,
        position_sizer: Any,
        ai_analysis: Any,
        regime_detector: Any,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.ai_analysis = ai_analysis
        self.regime_detector = regime_detector

        self._strategies: dict[str, BaseStrategy] = {}
        self._open_positions_ref: dict | None = None  # injected by scheduler for correlation check
        self._max_same_dir_positions = config.get("strategies", {}).get("max_same_direction_positions", 3)
        self._confidence_threshold = config.get("strategies", {}).get(
            "default_confidence_threshold", 0.6
        )
        # Shorting crypto is inherently riskier — require higher confidence for SELL signals.
        self._sell_confidence_threshold = config.get("strategies", {}).get(
            "sell_confidence_threshold", 0.75
        )
        # Signal spam suppression: track last signal time per (symbol, side, strategy)
        self._last_signal_log: dict[tuple[str, str, str], datetime] = {}
        self._signal_log_interval = 300  # seconds between repeated signal logs

        # --- Signal burst limiter ---
        burst_cfg = config.get("strategies", {})
        self._burst_window_seconds = burst_cfg.get("signal_burst_window_seconds", 600)  # 10 min
        self._max_signals_per_window = burst_cfg.get("max_signals_per_window", 3)
        self._per_symbol_cooldown_seconds = burst_cfg.get("per_symbol_cooldown_seconds", 900)  # 15 min
        self._recent_signals: deque[tuple[datetime, str]] = deque()  # (timestamp, symbol)
        self._last_signal_per_symbol: dict[str, datetime] = {}  # symbol -> last signal time

        # --- BTC market direction filter ---
        self._market_data: MarketDataEngine | None = None
        self._btc_filter_enabled = burst_cfg.get("btc_trend_filter_enabled", True)
        self._btc_filter_symbol = burst_cfg.get("btc_trend_symbol", "BTC/USDT")
        self._btc_filter_tf = burst_cfg.get("btc_trend_timeframe", "1h")
        self._last_btc_filter_log: datetime | None = None

        # --- Macro context (MacroAnalyst) ---
        self._macro_analyst: Any = None
        macro_cfg = config.get("macro_analyst", {})
        self._macro_apply_mood_filter: bool = macro_cfg.get("apply_mood_filter", True)
        self._macro_apply_pair_bias: bool = macro_cfg.get("apply_pair_bias", True)
        self._last_macro_block_log: datetime | None = None

        logger.info("StrategyEngine initialized (threshold=%.2f, burst_max=%d/%ds, btc_filter=%s)",
                    self._confidence_threshold, self._max_signals_per_window,
                    self._burst_window_seconds, self._btc_filter_enabled)

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        """Wire in the market data engine for BTC trend checking."""
        self._market_data = market_data

    def set_macro_analyst(self, macro_analyst: Any) -> None:
        """Wire in the MacroAnalyst for macro-driven signal gating."""
        self._macro_analyst = macro_analyst

    def set_open_positions_ref(self, positions_ref: dict) -> None:
        """Wire in live open positions dict for same-direction correlation check."""
        self._open_positions_ref = positions_ref

    def _btc_market_bullish(self) -> bool | None:
        """Check BTC/USDT 1h EMA9 vs EMA21.

        Returns True (bullish), False (bearish), or None (data unavailable — don't block).
        """
        if not self._btc_filter_enabled or self._market_data is None:
            return None
        try:
            df = self._market_data.get_dataframe(self._btc_filter_symbol, self._btc_filter_tf)
            if df.empty or len(df) < 25:
                return None
            ema9 = df.get("ema_9", pd.Series(dtype=float))
            ema21 = df.get("ema_21", pd.Series(dtype=float))
            if ema9.empty or ema21.empty:
                return None
            v9, v21 = ema9.iloc[-1], ema21.iloc[-1]
            close = float(df["close"].iloc[-1])
            if pd.isna(v9) or pd.isna(v21):
                return None
            # Require BOTH: EMA9 > EMA21 AND price above EMA21
            # EMA-only check lags — price below EMA21 means market already extended downward
            return bool(v9 > v21 and close > v21)
        except Exception:
            logger.debug("BTC trend check failed — not blocking signals")
            return None

    def load_strategies(self, market_data: MarketDataEngine) -> None:
        """Instantiate all configured strategies."""
        enabled = self.config.get("strategies", {}).get("enabled", [])

        strategy_map = {
            "trend_following": "crypto.strategy.strategies.trend_following.TrendFollowingStrategy",
            "mean_reversion": "crypto.strategy.strategies.mean_reversion.MeanReversionStrategy",
            "breakout_momentum": "crypto.strategy.strategies.breakout_momentum.BreakoutMomentumStrategy",
        }

        for name in enabled:
            if name not in strategy_map:
                logger.warning("Unknown strategy: %s", name)
                continue
            try:
                module_path, class_name = strategy_map[name].rsplit(".", 1)
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                strategy = cls(self.config, market_data)
                self._strategies[name] = strategy
                logger.info("Strategy loaded: %s", name)
            except Exception:
                logger.exception("Failed to load strategy: %s", name)

    def run_strategies(
        self, symbols: list[str], excluded_symbols: set[str] | None = None,
    ) -> list[Signal]:
        """Run all enabled strategies against the given symbols.

        Args:
            symbols: Pairs to analyze.
            excluded_symbols: Symbols to skip (e.g. those with open positions).
        """
        skip = excluded_symbols or set()
        signals: list[Signal] = []
        now = datetime.now(timezone.utc)

        # Prune stale burst tracking entries
        cutoff = now.timestamp() - self._burst_window_seconds
        while self._recent_signals and self._recent_signals[0][0].timestamp() < cutoff:
            self._recent_signals.popleft()

        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue

            for symbol in symbols:
                if symbol in skip:
                    continue
                try:
                    signal = strategy.analyze(symbol)
                    if signal is None:
                        continue

                    # --- Burst limiter: global window ---
                    if len(self._recent_signals) >= self._max_signals_per_window:
                        logger.debug(
                            "BURST BLOCKED: %s %s — %d signals in last %ds",
                            signal.side.name, symbol,
                            len(self._recent_signals), self._burst_window_seconds,
                        )
                        continue

                    # --- Burst limiter: per-symbol cooldown ---
                    last_sym_signal = self._last_signal_per_symbol.get(symbol)
                    if last_sym_signal:
                        elapsed = (now - last_sym_signal).total_seconds()
                        if elapsed < self._per_symbol_cooldown_seconds:
                            logger.debug(
                                "COOLDOWN BLOCKED: %s %s — %.0fs since last signal (need %ds)",
                                signal.side.name, symbol, elapsed,
                                self._per_symbol_cooldown_seconds,
                            )
                            continue

                    # AI confidence adjustment
                    if self.ai_analysis:
                        signal = self.ai_analysis.adjust_confidence(signal)

                    # Confidence filter — higher bar for SELL (shorting is riskier)
                    threshold = (
                        self._sell_confidence_threshold
                        if signal.side == OrderSide.SELL
                        else self._confidence_threshold
                    )
                    if signal.confidence < threshold:
                        logger.debug(
                            "CONF FILTERED: %s %s conf=%.2f < threshold=%.2f (by %s)",
                            signal.side.name, symbol, signal.confidence, threshold, name,
                        )
                        continue

                    # Regime filter
                    regime = self.regime_detector.current_regime if self.regime_detector else MarketRegime.UNKNOWN
                    if name == "mean_reversion":
                        if signal.side == OrderSide.BUY and regime in _REGIME_BLOCK_MR_BUY:
                            continue
                        if signal.side == OrderSide.SELL and regime in _REGIME_BLOCK_MR_SELL:
                            continue
                    if name == "trend_following" and regime in _REGIME_BLOCK_TF:
                        continue

                    # Macro context gates (mood filter, avoid list, directional bias)
                    # Applied AFTER regime filter, BEFORE BTC filter.
                    # Always fails-open: if no valid context, no signals are blocked.
                    if self._macro_analyst is not None:
                        macro_ctx = self._macro_analyst.get_context()
                        if macro_ctx and macro_ctx.is_valid:
                            # Gate 1a: strong risk_off (≥0.85) — hard-block all BUYs
                            if (
                                self._macro_apply_mood_filter
                                and signal.side == OrderSide.BUY
                                and macro_ctx.blocks_buys
                            ):
                                now_log = datetime.now(timezone.utc)
                                if (
                                    self._last_macro_block_log is None
                                    or (now_log - self._last_macro_block_log).total_seconds() >= 1800
                                ):
                                    logger.info(
                                        "MACRO BLOCKED BUY: mood=%s(%.2f) themes=%s — "
                                        "F&G=%d(%s)",
                                        macro_ctx.market_mood, macro_ctx.mood_confidence,
                                        ", ".join(macro_ctx.active_themes) or "none",
                                        macro_ctx.fear_greed_score, macro_ctx.fear_greed_label,
                                    )
                                    self._last_macro_block_log = now_log
                                continue

                            # Gate 1b: moderate risk_off (0.70–0.84) — confidence penalty
                            # Qualifies trades by reducing confidence rather than vetoing.
                            # Strong signals (e.g. VWAP at 1.0) survive; marginal ones don't.
                            if self._macro_apply_mood_filter and signal.side == OrderSide.BUY:
                                penalty = macro_ctx.buy_confidence_penalty
                                if penalty > 0:
                                    signal.confidence = max(0.0, signal.confidence - penalty)
                                    logger.debug(
                                        "MACRO PENALTY: %s BUY conf −%.2f → %.2f "
                                        "(mood=%s %.2f)",
                                        symbol, penalty, signal.confidence,
                                        macro_ctx.market_mood, macro_ctx.mood_confidence,
                                    )

                            # Gate 2: hard-block pairs on the avoid list
                            if macro_ctx.should_avoid(symbol):
                                logger.debug(
                                    "MACRO AVOID: %s is on avoid list — signal blocked",
                                    symbol,
                                )
                                continue

                            # Gate 3: directional bias mismatch
                            if self._macro_apply_pair_bias:
                                bias = macro_ctx.get_pair_bias(symbol)
                                if signal.side == OrderSide.BUY and bias == "SELL":
                                    logger.debug(
                                        "MACRO BIAS BLOCKED: %s BUY — AI bias is SELL",
                                        symbol,
                                    )
                                    continue
                                if signal.side == OrderSide.SELL and bias == "BUY":
                                    logger.debug(
                                        "MACRO BIAS BLOCKED: %s SELL — AI bias is BUY",
                                        symbol,
                                    )
                                    continue

                    # BTC market direction filter (symmetric):
                    #   BTC bearish  → block BUY  (don’t buy into a downtrend)
                    #   BTC bullish  → block SELL (don’t short into an uptrend)
                    # Requires BOTH EMA9>EMA21 AND price>EMA21 to avoid lagging-EMA false signals.
                    btc_bullish = self._btc_market_bullish()
                    if signal.side == OrderSide.BUY and btc_bullish is False:
                        now_log = datetime.now(timezone.utc)
                        if (self._last_btc_filter_log is None or
                                (now_log - self._last_btc_filter_log).total_seconds()
                                >= _BTC_FILTER_LOG_INTERVAL):
                            logger.info(
                                "BTC FILTER: %s bearish — blocking BUY signals",
                                self._btc_filter_symbol,
                            )
                            self._last_btc_filter_log = now_log
                        continue
                    if signal.side == OrderSide.SELL and btc_bullish is True:
                        now_log = datetime.now(timezone.utc)
                        if (self._last_btc_filter_log is None or
                                (now_log - self._last_btc_filter_log).total_seconds()
                                >= _BTC_FILTER_LOG_INTERVAL):
                            logger.info(
                                "BTC FILTER: %s bullish — blocking SELL (short) signals",
                                self._btc_filter_symbol,
                            )
                            self._last_btc_filter_log = now_log
                        continue

                    # Correlated-position cap: block if too many open positions in same direction
                    if self._open_positions_ref is not None:
                        same_dir_count = sum(
                            1 for pos in self._open_positions_ref.values()
                            if pos.side == signal.side
                        )
                        if same_dir_count >= self._max_same_dir_positions:
                            logger.debug(
                                "CORR_CAP BLOCKED: %s %s — already %d open %s positions (max %d)",
                                signal.side.name, symbol, same_dir_count,
                                signal.side.name, self._max_same_dir_positions,
                            )
                            continue

                    # Risk check
                    if not self.risk_manager.can_take_trade(signal):
                        continue

                    signals.append(signal)
                    self.event_bus.publish(SignalEvent(signal=signal))

                    # Record signal for burst tracking
                    self._recent_signals.append((now, symbol))
                    self._last_signal_per_symbol[symbol] = now

                    # Suppress repeated signal logs for the same symbol/side/strategy
                    sig_key = (symbol, signal.side.name, name)
                    last_logged = self._last_signal_log.get(sig_key)
                    if not last_logged or (now - last_logged).total_seconds() >= self._signal_log_interval:
                        logger.info(
                            "SIGNAL: %s %s %s conf=%.2f (by %s) [regime=%s]",
                            signal.side.name, symbol, signal.strength.name,
                            signal.confidence, name, regime.name,
                        )
                        self._last_signal_log[sig_key] = now

                except Exception:
                    logger.exception("Error in strategy %s for %s", name, symbol)

        return signals

    def check_exits(self, symbols: list[str], positions_info: dict) -> list[Signal]:
        """Check exit conditions for open positions.

        Args:
            positions_info: dict of symbol -> {strategy_id, entry_price, current_price, side, position}
                           'position' is an optional Position object for trailing/time exits.
        """
        exit_signals: list[Signal] = []

        for symbol in symbols:
            info = positions_info.get(symbol)
            if not info:
                continue

            strategy_id = info.get("strategy_id", "")
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                continue

            try:
                exit_sig = strategy.get_exit_signal(
                    symbol, info["entry_price"], info["current_price"],
                    side=info.get("side", OrderSide.BUY),
                    position=info.get("position"),
                )
                if exit_sig:
                    exit_signals.append(exit_sig)
            except Exception:
                logger.exception("Error checking exit for %s", symbol)

        return exit_signals

    def disable_strategy(self, strategy_id: str) -> None:
        strategy = self._strategies.get(strategy_id)
        if strategy:
            strategy.enabled = False
            logger.warning("Strategy disabled: %s", strategy_id)

    def enable_strategy(self, strategy_id: str) -> None:
        strategy = self._strategies.get(strategy_id)
        if strategy:
            strategy.enabled = True
            logger.info("Strategy re-enabled: %s", strategy_id)
