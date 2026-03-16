"""Strategy Engine — orchestrates all pluggable trading strategies.

Runs enabled strategies against watchlist symbols, collects signals,
filters through AI analysis and risk checks, then publishes validated signals.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from stocks.core.enums import MarketRegime, OrderSide
from stocks.core.event_bus import EventBus
from stocks.core.events import SignalEvent
from stocks.core.models import Signal
from stocks.data.market_data_engine import MarketDataEngine
from stocks.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# Strategy registry — maps config name to class import path
_STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {
    "mean_reversion": (
        "stocks.strategy.strategies.mean_reversion", "MeanReversionStrategy"
    ),
    "momentum_breakout": (
        "stocks.strategy.strategies.momentum_breakout", "MomentumBreakoutStrategy"
    ),
    "opening_range_breakout": (
        "stocks.strategy.strategies.opening_range_breakout", "OpeningRangeBreakoutStrategy"
    ),
    "vwap_reversion": (
        "stocks.strategy.strategies.vwap_reversion", "VWAPReversionStrategy"
    ),
}


class StrategyEngine:
    """Manages strategy lifecycle: instantiation, execution, signal collection."""

    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        risk_manager: Any,
        position_sizer: Any,
        ai_analysis: Any,
        regime_detector: Any,
        multi_tf: Any = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.ai_analysis = ai_analysis
        self.regime_detector = regime_detector
        self.multi_tf = multi_tf

        self._strategies: list[BaseStrategy] = []
        self._confidence_threshold = config.get("strategies", {}).get(
            "default_confidence_threshold", 0.6
        )

        # Macro context gating
        self._macro_analyst: Any = None
        macro_cfg = config.get("macro_analyst", {})
        self._macro_apply_mood_filter: bool = macro_cfg.get("apply_mood_filter", True)
        self._macro_apply_pair_bias: bool = macro_cfg.get("apply_pair_bias", True)
        self._last_macro_block_log: Optional[Any] = None

        logger.info("StrategyEngine initialized")

    def set_macro_analyst(self, macro_analyst: Any) -> None:
        """Wire in the MacroAnalyst for macro-driven signal gating."""
        self._macro_analyst = macro_analyst

    def load_strategies(self, market_data: MarketDataEngine) -> None:
        """Instantiate all enabled strategies from stocks.config."""
        enabled = self.config.get("strategies", {}).get("enabled", [])
        for name in enabled:
            strategy = self._create_strategy(name, market_data)
            if strategy:
                self._strategies.append(strategy)
                logger.info("Strategy loaded: %s", strategy)

        logger.info("Total strategies loaded: %d", len(self._strategies))

    @staticmethod
    def _create_strategy(name: str, market_data: MarketDataEngine) -> Optional[BaseStrategy]:
        """Dynamically import and instantiate a strategy by name."""
        if name not in _STRATEGY_REGISTRY:
            logger.warning("Unknown strategy: %s", name)
            return None

        module_path, class_name = _STRATEGY_REGISTRY[name]
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            # Each strategy receives global config + market data
            from stocks.config.settings import load_config
            config = load_config()
            return cls(config, market_data)
        except Exception:
            logger.exception("Failed to load strategy: %s", name)
            return None

    def run_strategies(self, symbols: list[str]) -> list[Signal]:
        """Run all enabled strategies against the given symbols.

        Returns a list of validated signals ready for execution.
        """
        raw_signals: list[Signal] = []

        # Get current market regime
        current_regime = self.regime_detector.current_regime

        for strategy in self._strategies:
            if not strategy.enabled:
                continue

            # Skip strategies unsuitable for current regime
            if not self._is_strategy_suitable(strategy, current_regime):
                continue

            for symbol in symbols:
                try:
                    signal = strategy.analyze(symbol)
                    if signal:
                        raw_signals.append(signal)
                except Exception:
                    logger.exception(
                        "Error in %s analyzing %s", strategy.strategy_id, symbol
                    )

        # Filter and enhance signals
        validated = self._filter_signals(raw_signals)

        # Publish validated signals
        for signal in validated:
            self.event_bus.publish(SignalEvent(signal=signal))

        if validated:
            logger.info(
                "Strategies produced %d raw signals, %d validated",
                len(raw_signals), len(validated),
            )

        return validated

    def _filter_signals(self, signals: list[Signal]) -> list[Signal]:
        """Apply AI analysis, multi-timeframe confirmation, confidence threshold, and risk checks."""
        validated: list[Signal] = []

        for signal in signals:
            # Multi-timeframe confirmation (if available)
            if self.multi_tf is not None:
                # Determine the signal's primary timeframe from the strategy
                signal_tf = self._get_strategy_timeframe(signal.strategy_id)
                mtf_score = self.multi_tf.confirm_signal(signal, signal_tf)

                # Reject signals strongly opposed by higher timeframes
                if mtf_score < -0.5:
                    logger.debug(
                        "Signal filtered (MTF against): %s %s mtf_score=%.2f",
                        signal.strategy_id, signal.symbol, mtf_score,
                    )
                    continue

                # Apply MTF score as confidence adjustment
                mtf_adjustment = mtf_score * 0.15  # Scale: -0.15 to +0.15
                signal.confidence = max(0.0, min(1.0, signal.confidence + mtf_adjustment))

            # AI confidence boost/penalty
            signal = self.ai_analysis.evaluate_signal(signal)

            # Macro context gates (mood filter, avoid list, directional bias)
            # Fails-open: no valid macro context → no signals blocked.
            if self._macro_analyst is not None:
                macro_ctx = self._macro_analyst.get_context()
                if macro_ctx and macro_ctx.is_valid:
                    from datetime import datetime, timezone
                    # Gate 1a: strong risk_off (≥0.85) — hard-block all BUYs
                    if (
                        self._macro_apply_mood_filter
                        and getattr(signal, "side", None) == OrderSide.BUY
                        and macro_ctx.blocks_buys
                    ):
                        now_log = datetime.now(timezone.utc)
                        if (
                            self._last_macro_block_log is None
                            or (now_log - self._last_macro_block_log).total_seconds() >= 1800
                        ):
                            logger.info(
                                "MACRO BLOCKED BUY: mood=%s(%.2f) themes=%s",
                                macro_ctx.market_mood, macro_ctx.mood_confidence,
                                ", ".join(macro_ctx.active_themes) or "none",
                            )
                            self._last_macro_block_log = now_log
                        continue

                    # Gate 1b: moderate risk_off (0.70–0.84) — confidence penalty
                    if (
                        self._macro_apply_mood_filter
                        and getattr(signal, "side", None) == OrderSide.BUY
                    ):
                        penalty = macro_ctx.buy_confidence_penalty
                        if penalty > 0:
                            signal.confidence = max(0.0, signal.confidence - penalty)
                            logger.debug(
                                "MACRO PENALTY: %s BUY conf −%.2f → %.2f "
                                "(mood=%s %.2f)",
                                signal.symbol, penalty, signal.confidence,
                                macro_ctx.market_mood, macro_ctx.mood_confidence,
                            )

                    # Gate 2: hard-block symbols on the avoid list
                    if macro_ctx.should_avoid(signal.symbol):
                        logger.debug(
                            "MACRO AVOID: %s is on avoid list — signal blocked", signal.symbol
                        )
                        continue

                    # Gate 3: directional bias mismatch
                    if self._macro_apply_pair_bias:
                        bias = macro_ctx.get_pair_bias(signal.symbol)
                        sig_side = getattr(signal, "side", None)
                        if sig_side == OrderSide.BUY and bias == "SELL":
                            logger.debug(
                                "MACRO BIAS BLOCKED: %s BUY — AI bias is SELL", signal.symbol
                            )
                            continue
                        if sig_side == OrderSide.SELL and bias == "BUY":
                            logger.debug(
                                "MACRO BIAS BLOCKED: %s SELL — AI bias is BUY", signal.symbol
                            )
                            continue

            # Confidence threshold
            if signal.confidence < self._confidence_threshold:
                logger.debug(
                    "Signal filtered (low confidence): %s %s conf=%.2f",
                    signal.strategy_id, signal.symbol, signal.confidence,
                )
                continue

            # Risk management pre-check
            if not self.risk_manager.can_take_trade(signal):
                logger.debug(
                    "Signal blocked by risk manager: %s %s",
                    signal.strategy_id, signal.symbol,
                )
                continue

            # Minimum risk-reward ratio
            min_rr = self.config.get("risk", {}).get("min_risk_reward_ratio", 1.5)
            if signal.risk_reward_ratio > 0 and signal.risk_reward_ratio < min_rr:
                logger.debug(
                    "Signal filtered (R:R too low): %s %s rr=%.2f",
                    signal.strategy_id, signal.symbol, signal.risk_reward_ratio,
                )
                continue

            validated.append(signal)

        return validated

    def _get_strategy_timeframe(self, strategy_id: str) -> Any:
        """Get the primary timeframe for a strategy."""
        from stocks.core.enums import Timeframe
        for s in self._strategies:
            if s.strategy_id == strategy_id:
                return s.primary_timeframe
        return Timeframe.M5

    @staticmethod
    def _is_strategy_suitable(strategy: BaseStrategy, regime: MarketRegime) -> bool:
        """Check if a strategy is suitable for the current market regime."""
        # Mean reversion works best in sideways/low-volatility markets
        if strategy.strategy_id == "mean_reversion":
            return regime in (
                MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY, MarketRegime.UNKNOWN
            )

        # Momentum breakout works best in trending markets
        if strategy.strategy_id == "momentum_breakout":
            return regime in (
                MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
                MarketRegime.HIGH_VOLATILITY, MarketRegime.UNKNOWN,
            )

        # VWAP reversion works in sideways and moderate conditions
        if strategy.strategy_id == "vwap_reversion":
            return regime in (
                MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY,
                MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
                MarketRegime.UNKNOWN,
            )

        # ORB works in most conditions
        return True

    def check_exits(self, symbols: list[str], positions: dict[str, Any]) -> list[Signal]:
        """Check if any open positions should be exited.

        Now passes the Position object to get_exit_signal for trailing stop
        and time-based exit checks.
        """
        exit_signals: list[Signal] = []

        for symbol, pos in positions.items():
            for strategy in self._strategies:
                if strategy.strategy_id != pos.get("strategy_id"):
                    continue
                current_price = pos.get("current_price", 0)
                entry_price = pos.get("entry_price", 0)
                position_obj = pos.get("position")  # Full Position object
                exit_signal = strategy.get_exit_signal(
                    symbol, entry_price, current_price, position=position_obj,
                )
                if exit_signal:
                    exit_signals.append(exit_signal)

        return exit_signals

    @property
    def strategies(self) -> list[BaseStrategy]:
        return list(self._strategies)

    def disable_strategy(self, strategy_id: str) -> None:
        for s in self._strategies:
            if s.strategy_id == strategy_id:
                s.enabled = False
                logger.info("Strategy disabled: %s", strategy_id)

    def enable_strategy(self, strategy_id: str) -> None:
        for s in self._strategies:
            if s.strategy_id == strategy_id:
                s.enabled = True
                logger.info("Strategy enabled: %s", strategy_id)
