"""Strategy Engine — orchestrates all pluggable trading strategies.

Runs enabled strategies against watchlist symbols, collects signals,
filters through AI analysis and risk checks, then publishes validated signals.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.enums import MarketRegime
from core.event_bus import EventBus
from core.events import SignalEvent
from core.models import Signal
from data.market_data_engine import MarketDataEngine
from strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# Strategy registry — maps config name to class import path
_STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {
    "mean_reversion": (
        "strategy.strategies.mean_reversion", "MeanReversionStrategy"
    ),
    "momentum_breakout": (
        "strategy.strategies.momentum_breakout", "MomentumBreakoutStrategy"
    ),
    "opening_range_breakout": (
        "strategy.strategies.opening_range_breakout", "OpeningRangeBreakoutStrategy"
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
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.ai_analysis = ai_analysis
        self.regime_detector = regime_detector

        self._strategies: list[BaseStrategy] = []
        self._confidence_threshold = config.get("strategies", {}).get(
            "default_confidence_threshold", 0.6
        )

        logger.info("StrategyEngine initialized")

    def load_strategies(self, market_data: MarketDataEngine) -> None:
        """Instantiate all enabled strategies from config."""
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
            from config.settings import load_config
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
        """Apply AI analysis, confidence threshold, and risk checks."""
        validated: list[Signal] = []

        for signal in signals:
            # AI confidence boost/penalty
            signal = self.ai_analysis.evaluate_signal(signal)

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

        # ORB works in most conditions
        return True

    def check_exits(self, symbols: list[str], positions: dict[str, Any]) -> list[Signal]:
        """Check if any open positions should be exited."""
        exit_signals: list[Signal] = []

        for symbol, pos in positions.items():
            for strategy in self._strategies:
                if strategy.strategy_id != pos.get("strategy_id"):
                    continue
                current_price = pos.get("current_price", 0)
                entry_price = pos.get("entry_price", 0)
                exit_signal = strategy.get_exit_signal(symbol, entry_price, current_price)
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
