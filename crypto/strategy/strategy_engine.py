"""Strategy Engine — orchestrates multiple strategies (PRD §8).

Loads configured strategies, runs them against active pairs,
and aggregates/filters trading signals.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from crypto.core.event_bus import EventBus
from crypto.core.events import SignalEvent
from crypto.core.models import Signal
from crypto.data.market_data_engine import MarketDataEngine
from crypto.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


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
        self._confidence_threshold = config.get("strategies", {}).get(
            "default_confidence_threshold", 0.6
        )
        logger.info("StrategyEngine initialized (threshold=%.2f)", self._confidence_threshold)

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

    def run_strategies(self, symbols: list[str]) -> list[Signal]:
        """Run all enabled strategies against the given symbols."""
        signals: list[Signal] = []

        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue

            for symbol in symbols:
                try:
                    signal = strategy.analyze(symbol)
                    if signal is None:
                        continue

                    # AI confidence adjustment
                    if self.ai_analysis:
                        signal = self.ai_analysis.adjust_confidence(signal)

                    # Confidence filter
                    if signal.confidence < self._confidence_threshold:
                        continue

                    # Risk check
                    if not self.risk_manager.can_take_trade(signal):
                        continue

                    signals.append(signal)
                    self.event_bus.publish(SignalEvent(signal=signal))
                    logger.info(
                        "SIGNAL: %s %s %s conf=%.2f (by %s)",
                        signal.side.name, symbol, signal.strength.name,
                        signal.confidence, name,
                    )

                except Exception:
                    logger.exception("Error in strategy %s for %s", name, symbol)

        return signals

    def check_exits(self, symbols: list[str], positions_info: dict) -> list[Signal]:
        """Check exit conditions for open positions."""
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
