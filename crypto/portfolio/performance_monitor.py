"""Strategy Performance Monitor (PRD §15).

Tracks per-strategy metrics and identifies underperformers.
"""

from __future__ import annotations

import logging
from typing import Any

from crypto.core.event_bus import EventBus
from crypto.core.models import StrategyPerformance, Trade

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Tracks and evaluates per-strategy performance."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._performances: dict[str, StrategyPerformance] = {}

        self._disable_after_losses = 5  # Disable after N consecutive losses
        self._min_win_rate = 30.0  # Min win rate % before considering disable
        self._min_trades_for_eval = 10

        logger.info("PerformanceMonitor initialized")

    def record_trade(self, trade: Trade) -> None:
        """Record a completed trade for its strategy."""
        sid = trade.strategy_id
        if sid not in self._performances:
            self._performances[sid] = StrategyPerformance(strategy_id=sid)

        perf = self._performances[sid]
        perf.total_trades += 1
        perf.total_pnl += trade.pnl

        if trade.is_winner:
            perf.winning_trades += 1
        else:
            perf.losing_trades += 1

        # Track drawdown
        perf.peak_pnl = max(perf.peak_pnl, perf.total_pnl)
        if perf.peak_pnl > 0:
            perf.current_drawdown = (perf.peak_pnl - perf.total_pnl) / perf.peak_pnl * 100

    def evaluate_strategies(self) -> list[str]:
        """Evaluate all strategies. Returns list of strategy IDs to disable."""
        underperformers: list[str] = []

        for sid, perf in self._performances.items():
            if not perf.is_enabled or perf.total_trades < self._min_trades_for_eval:
                continue

            if perf.win_rate < self._min_win_rate:
                logger.warning(
                    "Strategy %s underperforming: win_rate=%.1f%%, pnl=%.4f",
                    sid, perf.win_rate, perf.total_pnl,
                )
                underperformers.append(sid)
                perf.is_enabled = False

        return underperformers

    def get_performance(self, strategy_id: str) -> StrategyPerformance | None:
        return self._performances.get(strategy_id)

    def get_all_performances(self) -> dict[str, StrategyPerformance]:
        return dict(self._performances)
