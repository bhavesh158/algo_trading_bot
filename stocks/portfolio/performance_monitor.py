"""Strategy Performance Monitor (PRD §19).

Evaluates each strategy's win rate, profitability, drawdown, and consistency.
Underperforming strategies are flagged for reduced allocation or disabling.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from stocks.core.enums import AlertSeverity
from stocks.core.event_bus import EventBus
from stocks.core.events import AlertEvent
from stocks.core.models import Alert, StrategyPerformance, Trade

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Tracks and evaluates strategy-level performance metrics."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._performances: dict[str, StrategyPerformance] = {}
        self._min_win_rate = 30.0  # Disable if win rate drops below this
        self._min_trades_for_eval = 5  # Need at least this many trades

        logger.info("PerformanceMonitor initialized")

    def record_trade(self, trade: Trade) -> None:
        """Record a completed trade and update strategy performance."""
        perf = self._performances.get(trade.strategy_id)
        if perf is None:
            perf = StrategyPerformance(strategy_id=trade.strategy_id)
            self._performances[trade.strategy_id] = perf

        perf.total_trades += 1
        perf.total_pnl += trade.pnl

        if trade.is_winner:
            perf.winning_trades += 1
        else:
            perf.losing_trades += 1

        # Track peak and drawdown
        if perf.total_pnl > perf.peak_pnl:
            perf.peak_pnl = perf.total_pnl
        if perf.peak_pnl > 0:
            perf.current_drawdown = (perf.peak_pnl - perf.total_pnl) / perf.peak_pnl * 100
            perf.max_drawdown = max(perf.max_drawdown, perf.current_drawdown)

        logger.debug(
            "Strategy %s: trades=%d win_rate=%.1f%% pnl=%.2f",
            trade.strategy_id, perf.total_trades, perf.win_rate, perf.total_pnl,
        )

    def evaluate_strategies(self) -> list[str]:
        """Evaluate all strategies and return IDs of those that should be disabled.

        Returns list of strategy IDs that are underperforming.
        """
        underperforming: list[str] = []

        for strategy_id, perf in self._performances.items():
            if perf.total_trades < self._min_trades_for_eval:
                continue

            # Check win rate
            if perf.win_rate < self._min_win_rate:
                logger.warning(
                    "Strategy %s underperforming: win_rate=%.1f%% (min=%.1f%%)",
                    strategy_id, perf.win_rate, self._min_win_rate,
                )
                underperforming.append(strategy_id)

            # Check if strategy is consistently losing
            if perf.total_pnl < 0 and perf.total_trades >= self._min_trades_for_eval * 2:
                logger.warning(
                    "Strategy %s in net loss: pnl=%.2f across %d trades",
                    strategy_id, perf.total_pnl, perf.total_trades,
                )
                if strategy_id not in underperforming:
                    underperforming.append(strategy_id)

        if underperforming:
            self.event_bus.publish(AlertEvent(alert=Alert(
                severity=AlertSeverity.WARNING,
                source="PerformanceMonitor",
                message=f"Underperforming strategies: {', '.join(underperforming)}",
            )))

        return underperforming

    def get_performance(self, strategy_id: str) -> StrategyPerformance | None:
        return self._performances.get(strategy_id)

    def get_all_performances(self) -> dict[str, StrategyPerformance]:
        return dict(self._performances)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all strategy performances."""
        total_trades = sum(p.total_trades for p in self._performances.values())
        total_pnl = sum(p.total_pnl for p in self._performances.values())
        total_wins = sum(p.winning_trades for p in self._performances.values())

        return {
            "total_strategies": len(self._performances),
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "overall_win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "strategies": {
                sid: {
                    "trades": p.total_trades,
                    "win_rate": round(p.win_rate, 1),
                    "pnl": round(p.total_pnl, 2),
                    "max_drawdown": round(p.max_drawdown, 1),
                    "enabled": p.is_enabled,
                }
                for sid, p in self._performances.items()
            },
        }
