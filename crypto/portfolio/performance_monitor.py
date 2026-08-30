"""Strategy Performance Monitor (PRD §15).

Tracks per-strategy metrics and identifies underperformers.
Underperformers are disabled for a cooldown period, then automatically
re-enabled with a fresh evaluation window so the strategy gets a chance
to recover instead of being killed forever.
"""

from __future__ import annotations

import logging
import time
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

        # Cooldown before a disabled strategy is automatically re-enabled.
        # Configurable via strategies.disable_cooldown_hours (default 6h).
        self._disable_cooldown_seconds = (
            config.get("strategies", {}).get("disable_cooldown_hours", 6) * 3600
        )
        self._disabled_at: dict[str, float] = {}

        logger.info("PerformanceMonitor initialized (disable_cooldown=%dh)",
                    self._disable_cooldown_seconds // 3600)

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
        now = time.time()

        for sid, perf in self._performances.items():
            # Disabled strategies are skipped during their cooldown — they are
            # re-enabled (with a fresh evaluation window) by strategies_to_reenable().
            if not perf.is_enabled:
                continue
            if perf.total_trades < self._min_trades_for_eval:
                continue

            if perf.win_rate < self._min_win_rate:
                logger.warning(
                    "Strategy %s underperforming: win_rate=%.1f%%, pnl=%.4f — disabling for %.0fh",
                    sid, perf.win_rate, perf.total_pnl,
                    self._disable_cooldown_seconds / 3600,
                )
                underperformers.append(sid)
                perf.is_enabled = False
                self._disabled_at[sid] = now

        return underperformers

    def strategies_to_reenable(self) -> list[str]:
        """Return disabled strategies whose cooldown has expired.

        Their performance stats are reset so each re-enabled strategy
        gets a fresh evaluation window before it can be disabled again.
        """
        reenable: list[str] = []
        now = time.time()

        for sid, perf in self._performances.items():
            if perf.is_enabled:
                continue
            disabled_at = self._disabled_at.get(sid, now)
            elapsed = now - disabled_at
            if elapsed >= self._disable_cooldown_seconds:
                logger.info(
                    "Strategy %s re-enabled after %.1fh cooldown (fresh evaluation window)",
                    sid, elapsed / 3600,
                )
                reenable.append(sid)
                self._reset_performance(perf)
                self._disabled_at.pop(sid, None)

        return reenable

    @staticmethod
    def _reset_performance(perf: StrategyPerformance) -> None:
        """Reset a strategy's metrics for a fresh evaluation window."""
        perf.total_trades = 0
        perf.winning_trades = 0
        perf.losing_trades = 0
        perf.total_pnl = 0.0
        perf.peak_pnl = 0.0
        perf.current_drawdown = 0.0
        perf.is_enabled = True

    def get_performance(self, strategy_id: str) -> StrategyPerformance | None:
        return self._performances.get(strategy_id)

    def get_all_performances(self) -> dict[str, StrategyPerformance]:
        return dict(self._performances)
