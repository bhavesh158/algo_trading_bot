"""Drawdown Protection Monitor.

Tracks portfolio drawdown and triggers protective actions:
- Warning: log alert
- Reduce: cut position sizing
- Pause: stop all new trades
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from crypto.core.event_bus import EventBus

if TYPE_CHECKING:
    from crypto.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class DrawdownMonitor:
    """Monitors drawdown and triggers protective actions."""

    def __init__(
        self, config: dict[str, Any], event_bus: EventBus, portfolio_manager: PortfolioManager,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.portfolio = portfolio_manager

        dd = config.get("drawdown", {})
        self._warning_pct = dd.get("warning_threshold_pct", 3.0)
        self._reduce_pct = dd.get("reduce_size_threshold_pct", 5.0)
        self._pause_pct = dd.get("pause_trading_threshold_pct", 8.0)
        self.size_reduction_factor = dd.get("size_reduction_factor", 0.5)

        self.is_trading_paused = False
        self._last_warning_log: datetime | None = None
        self._warning_log_interval_seconds = dd.get("warning_log_interval_seconds", 300)  # 5 min
        logger.info(
            "DrawdownMonitor initialized (warn=%.1f%%, reduce=%.1f%%, pause=%.1f%%)",
            self._warning_pct, self._reduce_pct, self._pause_pct,
        )

    def check_drawdown(self) -> str:
        """Check current drawdown level.

        Returns: 'ok', 'warning', 'reduce_size', or 'pause_trading'
        """
        state = self.portfolio.get_state()
        dd_pct = state.current_drawdown

        if dd_pct >= self._pause_pct:
            if not self.is_trading_paused:
                logger.warning("DRAWDOWN PAUSE: %.1f%% — all new trades paused", dd_pct)
                self.is_trading_paused = True
            return "pause_trading"

        if dd_pct >= self._reduce_pct:
            logger.warning("DRAWDOWN REDUCE: %.1f%% — reducing position sizes", dd_pct)
            self.is_trading_paused = False
            return "reduce_size"

        if dd_pct >= self._warning_pct:
            now = datetime.now(timezone.utc)
            if (self._last_warning_log is None or
                    (now - self._last_warning_log).total_seconds() >= self._warning_log_interval_seconds):
                logger.info("Drawdown warning: %.1f%% (capital=%.2f)", dd_pct, state.total_capital)
                self._last_warning_log = now
            self.is_trading_paused = False
            return "warning"

        self.is_trading_paused = False
        return "ok"
