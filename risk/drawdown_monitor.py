"""Drawdown Protection (PRD §14).

Monitors account drawdown and takes protective actions:
- Warning level: log warning
- Reduce size: cut position sizes
- Pause trading: halt all new trades
- Disable underperforming strategies
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from core.enums import AlertSeverity
from core.event_bus import EventBus
from core.events import AlertEvent, RiskEvent
from core.models import Alert

if TYPE_CHECKING:
    from portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class DrawdownMonitor:
    """Continuously monitors drawdown and triggers protective actions."""

    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        portfolio_manager: PortfolioManager,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.portfolio = portfolio_manager

        dd_config = config.get("drawdown", {})
        self._warning_pct = dd_config.get("warning_threshold_pct", 2.0)
        self._reduce_pct = dd_config.get("reduce_size_threshold_pct", 3.0)
        self._pause_pct = dd_config.get("pause_trading_threshold_pct", 5.0)
        self._reduction_factor = dd_config.get("size_reduction_factor", 0.5)

        self._is_paused = False
        self._is_size_reduced = False

        logger.info(
            "DrawdownMonitor initialized (warn=%.1f%%, reduce=%.1f%%, pause=%.1f%%)",
            self._warning_pct, self._reduce_pct, self._pause_pct,
        )

    @property
    def is_trading_paused(self) -> bool:
        return self._is_paused

    @property
    def is_size_reduced(self) -> bool:
        return self._is_size_reduced

    def check_drawdown(self) -> str:
        """Check current drawdown and return action level.

        Returns: "normal", "warning", "reduce_size", or "pause_trading"
        """
        state = self.portfolio.get_state()
        drawdown_pct = state.current_drawdown

        if drawdown_pct >= self._pause_pct:
            if not self._is_paused:
                self._is_paused = True
                logger.warning(
                    "DRAWDOWN CRITICAL: %.1f%% — trading PAUSED", drawdown_pct
                )
                self.event_bus.publish(RiskEvent(
                    rule="drawdown_pause",
                    message=f"Drawdown {drawdown_pct:.1f}% exceeds pause threshold {self._pause_pct:.1f}%",
                    current_value=drawdown_pct,
                    threshold=self._pause_pct,
                    action="pause_trading",
                ))
                self.event_bus.publish(AlertEvent(alert=Alert(
                    severity=AlertSeverity.CRITICAL,
                    source="DrawdownMonitor",
                    message=f"Trading paused: drawdown {drawdown_pct:.1f}%",
                )))
            return "pause_trading"

        elif drawdown_pct >= self._reduce_pct:
            if not self._is_size_reduced:
                self._is_size_reduced = True
                logger.warning(
                    "DRAWDOWN ELEVATED: %.1f%% — reducing position sizes by %.0f%%",
                    drawdown_pct, (1 - self._reduction_factor) * 100,
                )
                self.event_bus.publish(RiskEvent(
                    rule="drawdown_reduce",
                    message=f"Drawdown {drawdown_pct:.1f}% — reducing position sizes",
                    current_value=drawdown_pct,
                    threshold=self._reduce_pct,
                    action="reduce_size",
                ))
            return "reduce_size"

        elif drawdown_pct >= self._warning_pct:
            logger.info("Drawdown warning: %.1f%%", drawdown_pct)
            return "warning"

        else:
            # Recovery — reset flags
            if self._is_paused or self._is_size_reduced:
                logger.info(
                    "Drawdown recovered to %.1f%% — resuming normal operation",
                    drawdown_pct,
                )
                self._is_paused = False
                self._is_size_reduced = False
            return "normal"

    @property
    def size_reduction_factor(self) -> float:
        """Factor to multiply position sizes by during drawdown."""
        if self._is_paused:
            return 0.0
        if self._is_size_reduced:
            return self._reduction_factor
        return 1.0
