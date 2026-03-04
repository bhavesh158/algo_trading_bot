"""Risk Management Engine (PRD §12).

Enforces global risk rules:
- Max risk per trade
- Max open positions
- Rolling 24h loss limit
- Total exposure limits
- Single pair exposure limits
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from crypto.core.enums import AlertSeverity
from crypto.core.event_bus import EventBus
from crypto.core.events import AlertEvent, RiskEvent
from crypto.core.models import Alert, Signal

if TYPE_CHECKING:
    from crypto.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates trades against risk rules and blocks violations."""

    def __init__(
        self, config: dict[str, Any], event_bus: EventBus, portfolio_manager: PortfolioManager,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.portfolio = portfolio_manager

        rc = config.get("risk", {})
        self._max_risk_per_trade_pct = rc.get("max_risk_per_trade_pct", 1.0)
        self._max_open_positions = rc.get("max_open_positions", 5)
        self._max_rolling_loss_pct = rc.get("max_rolling_loss_pct", 5.0)
        self._max_total_exposure_pct = rc.get("max_total_exposure_pct", 50.0)
        self._max_single_pair_pct = rc.get("max_single_pair_exposure_pct", 20.0)

        self._loss_limit_breached = False

        logger.info(
            "RiskManager initialized (max_risk=%.1f%%, max_pos=%d, max_loss=%.1f%%)",
            self._max_risk_per_trade_pct, self._max_open_positions, self._max_rolling_loss_pct,
        )

    def can_take_trade(self, signal: Signal) -> bool:
        """Check if a new trade is allowed under current risk constraints."""
        state = self.portfolio.get_state()

        # Rolling loss limit
        if state.total_capital > 0:
            loss_pct = abs(min(state.rolling_pnl, 0)) / state.total_capital * 100
            if loss_pct >= self._max_rolling_loss_pct:
                self._publish_risk_event(
                    "rolling_loss_limit",
                    f"Rolling loss {loss_pct:.1f}% exceeds limit {self._max_rolling_loss_pct:.1f}%",
                    loss_pct, self._max_rolling_loss_pct, "pause_trading",
                )
                self._loss_limit_breached = True
                return False

        # Max open positions
        if state.open_position_count >= self._max_open_positions:
            return False

        # Total exposure
        if state.total_capital > 0:
            exposure_pct = state.total_exposure / state.total_capital * 100
            if exposure_pct >= self._max_total_exposure_pct:
                return False

        # Single pair exposure
        if state.total_capital > 0:
            pair_exposure = sum(
                p.notional_value for p in state.positions if p.symbol == signal.symbol
            )
            pair_pct = pair_exposure / state.total_capital * 100
            if pair_pct >= self._max_single_pair_pct:
                return False

        # Per-trade risk
        if signal.entry_price > 0 and signal.stop_loss > 0:
            risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
            if risk_pct > self._max_risk_per_trade_pct * 5:
                return False

        return True

    @property
    def is_loss_limit_breached(self) -> bool:
        return self._loss_limit_breached

    def reset_rolling_state(self) -> None:
        self._loss_limit_breached = False
        logger.info("Rolling risk state reset")

    def _publish_risk_event(
        self, rule: str, message: str, current: float, threshold: float, action: str,
    ) -> None:
        self.event_bus.publish(RiskEvent(
            rule=rule, message=message,
            current_value=current, threshold=threshold, action=action,
        ))
        self.event_bus.publish(AlertEvent(alert=Alert(
            severity=AlertSeverity.CRITICAL, source="RiskManager", message=message,
        )))
        logger.warning("RISK BREACH: %s", message)
