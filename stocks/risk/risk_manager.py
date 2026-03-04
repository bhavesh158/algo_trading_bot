"""Risk Management Engine (PRD §12).

Enforces global risk rules:
- Max risk per trade
- Max open positions
- Daily loss limits
- Total exposure limits
- Single stock exposure limits
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from stocks.core.enums import AlertSeverity
from stocks.core.event_bus import EventBus
from stocks.core.events import AlertEvent, RiskEvent
from stocks.core.models import Alert, Signal

if TYPE_CHECKING:
    from stocks.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates trades against risk rules and blocks violations."""

    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        portfolio_manager: PortfolioManager,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.portfolio = portfolio_manager

        risk_config = config.get("risk", {})
        self._max_risk_per_trade_pct = risk_config.get("max_risk_per_trade_pct", 1.0)
        self._max_open_positions = risk_config.get("max_open_positions", 5)
        self._max_daily_loss_pct = risk_config.get("max_daily_loss_pct", 3.0)
        self._max_total_exposure_pct = risk_config.get("max_total_exposure_pct", 50.0)
        self._max_single_stock_pct = risk_config.get("max_single_stock_exposure_pct", 15.0)
        self._min_rr_ratio = risk_config.get("min_risk_reward_ratio", 1.5)

        self._daily_loss_breached = False

        logger.info(
            "RiskManager initialized (max_risk=%.1f%%, max_positions=%d, max_daily_loss=%.1f%%)",
            self._max_risk_per_trade_pct, self._max_open_positions, self._max_daily_loss_pct,
        )

    def can_take_trade(self, signal: Signal) -> bool:
        """Check if a new trade is allowed under current risk constraints.

        Returns True if the trade passes all risk checks.
        """
        state = self.portfolio.get_state()

        # Check: daily loss limit
        if state.total_capital > 0:
            daily_loss_pct = abs(min(state.daily_pnl, 0)) / state.total_capital * 100
            if daily_loss_pct >= self._max_daily_loss_pct:
                self._publish_risk_event(
                    "daily_loss_limit",
                    f"Daily loss {daily_loss_pct:.1f}% exceeds limit {self._max_daily_loss_pct:.1f}%",
                    daily_loss_pct, self._max_daily_loss_pct,
                    "pause_trading",
                )
                self._daily_loss_breached = True
                return False

        # Check: max open positions
        if state.open_position_count >= self._max_open_positions:
            logger.debug(
                "Max positions reached (%d/%d)",
                state.open_position_count, self._max_open_positions,
            )
            return False

        # Check: total exposure limit
        if state.total_capital > 0:
            exposure_pct = state.total_exposure / state.total_capital * 100
            if exposure_pct >= self._max_total_exposure_pct:
                logger.debug(
                    "Total exposure %.1f%% exceeds limit %.1f%%",
                    exposure_pct, self._max_total_exposure_pct,
                )
                return False

        # Check: single stock exposure
        if state.total_capital > 0:
            symbol_exposure = sum(
                abs(p.entry_price * p.quantity)
                for p in state.positions if p.symbol == signal.symbol
            )
            symbol_pct = symbol_exposure / state.total_capital * 100
            if symbol_pct >= self._max_single_stock_pct:
                logger.debug(
                    "Single stock exposure %.1f%% for %s exceeds limit %.1f%%",
                    symbol_pct, signal.symbol, self._max_single_stock_pct,
                )
                return False

        # Check: per-trade risk
        if signal.entry_price > 0 and signal.stop_loss > 0:
            risk_per_share = abs(signal.entry_price - signal.stop_loss)
            # This is a pre-check; actual position size is determined by position_sizer
            if risk_per_share / signal.entry_price * 100 > self._max_risk_per_trade_pct * 5:
                logger.debug(
                    "Per-share risk too high: %.2f%% of entry price",
                    risk_per_share / signal.entry_price * 100,
                )
                return False

        return True

    @property
    def is_daily_loss_breached(self) -> bool:
        return self._daily_loss_breached

    def reset_daily_state(self) -> None:
        """Reset daily risk counters (called at start of each trading day)."""
        self._daily_loss_breached = False
        logger.info("Daily risk state reset")

    def _publish_risk_event(
        self, rule: str, message: str, current: float, threshold: float, action: str
    ) -> None:
        self.event_bus.publish(RiskEvent(
            rule=rule, message=message,
            current_value=current, threshold=threshold,
            action=action,
        ))
        self.event_bus.publish(AlertEvent(alert=Alert(
            severity=AlertSeverity.CRITICAL,
            source="RiskManager",
            message=message,
        )))
        logger.warning("RISK BREACH: %s", message)
