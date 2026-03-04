"""Position Sizing System (PRD §13).

Determines trade sizes based on available capital, asset volatility,
and strategy confidence.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import pandas as pd

from crypto.core.event_bus import EventBus
from crypto.core.models import Signal

if TYPE_CHECKING:
    from crypto.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates position size for each trade."""

    def __init__(
        self, config: dict[str, Any], event_bus: EventBus, portfolio_manager: PortfolioManager,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.portfolio = portfolio_manager

        ps = config.get("position_sizing", {})
        self._method = ps.get("method", "volatility_adjusted")
        self._fixed_qty = ps.get("fixed_quantity", 0.001)
        self._vol_lookback = ps.get("volatility_lookback", 14)
        self._max_position_pct = ps.get("max_position_pct", 15.0) / 100
        self._max_risk_pct = config.get("risk", {}).get("max_risk_per_trade_pct", 1.0) / 100

        self._drawdown_factor = 1.0
        logger.info("PositionSizer initialized (method=%s)", self._method)

    def calculate_quantity(self, signal: Signal) -> float:
        """Calculate position quantity for a given signal."""
        if self._method == "fixed":
            return self._fixed_qty

        return self._volatility_adjusted_size(signal)

    def _volatility_adjusted_size(self, signal: Signal) -> float:
        """Size based on risk-per-trade and ATR-based stop distance."""
        state = self.portfolio.get_state()
        if state.available_capital <= 0 or signal.entry_price <= 0:
            return 0.0

        # Risk budget = available capital * max risk %
        risk_budget = state.available_capital * self._max_risk_pct * self._drawdown_factor

        # Stop distance
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance == 0:
            return 0.0

        # Quantity from risk budget
        quantity = risk_budget / stop_distance

        # Cap by max position % of capital
        max_notional = state.available_capital * self._max_position_pct
        max_qty = max_notional / signal.entry_price
        quantity = min(quantity, max_qty)

        # Scale by confidence
        quantity *= signal.confidence

        # Apply drawdown factor
        quantity *= self._drawdown_factor

        return max(quantity, 0.0)

    def set_drawdown_factor(self, factor: float) -> None:
        self._drawdown_factor = max(0.1, min(factor, 1.0))
        logger.info("Position sizing drawdown factor set to %.2f", self._drawdown_factor)
