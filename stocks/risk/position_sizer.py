"""Position Sizing System (PRD §13).

Automatically determines trade sizes based on:
- Account capital
- Market volatility (ATR-based)
- Strategy confidence
- Recent performance (drawdown reduction)
"""

from __future__ import annotations

import logging
import math
from typing import Any, TYPE_CHECKING

from stocks.core.models import Signal

if TYPE_CHECKING:
    from stocks.core.event_bus import EventBus
    from stocks.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates position sizes to maintain consistent risk exposure."""

    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        portfolio_manager: PortfolioManager,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.portfolio = portfolio_manager

        ps_config = config.get("position_sizing", {})
        self._method = ps_config.get("method", "volatility_adjusted")
        self._fixed_quantity = ps_config.get("fixed_quantity", 10)
        self._max_position_pct = ps_config.get("max_position_pct", 10.0)

        risk_config = config.get("risk", {})
        self._max_risk_per_trade_pct = risk_config.get("max_risk_per_trade_pct", 1.0)

        self._drawdown_factor = 1.0  # Reduced during drawdown

        logger.info("PositionSizer initialized (method=%s)", self._method)

    def calculate_quantity(self, signal: Signal) -> int:
        """Calculate the number of shares/units to trade.

        Returns 0 if the signal doesn't have enough info for sizing.
        """
        if self._method == "fixed":
            return self._fixed_quantity
        elif self._method == "volatility_adjusted":
            return self._volatility_adjusted_size(signal)
        else:
            return self._fixed_quantity

    def _volatility_adjusted_size(self, signal: Signal) -> int:
        """Size position based on risk-per-trade and distance to stop loss.

        Quantity = (Capital × Risk%) / Risk-per-share
        Capped by max position size as % of capital.
        """
        state = self.portfolio.get_state()
        capital = state.available_capital

        if capital <= 0 or signal.entry_price <= 0:
            return 0

        # Risk per share = distance from entry to stop
        if signal.stop_loss > 0:
            risk_per_share = abs(signal.entry_price - signal.stop_loss)
        else:
            # Default: 2% of entry price as risk
            risk_per_share = signal.entry_price * 0.02

        if risk_per_share <= 0:
            return 0

        # Capital at risk = max_risk% of total capital
        capital_at_risk = capital * (self._max_risk_per_trade_pct / 100)

        # Apply drawdown factor (reduces size during drawdowns)
        capital_at_risk *= self._drawdown_factor

        # Apply confidence scaling (lower confidence = smaller size)
        confidence_factor = max(0.5, min(1.0, signal.confidence))
        capital_at_risk *= confidence_factor

        # Calculate raw quantity
        quantity = capital_at_risk / risk_per_share

        # Cap by max position size
        max_value = capital * (self._max_position_pct / 100)
        max_quantity = max_value / signal.entry_price if signal.entry_price > 0 else 0

        quantity = min(quantity, max_quantity)

        # Floor to integer, minimum 1
        result = max(1, math.floor(quantity))

        # Minimum position value: ensure commission drag doesn't dominate.
        # At 0.5% target, position_value >= min_value guarantees gross > round-trip fees.
        min_pos_value = self.config.get("position_sizing", {}).get("min_position_value_inr", 10000)
        if min_pos_value > 0 and signal.entry_price > 0:
            min_qty = math.ceil(min_pos_value / signal.entry_price)
            if result < min_qty:
                # Only raise if the larger quantity is still within the max-position cap
                max_allowed_qty = max_value / signal.entry_price if signal.entry_price > 0 else 0
                if min_qty <= max_allowed_qty:
                    logger.debug(
                        "Position size for %s raised %d→%d (min_position_value=%.0f)",
                        signal.symbol, result, min_qty, min_pos_value,
                    )
                    result = min_qty

        logger.debug(
            "Position size for %s: qty=%d (risk_per_share=%.2f, capital_at_risk=%.2f, conf=%.2f)",
            signal.symbol, result, risk_per_share, capital_at_risk, signal.confidence,
        )

        return result

    def set_drawdown_factor(self, factor: float) -> None:
        """Adjust sizing based on drawdown (called by DrawdownMonitor)."""
        self._drawdown_factor = max(0.1, min(1.0, factor))
        logger.info("Drawdown factor set to %.2f", self._drawdown_factor)
