"""Paper Trading Simulator (PRD §3).

Simulates trade execution with configurable slippage and maker/taker fees.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any

from crypto.core.enums import OrderSide, OrderStatus
from crypto.core.models import Order

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates order execution for paper trading mode."""

    def __init__(self, config: dict[str, Any]) -> None:
        pt = config.get("paper_trading", {})
        self._slippage_pct = pt.get("slippage_pct", 0.05) / 100
        self._maker_fee_pct = pt.get("maker_fee_pct", 0.1) / 100
        self._taker_fee_pct = pt.get("taker_fee_pct", 0.1) / 100
        logger.info(
            "PaperTrader initialized (slippage=%.3f%%, taker_fee=%.2f%%)",
            self._slippage_pct * 100, self._taker_fee_pct * 100,
        )

    def submit_order(self, order: Order, current_price: float = 0.0) -> Order:
        """Simulate order fill with slippage."""
        price = current_price if current_price > 0 else order.price
        if price <= 0:
            order.status = OrderStatus.REJECTED
            return order

        # Apply random slippage
        slippage_direction = 1 if order.side == OrderSide.BUY else -1
        slippage = price * self._slippage_pct * random.uniform(0, 1) * slippage_direction
        fill_price = price + slippage

        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc)
        order.placed_at = order.filled_at

        logger.info(
            "[PAPER] %s %s qty=%.6f @ %.4f (slippage=%.4f)",
            order.side.name, order.symbol, order.quantity, fill_price, slippage,
        )
        return order

    @property
    def commission_rate(self) -> float:
        """Taker fee rate as a fraction."""
        return self._taker_fee_pct

    def calculate_commission(self, notional: float) -> float:
        """Calculate commission for a given notional trade value."""
        return notional * self._taker_fee_pct

    def cancel_order(self, order_id: str) -> bool:
        logger.info("[PAPER] Order cancelled: %s", order_id)
        return True
