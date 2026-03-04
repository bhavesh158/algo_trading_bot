"""Paper Trading Simulator (PRD §3).

Simulates order execution with configurable slippage and commission.
Uses real-time market data but maintains virtual capital and positions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from core.enums import OrderSide, OrderStatus, OrderType
from core.models import Order

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates order filling for paper trading mode."""

    def __init__(self, config: dict[str, Any]) -> None:
        paper_config = config.get("paper_trading", {})
        self._slippage_pct = paper_config.get("slippage_pct", 0.05)
        self._commission = paper_config.get("commission_per_trade", 20.0)

        self._orders: dict[str, Order] = {}
        logger.info(
            "PaperTrader initialized (slippage=%.2f%%, commission=%.2f)",
            self._slippage_pct, self._commission,
        )

    def submit_order(self, order: Order, current_price: float) -> Order:
        """Simulate submitting and immediately filling a market order.

        For limit orders, fills only if price is favorable.
        """
        order.placed_at = datetime.now()
        order.status = OrderStatus.SUBMITTED

        if order.order_type == OrderType.MARKET:
            # Apply slippage
            slippage = current_price * self._slippage_pct / 100
            if order.side == OrderSide.BUY:
                fill_price = current_price + slippage
            else:
                fill_price = current_price - slippage

            order.filled_price = round(fill_price, 2)
            order.filled_quantity = order.quantity
            order.filled_at = datetime.now()
            order.status = OrderStatus.FILLED

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                order.filled_price = order.price
                order.filled_quantity = order.quantity
                order.filled_at = datetime.now()
                order.status = OrderStatus.FILLED
            elif order.side == OrderSide.SELL and current_price >= order.price:
                order.filled_price = order.price
                order.filled_quantity = order.quantity
                order.filled_at = datetime.now()
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PENDING

        self._orders[order.id] = order

        if order.status == OrderStatus.FILLED:
            logger.info(
                "[PAPER] Order filled: %s %s %d @ %.2f (slippage applied)",
                order.side.name, order.symbol, order.filled_quantity, order.filled_price,
            )

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        order = self._orders.get(order_id)
        if order and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            order.status = OrderStatus.CANCELLED
            logger.info("[PAPER] Order cancelled: %s", order_id)
            return True
        return False

    @property
    def commission(self) -> float:
        return self._commission

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)
