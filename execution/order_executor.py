"""Order Execution Engine (PRD §11).

Routes orders to Paper Trader or Live Broker based on the active trading mode.
Prevents duplicate orders, confirms execution, and publishes order events.
"""

from __future__ import annotations

import logging
from typing import Any

from core.enums import OrderStatus, OrderType, TradingMode
from core.event_bus import EventBus
from core.events import OrderEvent
from core.models import Order, Signal
from execution.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Unified order execution interface for both paper and live trading."""

    def __init__(
        self, config: dict[str, Any], event_bus: EventBus, mode: TradingMode,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.mode = mode

        self._paper_trader = PaperTrader(config)
        self._broker_adapter = None  # Set externally for live mode
        self._recent_orders: dict[str, Order] = {}
        self._active_order_symbols: set[str] = set()  # Prevent duplicates

        logger.info("OrderExecutor initialized (mode=%s)", mode.name)

    def set_broker_adapter(self, adapter: Any) -> None:
        """Set the broker adapter for live trading."""
        self._broker_adapter = adapter

    def execute_order(self, order: Order, current_price: float = 0.0) -> Order:
        """Execute an order via the appropriate execution path.

        Prevents duplicate orders for the same symbol.
        """
        # Duplicate prevention
        if order.symbol in self._active_order_symbols:
            logger.warning(
                "Duplicate order blocked for %s (already has active order)",
                order.symbol,
            )
            order.status = OrderStatus.REJECTED
            return order

        if self.mode == TradingMode.PAPER:
            order = self._paper_trader.submit_order(order, current_price)
        elif self.mode == TradingMode.LIVE:
            if self._broker_adapter is None:
                logger.error("No broker adapter set for live trading")
                order.status = OrderStatus.REJECTED
                return order
            order = self._broker_adapter.submit_order(order)

        self._recent_orders[order.id] = order

        if order.status == OrderStatus.FILLED:
            self._active_order_symbols.add(order.symbol)

        # Publish order event
        self.event_bus.publish(OrderEvent(order=order))

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self._recent_orders.get(order_id)
        if not order:
            logger.warning("Order %s not found for cancellation", order_id)
            return False

        if self.mode == TradingMode.PAPER:
            success = self._paper_trader.cancel_order(order_id)
        else:
            if self._broker_adapter and order.broker_order_id:
                success = self._broker_adapter.cancel_order(order.broker_order_id)
            else:
                success = False

        if success:
            self._active_order_symbols.discard(order.symbol)
            self.event_bus.publish(OrderEvent(order=order))

        return success

    def release_symbol(self, symbol: str) -> None:
        """Release a symbol from the duplicate-prevention set (on position close)."""
        self._active_order_symbols.discard(symbol)

    @property
    def commission(self) -> float:
        """Get the commission per trade for the current mode."""
        if self.mode == TradingMode.PAPER:
            return self._paper_trader.commission
        return 0.0  # Live broker commissions handled by broker

    def cancel_all_pending(self) -> int:
        """Cancel all pending orders. Returns count of cancelled orders."""
        cancelled = 0
        for order_id, order in list(self._recent_orders.items()):
            if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                if self.cancel_order(order_id):
                    cancelled += 1
        if cancelled:
            logger.info("Cancelled %d pending orders", cancelled)
        return cancelled
