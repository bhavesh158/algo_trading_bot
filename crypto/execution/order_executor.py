"""Order Execution Engine (PRD §11).

Routes orders to Paper Trader or Live Exchange based on trading mode.
"""

from __future__ import annotations

import logging
from typing import Any

from crypto.core.enums import OrderStatus, TradingMode
from crypto.core.event_bus import EventBus
from crypto.core.events import OrderEvent
from crypto.core.models import Order
from crypto.execution.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Unified order execution interface for paper and live trading."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus, mode: TradingMode) -> None:
        self.config = config
        self.event_bus = event_bus
        self.mode = mode

        self._paper_trader = PaperTrader(config)
        self._exchange_adapter = None
        self._recent_orders: dict[str, Order] = {}
        self._active_order_symbols: set[str] = set()

        logger.info("OrderExecutor initialized (mode=%s)", mode.name)

    def set_exchange_adapter(self, adapter: Any) -> None:
        self._exchange_adapter = adapter

    def execute_order(self, order: Order, current_price: float = 0.0) -> Order:
        """Execute an order via the appropriate execution path."""
        if order.symbol in self._active_order_symbols:
            order.status = OrderStatus.REJECTED
            return order

        if self.mode == TradingMode.PAPER:
            order = self._paper_trader.submit_order(order, current_price)
        elif self.mode == TradingMode.LIVE:
            if self._exchange_adapter is None:
                logger.error("No exchange adapter set for live trading")
                order.status = OrderStatus.REJECTED
                return order
            order = self._exchange_adapter.submit_order(order)

        self._recent_orders[order.id] = order

        if order.status == OrderStatus.FILLED:
            self._active_order_symbols.add(order.symbol)

        self.event_bus.publish(OrderEvent(order=order))
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._recent_orders.get(order_id)
        if not order:
            return False

        if self.mode == TradingMode.PAPER:
            success = self._paper_trader.cancel_order(order_id)
        else:
            if self._exchange_adapter and order.exchange_order_id:
                success = self._exchange_adapter.cancel_order(order.exchange_order_id)
            else:
                success = False

        if success:
            self._active_order_symbols.discard(order.symbol)
        return success

    def release_symbol(self, symbol: str) -> None:
        self._active_order_symbols.discard(symbol)

    def get_commission(self, notional: float) -> float:
        if self.mode == TradingMode.PAPER:
            return self._paper_trader.calculate_commission(notional)
        return 0.0

    def cancel_all_pending(self) -> int:
        cancelled = 0
        for oid, order in list(self._recent_orders.items()):
            if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                if self.cancel_order(oid):
                    cancelled += 1
        return cancelled
