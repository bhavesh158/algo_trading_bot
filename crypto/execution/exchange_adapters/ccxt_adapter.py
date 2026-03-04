"""Live Exchange Adapter using ccxt.

Connects to any ccxt-supported exchange for live order execution.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from crypto.core.enums import OrderSide, OrderStatus, OrderType
from crypto.core.models import Order
from crypto.execution.exchange_adapters.base_adapter import BaseExchangeAdapter

logger = logging.getLogger(__name__)


class CcxtExchangeAdapter(BaseExchangeAdapter):
    """Live exchange adapter using the ccxt library."""

    def __init__(self, exchange_name: str, config: dict[str, Any]) -> None:
        self._exchange_name = exchange_name
        self._config = config
        self._exchange = None
        self._connected = False

    def connect(self) -> bool:
        try:
            import os
            import ccxt

            exchange_class = getattr(ccxt, self._exchange_name, None)
            if exchange_class is None:
                logger.error("Unsupported exchange: %s", self._exchange_name)
                return False

            params = {
                "apiKey": os.environ.get("EXCHANGE_API_KEY", ""),
                "secret": os.environ.get("EXCHANGE_API_SECRET", ""),
                "enableRateLimit": True,
            }
            password = os.environ.get("EXCHANGE_PASSWORD", "")
            if password:
                params["password"] = password

            self._exchange = exchange_class(params)
            if self._config.get("exchange", {}).get("sandbox", False):
                self._exchange.set_sandbox_mode(True)

            self._exchange.load_markets()
            self._connected = True
            logger.info("Exchange adapter connected: %s", self._exchange_name)
            return True

        except Exception:
            logger.exception("Failed to connect exchange adapter")
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._exchange = None
        logger.info("Exchange adapter disconnected")

    def submit_order(self, order: Order) -> Order:
        if not self._connected or self._exchange is None:
            order.status = OrderStatus.REJECTED
            return order

        try:
            side = "buy" if order.side == OrderSide.BUY else "sell"
            order_type = self._map_order_type(order.order_type)

            params = {}
            if order.stop_price > 0:
                params["stopPrice"] = order.stop_price

            result = self._exchange.create_order(
                symbol=order.symbol,
                type=order_type,
                side=side,
                amount=order.quantity,
                price=order.price if order.order_type == OrderType.LIMIT else None,
                params=params,
            )

            order.exchange_order_id = str(result.get("id", ""))
            order.status = OrderStatus.SUBMITTED
            order.placed_at = datetime.now(timezone.utc)
            logger.info("[LIVE] Order submitted: %s %s %.6f %s",
                        side, order.symbol, order.quantity, order.exchange_order_id)

        except Exception:
            logger.exception("Failed to submit order")
            order.status = OrderStatus.REJECTED

        return order

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected or self._exchange is None:
            return False
        try:
            self._exchange.cancel_order(order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def get_order_status(self, order_id: str) -> Order | None:
        if not self._connected or self._exchange is None:
            return None
        try:
            # Need symbol to fetch order; iterate open orders
            orders = self._exchange.fetch_open_orders()
            for o in orders:
                if str(o.get("id")) == order_id:
                    order = Order(exchange_order_id=order_id)
                    status_map = {
                        "open": OrderStatus.SUBMITTED,
                        "closed": OrderStatus.FILLED,
                        "canceled": OrderStatus.CANCELLED,
                        "expired": OrderStatus.EXPIRED,
                    }
                    order.status = status_map.get(o.get("status", ""), OrderStatus.PENDING)
                    order.filled_price = float(o.get("average", 0) or 0)
                    order.filled_quantity = float(o.get("filled", 0) or 0)
                    return order
        except Exception:
            logger.exception("Failed to get order status %s", order_id)
        return None

    def get_positions(self) -> list[dict]:
        if not self._connected or self._exchange is None:
            return []
        try:
            balance = self._exchange.fetch_balance()
            positions = []
            for currency, info in balance.get("total", {}).items():
                if info and float(info) > 0 and currency != "USDT":
                    positions.append({"currency": currency, "amount": float(info)})
            return positions
        except Exception:
            logger.exception("Failed to get positions")
            return []

    def get_balance(self) -> dict[str, float]:
        if not self._connected or self._exchange is None:
            return {}
        try:
            balance = self._exchange.fetch_balance()
            return {k: float(v) for k, v in balance.get("free", {}).items() if v and float(v) > 0}
        except Exception:
            logger.exception("Failed to get balance")
            return {}

    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        return {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "market",
            OrderType.STOP_LOSS_LIMIT: "limit",
        }.get(order_type, "market")
