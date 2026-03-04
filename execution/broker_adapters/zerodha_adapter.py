"""Zerodha (Kite Connect) Broker Adapter.

Stub implementation — requires kiteconnect package and valid API credentials.
Install: pip install kiteconnect
"""

from __future__ import annotations

import logging
from typing import Any

from core.enums import OrderSide, OrderStatus, OrderType
from core.models import Order
from execution.broker_adapters.base_adapter import BaseBrokerAdapter

logger = logging.getLogger(__name__)


class ZerodhaAdapter(BaseBrokerAdapter):
    """Broker adapter for Zerodha Kite Connect API."""

    def __init__(self, api_key: str, api_secret: str, access_token: str | None = None) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = access_token
        self._kite = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Kite Connect API."""
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=self._api_key)
            if self._access_token:
                self._kite.set_access_token(self._access_token)
                self._connected = True
                logger.info("Connected to Zerodha Kite API")
            else:
                logger.warning(
                    "Access token not set. Generate via: %s",
                    self._kite.login_url(),
                )
            return self._connected
        except ImportError:
            logger.error("kiteconnect package not installed. Run: pip install kiteconnect")
            return False
        except Exception:
            logger.exception("Failed to connect to Zerodha")
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._kite = None
        logger.info("Disconnected from Zerodha")

    def submit_order(self, order: Order) -> Order:
        if not self._connected or self._kite is None:
            order.status = OrderStatus.REJECTED
            return order

        try:
            kite_params = {
                "tradingsymbol": order.symbol.replace(".NS", ""),
                "exchange": "NSE",
                "transaction_type": "BUY" if order.side == OrderSide.BUY else "SELL",
                "quantity": order.quantity,
                "product": "MIS",  # Intraday
                "order_type": self._map_order_type(order.order_type),
            }
            if order.order_type == OrderType.LIMIT:
                kite_params["price"] = order.price
            if order.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT):
                kite_params["trigger_price"] = order.stop_price

            broker_id = self._kite.place_order(variety="regular", **kite_params)
            order.broker_order_id = str(broker_id)
            order.status = OrderStatus.SUBMITTED
            logger.info("[LIVE] Order submitted to Zerodha: %s", broker_id)
        except Exception:
            logger.exception("Failed to submit order to Zerodha")
            order.status = OrderStatus.REJECTED

        return order

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected or self._kite is None:
            return False
        try:
            self._kite.cancel_order(variety="regular", order_id=order_id)
            logger.info("[LIVE] Order cancelled: %s", order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def get_order_status(self, order_id: str) -> Order | None:
        if not self._connected or self._kite is None:
            return None
        try:
            history = self._kite.order_history(order_id)
            if history:
                latest = history[-1]
                order = Order(broker_order_id=order_id)
                status_map = {
                    "COMPLETE": OrderStatus.FILLED,
                    "REJECTED": OrderStatus.REJECTED,
                    "CANCELLED": OrderStatus.CANCELLED,
                    "OPEN": OrderStatus.SUBMITTED,
                }
                order.status = status_map.get(latest.get("status", ""), OrderStatus.PENDING)
                order.filled_price = float(latest.get("average_price", 0))
                order.filled_quantity = int(latest.get("filled_quantity", 0))
                return order
        except Exception:
            logger.exception("Failed to get order status for %s", order_id)
        return None

    def get_positions(self) -> list[dict]:
        if not self._connected or self._kite is None:
            return []
        try:
            positions = self._kite.positions()
            return positions.get("day", [])
        except Exception:
            logger.exception("Failed to get positions")
            return []

    def get_account_balance(self) -> float:
        if not self._connected or self._kite is None:
            return 0.0
        try:
            margins = self._kite.margins()
            equity = margins.get("equity", {})
            return float(equity.get("available", {}).get("cash", 0))
        except Exception:
            logger.exception("Failed to get account balance")
            return 0.0

    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "SL",
            OrderType.STOP_LOSS_LIMIT: "SL",
        }
        return mapping.get(order_type, "MARKET")
