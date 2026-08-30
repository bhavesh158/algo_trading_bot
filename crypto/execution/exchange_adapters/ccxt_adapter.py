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
        # exchange_order_id -> symbol (needed for fetch/cancel with symbol)
        self._order_symbols: dict[str, str] = {}

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
        self._order_symbols.clear()
        logger.info("Exchange adapter disconnected")

    def submit_order(self, order: Order) -> Order:
        if not self._connected or self._exchange is None:
            order.status = OrderStatus.REJECTED
            return order

        try:
            side = "buy" if order.side == OrderSide.BUY else "sell"
            order_type = self._map_order_type(order.order_type)

            # Round quantity to the exchange lot-size step and enforce minimums
            quantity = self._sanitize_quantity(order, order.quantity)
            if quantity <= 0:
                logger.error(
                    "[LIVE] Order rejected: %s %s qty=%.8f below exchange minimum "
                    "or invalid after lot-size rounding",
                    side, order.symbol, order.quantity,
                )
                order.status = OrderStatus.REJECTED
                return order

            params = {}
            # Stop price only for STOP_LOSS / STOP_LOSS_LIMIT orders
            # MARKET orders cannot have stop prices on Binance (would trigger immediately)
            if order.stop_price > 0 and order.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT):
                params["stopPrice"] = order.stop_price

            result = self._exchange.create_order(
                symbol=order.symbol,
                type=order_type,
                side=side,
                amount=quantity,
                price=order.price if order.order_type == OrderType.LIMIT else None,
                params=params,
            )

            order.exchange_order_id = str(result.get("id", ""))
            order.placed_at = datetime.now(timezone.utc)
            if order.exchange_order_id:
                self._order_symbols[order.exchange_order_id] = order.symbol

            # Binance market orders fill synchronously — read the fill from the response
            filled_qty = float(result.get("filled", 0) or 0)
            avg_price = float(result.get("average", 0) or 0)
            if avg_price <= 0 and filled_qty > 0:
                cost = float(result.get("cost", 0) or 0)
                if cost > 0:
                    avg_price = cost / filled_qty
            if filled_qty > 0 and avg_price > 0:
                order.filled_quantity = filled_qty
                order.filled_price = avg_price
                order.filled_at = datetime.now(timezone.utc)

            result_status = str(result.get("status", "")).lower()
            if result_status in ("closed", "filled"):
                order.status = OrderStatus.FILLED
            elif result_status in ("canceled", "cancelled"):
                order.status = OrderStatus.CANCELLED
            elif result_status == "expired":
                order.status = OrderStatus.EXPIRED
            elif result_status in ("rejected", "failed"):
                order.status = OrderStatus.REJECTED
            else:
                order.status = OrderStatus.SUBMITTED

            if order.status == OrderStatus.FILLED:
                logger.info(
                    "[LIVE] Order FILLED: %s %s qty=%.8f @ %.6f (id=%s)",
                    side, order.symbol, order.filled_quantity, order.filled_price,
                    order.exchange_order_id,
                )
            else:
                logger.info(
                    "[LIVE] Order %s: %s %s qty=%.8f (id=%s)",
                    order.status.name, side, order.symbol, quantity, order.exchange_order_id,
                )

        except Exception:
            logger.exception("Failed to submit order")
            order.status = OrderStatus.REJECTED

        return order

    def _sanitize_quantity(self, order: Order, quantity: float) -> float:
        """Round quantity to the exchange lot-size step and enforce minimums.

        ccxt's ``amount_to_precision`` truncates to the lot step, which is the
        safe direction for buys (never exceeds available funds). Returns 0.0
        when the order cannot be placed (below min amount or min notional).
        """
        if quantity <= 0 or self._exchange is None:
            return 0.0
        try:
            qty = float(self._exchange.amount_to_precision(order.symbol, quantity))
        except Exception:
            logger.exception("Failed to round quantity for %s", order.symbol)
            return 0.0
        if qty <= 0:
            return 0.0

        try:
            market = self._exchange.market(order.symbol)
        except Exception:
            logger.exception("Failed to load market %s for minimum checks", order.symbol)
            return 0.0

        limits = market.get("limits", {}) or {}
        min_amount = (limits.get("amount") or {}).get("min") or 0
        if min_amount and qty < min_amount:
            logger.warning(
                "[LIVE] %s qty=%.8f < minAmount=%.8f — order rejected",
                order.symbol, qty, min_amount,
            )
            return 0.0

        # Min notional (cost) — use order price, fall back to last ticker price
        min_cost = (limits.get("cost") or {}).get("min") or 0
        if min_cost:
            price = order.price
            if price <= 0:
                try:
                    ticker = self._exchange.fetch_ticker(order.symbol)
                    price = float(ticker.get("last", 0) or 0)
                except Exception:
                    logger.exception("Failed to fetch ticker for %s min-notional check", order.symbol)
                    price = 0.0
            if price > 0 and qty * price < min_cost:
                logger.warning(
                    "[LIVE] %s notional=%.4f < minNotional=%.4f — order rejected",
                    order.symbol, qty * price, min_cost,
                )
                return 0.0
        return qty

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected or self._exchange is None:
            return False
        try:
            symbol = self._order_symbols.get(order_id)
            if symbol:
                self._exchange.cancel_order(order_id, symbol)
            else:
                self._exchange.cancel_order(order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def get_order_status(self, order_id: str) -> Order | None:
        if not self._connected or self._exchange is None:
            return None
        try:
            symbol = self._order_symbols.get(order_id)
            if symbol:
                try:
                    o = self._exchange.fetch_order(order_id, symbol)
                except Exception:
                    o = None
            else:
                o = self._scan_for_order(order_id)
            if o is None:
                logger.warning("Order %s not found on exchange", order_id)
                return None

            order = Order(exchange_order_id=order_id, symbol=symbol or "")
            status_map = {
                "open": OrderStatus.SUBMITTED,
                "closed": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "cancelled": OrderStatus.CANCELLED,
                "expired": OrderStatus.EXPIRED,
                "rejected": OrderStatus.REJECTED,
            }
            order.status = status_map.get(str(o.get("status", "")).lower(), OrderStatus.PENDING)
            order.filled_price = float(o.get("average", 0) or 0)
            order.filled_quantity = float(o.get("filled", 0) or 0)
            return order
        except Exception:
            logger.exception("Failed to get order status %s", order_id)
        return None

    def _scan_for_order(self, order_id: str) -> dict | None:
        """Search open then closed orders for a matching id (symbol unknown)."""
        for fetch in ("fetch_open_orders", "fetch_closed_orders"):
            try:
                orders = getattr(self._exchange, fetch)()
            except Exception:
                continue
            for o in orders or []:
                if str(o.get("id")) == order_id:
                    return o
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
