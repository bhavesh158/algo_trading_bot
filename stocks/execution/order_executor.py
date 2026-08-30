"""Order Execution Engine (PRD §11).

Routes orders to Paper Trader or Live Broker based on the active trading mode.
Prevents duplicate orders, confirms execution, and publishes order events.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from stocks.core.enums import OrderStatus, OrderType, TradingMode
from stocks.core.event_bus import EventBus
from stocks.core.events import OrderEvent
from stocks.core.models import Order, Signal
from stocks.execution.paper_trader import PaperTrader

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
        
        # Blacklist: symbols that should not be traded (e.g., cautionary listings)
        self._blacklisted_symbols: set[str] = set(
            config.get("execution", {}).get("blacklisted_symbols", [])
        )
        
        # Runtime cautionary blacklist: auto-add symbols that fail with AB4036
        # (cautionary listing error) so we don't keep retrying every cycle
        self._cautionary_blacklisted: set[str] = set()

        broker_cfg = config.get("broker", {})
        self._fill_timeout_seconds = broker_cfg.get("order_fill_timeout_seconds", 20)
        self._poll_interval_seconds = broker_cfg.get("order_poll_interval_seconds", 3)
        # Live-mode per-leg charge so the bot's books track the broker statement
        # (broker/exchange fees are not returned by the API, only the fill price).
        self._live_commission_per_trade = float(
            broker_cfg.get("live_commission_per_trade", 0.0)
        )

        logger.info("OrderExecutor initialized (mode=%s, blacklisted=%s)", mode.name, self._blacklisted_symbols)

    def set_broker_adapter(self, adapter: Any) -> None:
        """Set the broker adapter for live trading."""
        self._broker_adapter = adapter

    def execute_order(self, order: Order, current_price: float = 0.0) -> Order:
        """Execute an order via the appropriate execution path.

        Prevents duplicate orders for the same symbol.
        """
        # Blacklist check: skip symbols that are blocked by exchange
        if order.symbol in self._blacklisted_symbols:
            logger.warning(
                "Order blocked for %s (blacklisted by exchange)",
                order.symbol,
            )
            order.status = OrderStatus.REJECTED
            return order
        
        # Runtime cautionary blacklist: auto-block after a failed cautionary order
        if order.symbol in self._cautionary_blacklisted:
            logger.warning(
                "Order blocked for %s (cautionary listing detected this session)",
                order.symbol,
            )
            order.status = OrderStatus.REJECTED
            return order
        
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

        if self.mode == TradingMode.LIVE and order.status in (
            OrderStatus.PENDING, OrderStatus.SUBMITTED,
        ):
            order = self._confirm_fill(order)
            self._recent_orders[order.id] = order

        # Auto-blacklist cautionary-listed symbols (AB4036) so we don't keep retrying
        if order.status == OrderStatus.REJECTED and order.error_code == "AB4036":
            self._cautionary_blacklisted.add(order.symbol)
            logger.warning(
                "Auto-blacklisting %s for session (cautionary listing by exchange)",
                order.symbol,
            )

        if order.status == OrderStatus.FILLED:
            self._active_order_symbols.add(order.symbol)

        # Publish order event
        self.event_bus.publish(OrderEvent(order=order))

        return order

    def _confirm_fill(self, order: Order) -> Order:
        """Poll the broker until the order fills, rejects, or times out (live mode).

        Angel One does not return fill details from placeOrder, so we poll
        get_order_status. On timeout the order is cancelled to prevent the
        duplicate-order guard from being bypassed — UNLESS we actually observed
        a fill during polling (Angel One may move a filled order into its trade
        book before our poll lands, making the order book report nothing).
        """
        order_id = order.broker_order_id or order.id
        saw_fill = False
        deadline = time.monotonic() + self._fill_timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(self._poll_interval_seconds)
            try:
                status = self._broker_adapter.get_order_status(order_id)
            except Exception:
                logger.exception("Failed to poll order status for %s", order_id)
                continue
            if status is None:
                continue
            order.status = status.status
            if status.filled_price > 0:
                order.filled_price = status.filled_price
                saw_fill = True
            if status.filled_quantity > 0:
                order.filled_quantity = status.filled_quantity
                saw_fill = True
            if order.status in (
                OrderStatus.FILLED, OrderStatus.REJECTED,
                OrderStatus.CANCELLED, OrderStatus.EXPIRED,
            ):
                break

        if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            if saw_fill:
                # Confirmation was missed but the order clearly executed.
                logger.warning(
                    "Order %s not marked complete but fill observed "
                    "(px=%.2f, qty=%d) — accepting as FILLED",
                    order_id, order.filled_price, order.filled_quantity,
                )
                order.status = OrderStatus.FILLED
            else:
                logger.warning(
                    "Order %s still unfilled after %.0fs — cancelling to prevent duplicates",
                    order_id, self._fill_timeout_seconds,
                )
                self.cancel_order(order.id)
                order.status = OrderStatus.CANCELLED
        return order

    def get_broker_positions(self) -> Optional[dict[str, int]]:
        """Get all open positions from the broker, mapped as {symbol: netqty}.
        
        Symbol format: Yahoo-style (e.g., RELIANCE.NS).
        Only returns positions with non-zero quantity.

        Returns None when the broker query failed, so calling code can avoid
        mistaking a failed query for an empty position book (which would
        otherwise cause real positions to be dropped as 'ghosts' and re-entered).
        """
        if self.mode != TradingMode.LIVE or self._broker_adapter is None:
            return {}
        try:
            positions = self._broker_adapter.get_positions()
        except Exception:
            logger.exception("Failed to query broker positions")
            return None
        if positions is None:
            return None
        result: dict[str, int] = {}
        if not positions:
            return result
        for p in positions:
            try:
                netqty = int(p.get("netqty", 0))
                if netqty == 0:
                    continue
                tradingsymbol = p.get("tradingsymbol", "")
                # Convert from Angel One format (RELIANCE-EQ) to Yahoo (RELIANCE.NS)
                symbol = tradingsymbol.replace("-EQ", ".NS")
                result[symbol] = netqty
            except (TypeError, ValueError):
                continue
        return result

    def position_still_open(self, symbol: str) -> bool:
        """Check the broker's position book for an open position in `symbol`.

        Used as a reconciliation fallback: if an exit order's confirmation was
        missed but the broker no longer holds the position, we know it closed.
        """
        if self.mode != TradingMode.LIVE or self._broker_adapter is None:
            return False
        try:
            positions = self._broker_adapter.get_positions()
        except Exception:
            logger.exception("Failed to query broker positions for %s", symbol)
            return False
        trading_symbol = symbol.replace(".NS", "-EQ")
        if not positions:
            return False
        for p in positions:
            if p.get("tradingsymbol") != trading_symbol:
                continue
            try:
                if int(p.get("netqty", 0)) != 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

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

    def register_active_symbol(self, symbol: str) -> None:
        """Register a symbol as having an active order/position (used after a failed exit)."""
        self._active_order_symbols.add(symbol)

    @property
    def commission(self) -> float:
        """Get the commission per trade for the current mode."""
        if self.mode == TradingMode.PAPER:
            return self._paper_trader.commission
        return self._live_commission_per_trade  # Live: flat per-leg broker charge

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
