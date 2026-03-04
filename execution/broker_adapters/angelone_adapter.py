"""Angel One (SmartAPI) Broker Adapter.

Requires: pip install smartapi-python
Credentials needed (as env vars):
    BROKER_API_KEY      - Angel One API key
    BROKER_CLIENT_ID    - Your Angel One client ID (e.g., "A12345")
    BROKER_PASSWORD     - Your trading password
    BROKER_TOTP_SECRET  - TOTP secret for 2FA (from Angel One app setup)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from core.enums import OrderSide, OrderStatus, OrderType
from core.models import Order
from execution.broker_adapters.base_adapter import BaseBrokerAdapter

logger = logging.getLogger(__name__)


class AngelOneAdapter(BaseBrokerAdapter):
    """Broker adapter for Angel One SmartAPI."""

    def __init__(
        self,
        api_key: str,
        client_id: str,
        password: str,
        totp_secret: str,
    ) -> None:
        self._api_key = api_key
        self._client_id = client_id
        self._password = password
        self._totp_secret = totp_secret
        self._smart_api = None
        self._connected = False
        self._auth_token = None
        self._feed_token = None

    def connect(self) -> bool:
        """Connect to Angel One SmartAPI and generate session."""
        try:
            from SmartApi import SmartConnect
            import pyotp

            self._smart_api = SmartConnect(api_key=self._api_key)

            # Generate TOTP for 2FA
            totp = pyotp.TOTP(self._totp_secret).now()

            # Login
            session = self._smart_api.generateSession(
                self._client_id,
                self._password,
                totp,
            )

            if session.get("status"):
                self._auth_token = session["data"]["jwtToken"]
                self._feed_token = self._smart_api.getfeedToken()
                self._connected = True
                logger.info("Connected to Angel One SmartAPI (client=%s)", self._client_id)
                return True
            else:
                logger.error("Angel One login failed: %s", session.get("message", "Unknown error"))
                return False

        except ImportError:
            logger.error("smartapi-python not installed. Run: pip install smartapi-python pyotp")
            return False
        except Exception:
            logger.exception("Failed to connect to Angel One")
            return False

    def disconnect(self) -> None:
        """Logout from Angel One."""
        if self._smart_api and self._connected:
            try:
                self._smart_api.terminateSession(self._client_id)
            except Exception:
                logger.exception("Error during logout")
        self._connected = False
        self._smart_api = None
        logger.info("Disconnected from Angel One")

    def submit_order(self, order: Order) -> Order:
        """Place an order via Angel One SmartAPI."""
        if not self._connected or self._smart_api is None:
            order.status = OrderStatus.REJECTED
            return order

        try:
            # Convert symbol from Yahoo format (RELIANCE.NS -> RELIANCE-EQ)
            trading_symbol = order.symbol.replace(".NS", "-EQ")

            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": trading_symbol,
                "symboltoken": self._get_symbol_token(trading_symbol),
                "transactiontype": "BUY" if order.side == OrderSide.BUY else "SELL",
                "exchange": "NSE",
                "ordertype": self._map_order_type(order.order_type),
                "producttype": "INTRADAY",
                "duration": "DAY",
                "quantity": str(order.quantity),
            }

            if order.order_type == OrderType.LIMIT:
                order_params["price"] = str(order.price)
            if order.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT):
                order_params["triggerprice"] = str(order.stop_price)

            response = self._smart_api.placeOrder(order_params)

            if response:
                order.broker_order_id = str(response)
                order.status = OrderStatus.SUBMITTED
                order.placed_at = datetime.now()
                logger.info("[LIVE] Order submitted to Angel One: %s", response)
            else:
                order.status = OrderStatus.REJECTED
                logger.error("[LIVE] Order rejected by Angel One")

        except Exception:
            logger.exception("Failed to submit order to Angel One")
            order.status = OrderStatus.REJECTED

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self._connected or self._smart_api is None:
            return False
        try:
            response = self._smart_api.cancelOrder(order_id, "NORMAL")
            logger.info("[LIVE] Order cancelled: %s", order_id)
            return bool(response)
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def get_order_status(self, order_id: str) -> Order | None:
        """Get order status from Angel One."""
        if not self._connected or self._smart_api is None:
            return None
        try:
            order_book = self._smart_api.orderBook()
            if not order_book or not order_book.get("data"):
                return None

            for entry in order_book["data"]:
                if str(entry.get("orderid")) == order_id:
                    order = Order(broker_order_id=order_id)
                    status_map = {
                        "complete": OrderStatus.FILLED,
                        "rejected": OrderStatus.REJECTED,
                        "cancelled": OrderStatus.CANCELLED,
                        "open": OrderStatus.SUBMITTED,
                        "pending": OrderStatus.PENDING,
                    }
                    angel_status = entry.get("orderstatus", "").lower()
                    order.status = status_map.get(angel_status, OrderStatus.PENDING)
                    order.filled_price = float(entry.get("averageprice", 0))
                    order.filled_quantity = int(entry.get("filledshares", 0))
                    return order

        except Exception:
            logger.exception("Failed to get order status for %s", order_id)
        return None

    def get_positions(self) -> list[dict]:
        """Get open positions."""
        if not self._connected or self._smart_api is None:
            return []
        try:
            positions = self._smart_api.position()
            return positions.get("data", []) if positions else []
        except Exception:
            logger.exception("Failed to get positions")
            return []

    def get_account_balance(self) -> float:
        """Get available margin/balance."""
        if not self._connected or self._smart_api is None:
            return 0.0
        try:
            rms = self._smart_api.rmsLimit()
            if rms and rms.get("data"):
                return float(rms["data"].get("availablecash", 0))
        except Exception:
            logger.exception("Failed to get account balance")
        return 0.0

    def is_connected(self) -> bool:
        return self._connected

    def _get_symbol_token(self, trading_symbol: str) -> str:
        """Look up the symbol token for a trading symbol.

        Angel One requires a numeric token for each symbol.
        In production, load the full instrument list from:
        https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
        """
        # Common Nifty 50 tokens — extend as needed or load from master file
        token_map = {
            "RELIANCE-EQ": "2885",
            "TCS-EQ": "11536",
            "HDFCBANK-EQ": "1333",
            "INFY-EQ": "1594",
            "ICICIBANK-EQ": "4963",
            "HINDUNILVR-EQ": "1394",
            "ITC-EQ": "1660",
            "SBIN-EQ": "3045",
            "BHARTIARTL-EQ": "10604",
            "KOTAKBANK-EQ": "1922",
            "LT-EQ": "11483",
            "AXISBANK-EQ": "5900",
            "BAJFINANCE-EQ": "317",
            "ASIANPAINT-EQ": "236",
            "MARUTI-EQ": "10999",
            "TITAN-EQ": "3506",
            "SUNPHARMA-EQ": "3351",
            "ULTRACEMCO-EQ": "11532",
            "NESTLEIND-EQ": "17963",
            "WIPRO-EQ": "3787",
            "HCLTECH-EQ": "7229",
            "M&M-EQ": "2031",
            "NTPC-EQ": "11630",
            "POWERGRID-EQ": "14977",
            "TATASTEEL-EQ": "3499",
            "INDUSINDBK-EQ": "5258",
            "BAJAJFINSV-EQ": "16675",
            "JSWSTEEL-EQ": "11723",
            "ADANIPORTS-EQ": "15083",
            "TRENT-EQ": "1964",
        }
        token = token_map.get(trading_symbol, "")
        if not token:
            logger.warning("No symbol token found for %s", trading_symbol)
        return token

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOPLOSS_MARKET",
            OrderType.STOP_LOSS_LIMIT: "STOPLOSS_LIMIT",
        }
        return mapping.get(order_type, "MARKET")
