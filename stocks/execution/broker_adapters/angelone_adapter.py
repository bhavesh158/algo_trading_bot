"""Angel One (SmartAPI) Broker Adapter.

Requires: pip install smartapi-python
Credentials needed (as env vars):
    BROKER_API_KEY      - Angel One API key
    BROKER_CLIENT_ID    - Your Angel One client ID (e.g., "A12345")
    BROKER_PASSWORD     - Your trading password
    BROKER_TOTP_SECRET  - TOTP secret for 2FA (from Angel One app setup)

The full NSE symbol->token map is loaded from Angel One's public
ScripMaster file (cached to disk) so any NSE symbol can be traded, not
just a hardcoded subset.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

from stocks.core.enums import OrderSide, OrderStatus, OrderType
from stocks.core.models import Order
from stocks.execution.broker_adapters.base_adapter import BaseBrokerAdapter

logger = logging.getLogger(__name__)

_SCRIPMASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
_SCRIPMASTER_TTL_SECONDS = 24 * 3600


class AngelOneAdapter(BaseBrokerAdapter):
    """Broker adapter for Angel One SmartAPI."""

    def __init__(
        self,
        api_key: str,
        client_id: str,
        password: str,
        totp_secret: str,
        data_dir: str = "data_store",
    ) -> None:
        self._api_key = api_key
        self._client_id = client_id
        self._password = password
        self._totp_secret = totp_secret
        self._data_dir = data_dir
        self._smart_api = None
        self._connected = False
        self._auth_token = None
        self._feed_token = None
        self._token_map: dict[str, str] = {}    # tradingsymbol -> token
        self._lot_size_map: dict[str, int] = {}  # tradingsymbol -> lot size
        self._last_login_ts: float = 0.0
        self._client_public_ip: str = ""

    # Angel One force-logs-out all sessions at midnight IST. Because the bot
    # runs 24/7 we must re-authenticate proactively and recover on token errors.
    MAX_SESSION_AGE_SECONDS = 6 * 3600

    def _detect_public_ip(self) -> str:
        """Detect this server's public egress IP for the X-ClientPublicIP header."""
        try:
            import urllib.request
            return urllib.request.urlopen("https://api.ipify.org", timeout=8).read().decode().strip()
        except Exception:
            logger.warning("Could not detect public IP; relying on library default")
            return ""

    def _login(self) -> bool:
        """Authenticate with Angel One and store the session tokens."""
        try:
            from SmartApi import SmartConnect
            import pyotp

            self._smart_api = SmartConnect(api_key=self._api_key)

            # The SmartAPI library hardcodes X-ClientPublicIP; set it explicitly
            # so the value is deterministic and can be whitelisted in Angel One.
            public_ip = self._detect_public_ip()
            if public_ip:
                try:
                    self._smart_api.clientPublicIp = public_ip
                except Exception:
                    pass
                self._client_public_ip = public_ip
                logger.info("Angel One X-ClientPublicIP set to %s", public_ip)

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
                self._last_login_ts = time.time()
                self._load_token_map()
                logger.info("Connected to Angel One SmartAPI (client=%s)", self._client_id)
                return True
            else:
                logger.error("Angel One login failed: %s", session.get("message", "Unknown error"))
                self._connected = False
                return False

        except ImportError:
            logger.error("smartapi-python not installed. Run: pip install smartapi-python pyotp")
            return False
        except Exception:
            logger.exception("Failed to connect to Angel One")
            self._connected = False
            return False

    def connect(self) -> bool:
        """Connect to Angel One SmartAPI and generate session."""
        return self._login()

    def _ensure_session(self) -> bool:
        """Ensure a live session, re-logging in if stale or disconnected."""
        if not self._connected or self._smart_api is None:
            return self._login()
        if (time.time() - self._last_login_ts) > self.MAX_SESSION_AGE_SECONDS:
            logger.info("Angel One session older than %ss, refreshing", self.MAX_SESSION_AGE_SECONDS)
            return self._login()
        return True

    @staticmethod
    def _is_token_error(response: Any) -> bool:
        """Detect an expired/invalid auth token rejection from Angel One."""
        if not isinstance(response, dict):
            return False
        if response.get("errorCode") == "AG8001":
            return True
        return "invalid token" in str(response.get("message", "")).lower()

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
        if not self._ensure_session():
            logger.error("[LIVE] Cannot place order: Angel One session unavailable")
            order.status = OrderStatus.REJECTED
            return order

        try:
            # Convert symbol from Yahoo format (RELIANCE.NS -> RELIANCE-EQ)
            trading_symbol = order.symbol.replace(".NS", "-EQ")

            symbol_token = self._get_symbol_token(trading_symbol)
            if not symbol_token:
                order.status = OrderStatus.REJECTED
                logger.error(
                    "[LIVE] Order rejected: no Angel One token for %s", trading_symbol,
                )
                return order

            quantity = self._round_to_lot(trading_symbol, order.quantity)
            if quantity <= 0:
                order.status = OrderStatus.REJECTED
                logger.error(
                    "[LIVE] Order rejected: qty=%s below one lot for %s",
                    order.quantity, trading_symbol,
                )
                return order

            is_stop = order.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT)
            order_params = {
                "variety": "STOPLOSS" if is_stop else "NORMAL",
                "tradingsymbol": trading_symbol,
                "symboltoken": symbol_token,
                "transactiontype": "BUY" if order.side == OrderSide.BUY else "SELL",
                "exchange": "NSE",
                "ordertype": self._map_order_type(order.order_type),
                "producttype": self._map_product_type(order.service_origin),
                "duration": "DAY",
                "quantity": str(quantity),
            }

            if order.order_type == OrderType.LIMIT:
                order_params["price"] = str(order.price)
            if is_stop:
                order_params["triggerprice"] = str(order.stop_price)

            response = self._smart_api.placeOrder(order_params)

            # Token may have expired mid-session; re-login once and retry.
            if self._is_token_error(response):
                logger.warning("[LIVE] Angel One token invalid, re-logging in and retrying")
                if self._login():
                    response = self._smart_api.placeOrder(order_params)

            order.placed_at = datetime.now()
            if isinstance(response, dict) and response.get("status"):
                order.broker_order_id = str(response.get("data", ""))
                order.status = OrderStatus.SUBMITTED
                logger.info("[LIVE] Order submitted to Angel One: %s", order.broker_order_id)
            elif isinstance(response, str) and response:
                order.broker_order_id = response
                order.status = OrderStatus.SUBMITTED
                logger.info("[LIVE] Order submitted to Angel One: %s", response)
            else:
                message = str(response.get("message", "")) if isinstance(response, dict) else ""
                error_code = str(response.get("errorcode", "")) if isinstance(response, dict) else ""
                order.status = OrderStatus.REJECTED
                # Detect cautionary listing errors (AB4036) for runtime auto-blacklisting
                if error_code == "AB4036" or "cautionary" in message.lower():
                    order.error_code = "AB4036"
                    order.error_message = message
                logger.error("[LIVE] Order rejected by Angel One: %s (code=%s)", message or response, error_code)

        except Exception:
            logger.exception("Failed to submit order to Angel One")
            order.status = OrderStatus.REJECTED

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self._ensure_session():
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
        if not self._ensure_session() or self._smart_api is None:
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

    def get_positions(self) -> Optional[list[dict]]:
        """Get open positions.

        Returns None when the broker query failed (so callers distinguish a
        failed query from a successful-but-empty result). A real (successful)
        query returning no positions yields [].
        """
        if not self._ensure_session() or self._smart_api is None:
            return None
        try:
            positions = self._smart_api.position()
            return positions.get("data") or [] if positions else []
        except Exception:
            logger.exception("Failed to get positions")
            return None

    def get_holdings(self) -> Optional[list[dict]]:
        """Get delivery holdings (demat/book), not the intraday position book.

        Delivery (CNC) positions settle into holdings and appear here, while
        the daily position book (get_positions) only tracks today's tradable
        open positions. Returns None when the query failed and [] on success
        with no holdings, mirroring get_positions() semantics.
        """
        if not self._ensure_session() or self._smart_api is None:
            return None
        try:
            holdings = self._smart_api.getHolding()
            return holdings.get("data") or [] if holdings else []
        except Exception:
            logger.exception("Failed to get holdings")
            return None

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
        """Look up the numeric token for a trading symbol."""
        return self._token_map.get(trading_symbol, "")

    def _round_to_lot(self, trading_symbol: str, quantity: float) -> int:
        """Floor quantity to the exchange lot size. Returns 0 if below one lot."""
        lot = self._lot_size_map.get(trading_symbol, 1) or 1
        qty = int(quantity)
        if lot <= 1:
            return qty
        rounded = (qty // lot) * lot
        return rounded

    def _load_token_map(self) -> None:
        """Build the symbol->token and lot-size maps from ScripMaster.

        Priority: fresh disk cache > download ScripMaster > fallback map.
        """
        if self._token_map:
            return

        cache_path = os.path.join(self._data_dir, "scripmaster_tokens.json")
        if os.path.exists(cache_path):
            try:
                if time.time() - os.path.getmtime(cache_path) < _SCRIPMASTER_TTL_SECONDS:
                    data = json.load(open(cache_path))
                    self._token_map = data.get("tokens", {})
                    self._lot_size_map = data.get("lots", {})
                    logger.info("Loaded ScripMaster from cache (%d symbols)", len(self._token_map))
                    return
            except Exception:
                logger.exception("Failed to load cached ScripMaster tokens")

        tokens, lots = self._fetch_scripmaster()
        if tokens:
            self._token_map = tokens
            self._lot_size_map = lots
            try:
                os.makedirs(self._data_dir, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({"tokens": tokens, "lots": lots}, f)
                logger.info("ScripMaster cached to %s (%d symbols)", cache_path, len(tokens))
            except Exception:
                logger.exception("Failed to cache ScripMaster tokens")
            return

        logger.warning("ScripMaster unavailable — using fallback token map (%d symbols)", len(self._FALLBACK_TOKEN_MAP))
        self._token_map = dict(self._FALLBACK_TOKEN_MAP)
        self._lot_size_map = {s: 1 for s in self._token_map}

    def _fetch_scripmaster(self) -> tuple[dict[str, str], dict[str, int]]:
        """Download the Angel One ScripMaster file and extract NSE symbols."""
        try:
            import urllib.request

            req = urllib.request.Request(
                _SCRIPMASTER_URL, headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)

            tokens: dict[str, str] = {}
            lots: dict[str, int] = {}
            for item in data:
                if item.get("exch_seg") != "NSE" or not item.get("symbol"):
                    continue
                trading_symbol = item.get("symbol") or item.get("tradingsymbol", "")
                if not trading_symbol:
                    continue
                tokens[trading_symbol] = str(item.get("token", ""))
                try:
                    lots[trading_symbol] = int(item.get("lotsize", 1) or 1)
                except (TypeError, ValueError):
                    lots[trading_symbol] = 1
            logger.info("ScripMaster downloaded: %d NSE symbols", len(tokens))
            return tokens, lots
        except Exception:
            logger.exception("Failed to download ScripMaster from %s", _SCRIPMASTER_URL)
            return {}, {}

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "MARKET",
            OrderType.STOP_LOSS_LIMIT: "LIMIT",
        }
        return mapping.get(order_type, "MARKET")

    @staticmethod
    def _map_product_type(service_origin: str) -> str:
        """Map a service's order to the broker product type.

        Positional (delivery / carry-forward) orders MUST use DELIVERY (CNC) so
        the broker does NOT auto square-off the position at market close.
        Intraday orders use INTRADAY (MIS).
        """
        if service_origin == "position_trader":
            return "DELIVERY"
        return "INTRADAY"

    _FALLBACK_TOKEN_MAP: dict[str, str] = {
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
