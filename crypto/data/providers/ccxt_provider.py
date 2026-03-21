"""Exchange data provider using the ccxt library.

Provides unified access to OHLCV data, tickers, and orderbook depth
across any ccxt-supported exchange.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CcxtProvider:
    """Fetches market data from a crypto exchange via ccxt."""

    def __init__(self, exchange_name: str, config: dict[str, Any]) -> None:
        self._exchange_name = exchange_name
        self._config = config
        self._exchange = None

    def connect(self, api_key: str = "", api_secret: str = "", password: str = "") -> bool:
        """Initialize the ccxt exchange instance."""
        try:
            import ccxt

            exchange_class = getattr(ccxt, self._exchange_name, None)
            if exchange_class is None:
                logger.error("Unsupported exchange: %s", self._exchange_name)
                return False

            params: dict[str, Any] = {
                "enableRateLimit": self._config.get("exchange", {}).get("rate_limit", True),
            }
            if api_key:
                params["apiKey"] = api_key
                params["secret"] = api_secret
            if password:
                params["password"] = password

            # Use sandbox/testnet if configured
            self._exchange = exchange_class(params)
            if self._config.get("exchange", {}).get("sandbox", False):
                self._exchange.set_sandbox_mode(True)

            self._exchange.load_markets()
            logger.info(
                "Connected to %s (%d markets loaded)",
                self._exchange_name, len(self._exchange.markets),
            )
            return True

        except ImportError:
            logger.error("ccxt not installed. Run: pip install ccxt")
            return False
        except Exception:
            logger.exception("Failed to connect to %s", self._exchange_name)
            return False

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 200,
        since_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data for a symbol.

        Args:
            symbol: Trading pair (e.g., BTC/USDT).
            timeframe: Candle timeframe (5m, 15m, 1h, etc.).
            limit: Max candles to fetch (exchange-dependent, typically 500-1000).
            since_ms: Optional timestamp in ms to fetch candles from.

        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        if self._exchange is None:
            return pd.DataFrame()

        try:
            kwargs = {"limit": limit}
            if since_ms:
                kwargs["since"] = since_ms
                
            raw = self._exchange.fetch_ohlcv(symbol, timeframe, **kwargs)
            if not raw:
                return pd.DataFrame()

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception:
            logger.exception("Failed to fetch OHLCV for %s %s", symbol, timeframe)
            return pd.DataFrame()

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker data (price, volume, bid/ask).

        Returns dict with keys: last, bid, ask, baseVolume, quoteVolume, percentage.
        """
        if self._exchange is None:
            return {}
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return {
                "last": ticker.get("last", 0),
                "bid": ticker.get("bid", 0),
                "ask": ticker.get("ask", 0),
                "base_volume": ticker.get("baseVolume", 0),
                "quote_volume": ticker.get("quoteVolume", 0),
                "change_pct": ticker.get("percentage", 0),
            }
        except Exception:
            logger.exception("Failed to fetch ticker for %s", symbol)
            return {}

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Fetch orderbook depth.

        Returns dict with keys: bids_depth, asks_depth (in quote currency).
        """
        if self._exchange is None:
            return {}
        try:
            ob = self._exchange.fetch_order_book(symbol, limit=limit)
            bids_depth = sum(price * amount for price, amount in ob.get("bids", []))
            asks_depth = sum(price * amount for price, amount in ob.get("asks", []))
            spread = 0.0
            if ob.get("bids") and ob.get("asks"):
                best_bid = ob["bids"][0][0]
                best_ask = ob["asks"][0][0]
                mid = (best_bid + best_ask) / 2
                spread = (best_ask - best_bid) / mid * 100 if mid > 0 else 0

            return {
                "bids_depth": bids_depth,
                "asks_depth": asks_depth,
                "spread_pct": spread,
                "best_bid": ob["bids"][0][0] if ob.get("bids") else 0,
                "best_ask": ob["asks"][0][0] if ob.get("asks") else 0,
            }
        except Exception:
            logger.exception("Failed to fetch orderbook for %s", symbol)
            return {}

    def get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        ticker = self.fetch_ticker(symbol)
        return ticker.get("last", 0.0)

    @property
    def exchange(self):
        return self._exchange

    @property
    def is_connected(self) -> bool:
        return self._exchange is not None
