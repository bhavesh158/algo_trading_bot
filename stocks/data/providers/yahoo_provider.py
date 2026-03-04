"""Yahoo Finance data provider implementation."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from stocks.core.enums import Timeframe
from stocks.data.data_provider import DataProvider

logger = logging.getLogger(__name__)

# Mapping from our Timeframe enum to yfinance interval strings
_TIMEFRAME_MAP = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.M30: "30m",
    Timeframe.H1: "1h",
    Timeframe.D1: "1d",
}

# yfinance max history per interval
_MAX_HISTORY_DAYS = {
    Timeframe.M1: 7,
    Timeframe.M5: 60,
    Timeframe.M15: 60,
    Timeframe.M30: 60,
    Timeframe.H1: 730,
    Timeframe.D1: 10000,
}


class YahooProvider(DataProvider):
    """Market data provider using Yahoo Finance (yfinance)."""

    def __init__(self) -> None:
        self._cache: dict[str, yf.Ticker] = {}

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a cached Ticker object."""
        if symbol not in self._cache:
            self._cache[symbol] = yf.Ticker(symbol)
        return self._cache[symbol]

    def get_historical_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Yahoo Finance."""
        interval = _TIMEFRAME_MAP.get(timeframe, "1d")
        max_days = _MAX_HISTORY_DAYS.get(timeframe, 30)

        # Clamp start date to max allowed history
        earliest = datetime.now() - timedelta(days=max_days)
        if start < earliest:
            start = earliest

        end = end or datetime.now()

        try:
            ticker = self._get_ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
            )
            if df.empty:
                logger.warning("No data returned for %s (%s)", symbol, interval)
                return pd.DataFrame()

            # Normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            # Keep only OHLCV columns
            required = ["open", "high", "low", "close", "volume"]
            available = [c for c in required if c in df.columns]
            df = df[available]
            return df

        except Exception:
            logger.exception("Failed to fetch historical data for %s", symbol)
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.fast_info
            price = getattr(info, "last_price", None)
            if price is None:
                # Fallback: use last close from history
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            return float(price) if price else 0.0
        except Exception:
            logger.exception("Failed to get current price for %s", symbol)
            return 0.0

    def get_quote(self, symbol: str) -> dict:
        """Get a full quote for a symbol."""
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "price": getattr(info, "last_price", 0.0),
                "volume": getattr(info, "last_volume", 0),
                "market_cap": getattr(info, "market_cap", 0),
                "day_high": getattr(info, "day_high", 0.0),
                "day_low": getattr(info, "day_low", 0.0),
                "previous_close": getattr(info, "previous_close", 0.0),
            }
        except Exception:
            logger.exception("Failed to get quote for %s", symbol)
            return {"symbol": symbol}

    def get_stock_info(self, symbol: str) -> dict:
        """Get stock metadata."""
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info or {}
            return {
                "symbol": symbol,
                "name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "avg_volume": info.get("averageVolume", 0),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0.0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0.0),
            }
        except Exception:
            logger.exception("Failed to get stock info for %s", symbol)
            return {"symbol": symbol}

    def search_symbols(self, query: str) -> list[dict]:
        """Search for symbols (limited support in yfinance)."""
        # yfinance doesn't have a robust symbol search — return empty
        logger.debug("Symbol search not fully supported via Yahoo provider")
        return []
