"""Abstract interface for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from core.enums import Timeframe


class DataProvider(ABC):
    """Base class for market data providers.

    Concrete implementations (e.g., Yahoo Finance, broker APIs) must implement
    all abstract methods.
    """

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Returns DataFrame with columns: open, high, low, close, volume
        indexed by datetime.
        """
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> dict:
        """Get a full quote (price, volume, bid, ask, etc.) for a symbol."""
        ...

    @abstractmethod
    def get_stock_info(self, symbol: str) -> dict:
        """Get stock metadata (name, sector, market cap, avg volume, etc.)."""
        ...

    @abstractmethod
    def search_symbols(self, query: str) -> list[dict]:
        """Search for symbols matching a query string."""
        ...
