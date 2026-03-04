"""Market Data Engine for the crypto trading system.

Manages OHLCV data fetching, caching, and technical indicator computation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from crypto.core.enums import Timeframe
from crypto.core.event_bus import EventBus
from crypto.data.providers.ccxt_provider import CcxtProvider

logger = logging.getLogger(__name__)


class MarketDataEngine:
    """Fetches, caches, and enriches market data with technical indicators."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus, provider: CcxtProvider) -> None:
        self.config = config
        self.event_bus = event_bus
        self.provider = provider

        md_config = config.get("market_data", {})
        self._timeframes = md_config.get("timeframes", ["5m", "15m", "1h"])
        self._history_bars = md_config.get("history_bars", 200)

        # Cache: {symbol: {timeframe_str: DataFrame}}
        self._data: dict[str, dict[str, pd.DataFrame]] = {}
        self._price_cache: dict[str, float] = {}

        logger.info("MarketDataEngine initialized (timeframes=%s)", self._timeframes)

    def load_historical_data(self, symbols: list[str]) -> None:
        """Load historical OHLCV data for all symbols and timeframes."""
        for symbol in symbols:
            self._data.setdefault(symbol, {})
            for tf in self._timeframes:
                df = self.provider.fetch_ohlcv(symbol, tf, limit=self._history_bars)
                if not df.empty:
                    df = self._compute_indicators(df)
                    self._data[symbol][tf] = df
                    if not df.empty:
                        self._price_cache[symbol] = float(df["close"].iloc[-1])

        loaded = sum(1 for s in self._data if self._data[s])
        logger.info("Historical data loaded for %d/%d symbols", loaded, len(symbols))

    def update_data(self, symbols: list[str]) -> None:
        """Fetch latest candle data and append to cache."""
        for symbol in symbols:
            self._data.setdefault(symbol, {})
            for tf in self._timeframes:
                df = self.provider.fetch_ohlcv(symbol, tf, limit=5)
                if df.empty:
                    continue

                existing = self._data.get(symbol, {}).get(tf)
                if existing is not None and not existing.empty:
                    combined = pd.concat([existing, df])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.iloc[-self._history_bars:]
                    df = combined

                df = self._compute_indicators(df)
                self._data[symbol][tf] = df

                if not df.empty:
                    self._price_cache[symbol] = float(df["close"].iloc[-1])

    def get_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get the cached DataFrame for a symbol and timeframe."""
        return self._data.get(symbol, {}).get(timeframe, pd.DataFrame())

    def get_current_price(self, symbol: str) -> float:
        """Get the latest cached price for a symbol."""
        return self._price_cache.get(symbol, 0.0)

    def get_indicator(self, symbol: str, timeframe: str, indicator: str) -> float:
        """Get the latest value of a specific indicator."""
        df = self.get_dataframe(symbol, timeframe)
        if df.empty or indicator not in df.columns:
            return 0.0
        val = df[indicator].iloc[-1]
        return 0.0 if pd.isna(val) else float(val)

    @staticmethod
    def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators on an OHLCV DataFrame."""
        if df.empty or len(df) < 2:
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # EMA
        df["ema_9"] = close.ewm(span=9, adjust=False).mean()
        df["ema_21"] = close.ewm(span=21, adjust=False).mean()
        df["sma_20"] = close.rolling(20).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # Bollinger Bands
        df["bb_mid"] = df["sma_20"]
        bb_std = close.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std

        # ADX
        df["adx"] = _compute_adx(high, low, close, period=14)

        # Volume SMA
        df["volume_sma"] = volume.rolling(20).mean()

        return df

    @property
    def symbols_loaded(self) -> list[str]:
        return [s for s, data in self._data.items() if data]


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index)."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Only keep the larger of the two
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return adx
