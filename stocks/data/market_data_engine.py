"""Market Data Engine — real-time data ingestion, candle building, and indicator computation."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from stocks.core.enums import Timeframe
from stocks.core.event_bus import EventBus
from stocks.core.events import MarketDataEvent
from stocks.core.models import Candle
from stocks.data.data_provider import DataProvider
from stocks.data.providers.yahoo_provider import YahooProvider

logger = logging.getLogger(__name__)


class MarketDataEngine:
    """Manages market data: fetching, candle building, and indicator computation.

    Publishes MarketDataEvent on the event bus whenever new data arrives.
    """

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus

        # Initialize data provider
        provider_name = config.get("market_data", {}).get("provider", "yahoo")
        self.provider: DataProvider = self._create_provider(provider_name)

        # Data storage: symbol -> timeframe -> DataFrame of OHLCV
        self._data: dict[str, dict[Timeframe, pd.DataFrame]] = defaultdict(dict)

        # Indicator cache: symbol -> timeframe -> dict of indicator series
        self._indicators: dict[str, dict[Timeframe, dict[str, pd.Series]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        self._timeframes = self._parse_timeframes(
            config.get("market_data", {}).get("timeframes", ["5m"])
        )
        self._history_days = config.get("market_data", {}).get("history_days", 30)

        logger.info(
            "MarketDataEngine initialized (provider=%s, timeframes=%s)",
            provider_name, [tf.value for tf in self._timeframes],
        )

    @staticmethod
    def _create_provider(name: str) -> DataProvider:
        if name == "yahoo":
            return YahooProvider()
        raise ValueError(f"Unknown data provider: {name}")

    @staticmethod
    def _parse_timeframes(tf_strings: list[str]) -> list[Timeframe]:
        tf_map = {tf.value: tf for tf in Timeframe}
        result = []
        for s in tf_strings:
            if s in tf_map:
                result.append(tf_map[s])
            else:
                logger.warning("Unknown timeframe '%s', skipping", s)
        return result or [Timeframe.M5]

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def load_daily_data(self, symbols: list[str]) -> None:
        """Fetch daily (D1) data for stock selection scoring.

        This is separate from load_historical_data because the configured
        intraday timeframes (1m, 5m, 15m) don't include D1, but the stock
        selector needs daily data for volume/volatility/trend scoring.
        """
        start = datetime.now() - timedelta(days=self._history_days)
        loaded = 0
        for symbol in symbols:
            df = self.provider.get_historical_data(symbol, Timeframe.D1, start)
            if not df.empty:
                self._data[symbol][Timeframe.D1] = df
                self._compute_indicators(symbol, Timeframe.D1)
                loaded += 1
        logger.info("Loaded daily data for %d/%d symbols", loaded, len(symbols))

    def load_historical_data(self, symbols: list[str]) -> None:
        """Fetch historical data for all symbols and configured timeframes."""
        start = datetime.now() - timedelta(days=self._history_days)
        for symbol in symbols:
            for tf in self._timeframes:
                df = self.provider.get_historical_data(symbol, tf, start)
                if not df.empty:
                    self._data[symbol][tf] = df
                    self._compute_indicators(symbol, tf)
                    logger.debug(
                        "Loaded %d candles for %s (%s)", len(df), symbol, tf.value
                    )

    def update_data(self, symbols: list[str]) -> None:
        """Fetch latest data for all symbols and publish events."""
        for symbol in symbols:
            for tf in self._timeframes:
                try:
                    df = self.provider.get_historical_data(
                        symbol, tf,
                        start=datetime.now() - timedelta(days=2),
                    )
                    if df.empty:
                        continue

                    # Merge with existing data
                    existing = self._data.get(symbol, {}).get(tf)
                    if existing is not None and not existing.empty:
                        df = pd.concat([existing, df])
                        df = df[~df.index.duplicated(keep="last")]
                        df.sort_index(inplace=True)

                    self._data[symbol][tf] = df
                    self._compute_indicators(symbol, tf)

                    # Publish latest candle as event
                    latest = df.iloc[-1]
                    candle = Candle(
                        symbol=symbol,
                        timeframe=tf,
                        timestamp=df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else datetime.now(),
                        open=float(latest["open"]),
                        high=float(latest["high"]),
                        low=float(latest["low"]),
                        close=float(latest["close"]),
                        volume=float(latest.get("volume", 0)),
                    )
                    self.event_bus.publish(MarketDataEvent(
                        symbol=symbol, candle=candle,
                        price=candle.close, volume=candle.volume,
                        timeframe=tf,
                    ))
                except Exception:
                    logger.exception("Failed to update data for %s (%s)", symbol, tf.value)

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(self, symbol: str, tf: Timeframe) -> None:
        """Compute standard technical indicators for the given data."""
        df = self._data.get(symbol, {}).get(tf)
        if df is None or len(df) < 20:
            return

        indicators = {}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Simple Moving Averages
        indicators["sma_20"] = close.rolling(window=20).mean()
        indicators["sma_50"] = close.rolling(window=50).mean()

        # Exponential Moving Averages
        indicators["ema_9"] = close.ewm(span=9, adjust=False).mean()
        indicators["ema_21"] = close.ewm(span=21, adjust=False).mean()

        # Bollinger Bands
        sma20 = indicators["sma_20"]
        std20 = close.rolling(window=20).std()
        indicators["bb_upper"] = sma20 + 2 * std20
        indicators["bb_lower"] = sma20 - 2 * std20
        indicators["bb_mid"] = sma20

        # RSI (14)
        indicators["rsi_14"] = self._compute_rsi(close, 14)

        # ATR (14)
        indicators["atr_14"] = self._compute_atr(high, low, close, 14)

        # VWAP (intraday)
        if "volume" in df.columns:
            typical_price = (high + low + close) / 3
            cum_vol = df["volume"].cumsum()
            cum_tp_vol = (typical_price * df["volume"]).cumsum()
            indicators["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

        # ADX (for trend strength)
        indicators["adx_14"] = self._compute_adx(high, low, close, 14)

        self._indicators[symbol][tf] = indicators

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.rolling(window=period).mean()
        return adx

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_dataframe(self, symbol: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        """Get the OHLCV DataFrame for a symbol/timeframe."""
        return self._data.get(symbol, {}).get(timeframe)

    def get_indicator(self, symbol: str, timeframe: Timeframe, indicator: str) -> Optional[pd.Series]:
        """Get a computed indicator series."""
        return self._indicators.get(symbol, {}).get(timeframe, {}).get(indicator)

    def get_latest_candle(self, symbol: str, timeframe: Timeframe) -> Optional[Candle]:
        """Get the most recent candle for a symbol/timeframe."""
        df = self.get_dataframe(symbol, timeframe)
        if df is None or df.empty:
            return None
        row = df.iloc[-1]
        return Candle(
            symbol=symbol, timeframe=timeframe,
            timestamp=df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else datetime.now(),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row.get("volume", 0)),
        )

    def get_current_price(self, symbol: str) -> float:
        """Get the latest price (from data or live quote)."""
        # Try from cached data first
        for tf in self._timeframes:
            df = self._data.get(symbol, {}).get(tf)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        # Fall back to provider
        return self.provider.get_current_price(symbol)
