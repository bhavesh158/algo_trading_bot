"""Opening Range Breakout (ORB) Strategy.

Monitors the first N minutes of trading to establish an opening range,
then trades breakouts above/below that range.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Any, Optional

import numpy as np

from stocks.core.enums import OrderSide, SignalStrength, Timeframe
from stocks.core.models import Signal
from stocks.data.market_data_engine import MarketDataEngine
from stocks.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Trade breakouts from the opening range (first N minutes)."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("opening_range_breakout", config, market_data)

        strat_config = config.get("opening_range_breakout", {})
        self._or_minutes = strat_config.get("opening_range_minutes", 15)
        self._buffer_pct = strat_config.get("breakout_buffer_pct", 0.1)
        self._atr_stop = strat_config.get("atr_multiplier_stop", 1.5)
        self._atr_target = strat_config.get("atr_multiplier_target", 2.5)
        self.primary_timeframe = Timeframe.M1

        # State per symbol per day
        self._opening_ranges: dict[str, dict] = {}
        self._last_date: Optional[str] = None

    def _compute_opening_range(self, symbol: str) -> Optional[dict]:
        """Compute the opening range (high/low of first N minutes)."""
        df = self.market_data.get_dataframe(symbol, Timeframe.M1)
        if df is None or df.empty:
            return None

        today = datetime.now().date()

        # Filter to today's candles
        today_data = df[df.index.date == today] if hasattr(df.index, 'date') else df
        if today_data.empty or len(today_data) < self._or_minutes:
            return None

        # Take first N minutes
        or_data = today_data.iloc[:self._or_minutes]
        return {
            "high": float(or_data["high"].max()),
            "low": float(or_data["low"].min()),
            "computed_at": datetime.now(),
        }

    def analyze(self, symbol: str) -> Optional[Signal]:
        # Reset opening ranges on new day
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._last_date != today_str:
            self._opening_ranges.clear()
            self._last_date = today_str

        # Compute opening range if not yet available
        if symbol not in self._opening_ranges:
            or_range = self._compute_opening_range(symbol)
            if or_range is None:
                return None
            self._opening_ranges[symbol] = or_range

        or_range = self._opening_ranges[symbol]
        or_high = or_range["high"]
        or_low = or_range["low"]

        # Get current price
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is None or df.empty:
            return None
        current_price = float(df["close"].iloc[-1])

        # ATR for stop/target
        atr = self.market_data.get_indicator(symbol, Timeframe.M5, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return None
        atr_value = float(atr.iloc[-1])

        buffer = or_high * self._buffer_pct / 100

        # Bullish breakout
        if current_price > or_high + buffer:
            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=0.65,
                entry_price=current_price,
                stop_loss=current_price - self._atr_stop * atr_value,
                target_price=current_price + self._atr_target * atr_value,
                metadata={"or_high": or_high, "or_low": or_low, "direction": "bullish"},
            )
            logger.info(
                "[%s] ORB bullish breakout: %s price=%.2f > OR_high=%.2f",
                self.strategy_id, symbol, current_price, or_high,
            )
            return signal

        # Bearish breakout (sell signal)
        if current_price < or_low - buffer:
            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=0.65,
                entry_price=current_price,
                stop_loss=current_price + self._atr_stop * atr_value,
                target_price=current_price - self._atr_target * atr_value,
                metadata={"or_high": or_high, "or_low": or_low, "direction": "bearish"},
            )
            logger.info(
                "[%s] ORB bearish breakout: %s price=%.2f < OR_low=%.2f",
                self.strategy_id, symbol, current_price, or_low,
            )
            return signal

        return None

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        atr = self.market_data.get_indicator(symbol, Timeframe.M5, "atr_14")
        if atr is None or atr.empty:
            return False

        atr_value = float(atr.iloc[-1])

        # Symmetric exit for long positions
        stop = entry_price - self._atr_stop * atr_value
        target = entry_price + self._atr_target * atr_value
        return current_price <= stop or current_price >= target
