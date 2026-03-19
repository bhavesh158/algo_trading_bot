"""Opening Range Breakout (ORB) Strategy (Enhanced).

Monitors the first N minutes of trading to establish an opening range,
then trades breakouts above/below that range.

Relaxed filters for more signal generation:
- Volume confirmation disabled by default (was blocking valid breakouts)
- Lower minimum range size requirement
- Smaller breakout buffer for earlier entries
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Any, Optional

import numpy as np

from stocks.core.enums import OrderSide, SignalStrength, Timeframe
from stocks.core.models import Signal
from stocks.data.market_data_engine import MarketDataEngine
from stocks.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Trade breakouts/breakdowns from the opening range (first N minutes)."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("opening_range_breakout", config, market_data)

        strat_config = config.get("opening_range_breakout", {})
        self._or_minutes = strat_config.get("opening_range_minutes", 15)
        self._buffer_pct = strat_config.get("breakout_buffer_pct", 0.15)  # Lower from 0.2
        self._atr_stop = strat_config.get("atr_multiplier_stop", 1.5)
        self._atr_target = strat_config.get("atr_multiplier_target", 3.0)
        self._max_window_minutes = strat_config.get("max_window_minutes", 90)  # Extended from 60
        self._volume_confirm = strat_config.get("volume_confirm", False)  # Disabled by default
        self._min_range_atr_ratio = strat_config.get("min_range_atr_ratio", 0.3)  # Lower from 0.5
        self.primary_timeframe = Timeframe.M1

        # Enable trailing stop and time exit from base class
        self._trailing_stop_atr = strat_config.get("trailing_stop_atr", 1.2)
        self._max_hold_minutes = strat_config.get("max_hold_minutes", 150)

        # Market open time (IST)
        sched = config.get("schedule", {})
        self._market_open_str = sched.get("market_open", "09:15")

        # State per symbol per day
        self._opening_ranges: dict[str, dict] = {}
        self._signals_fired: dict[str, bool] = {}  # One signal per symbol per day
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
            self._signals_fired.clear()
            self._last_date = today_str

        # Only one ORB signal per symbol per day
        if self._signals_fired.get(symbol, False):
            return None

        # Time window: only fire within max_window_minutes of market open
        now = datetime.now()
        market_open_time = datetime.strptime(self._market_open_str, "%H:%M").time()
        market_open_dt = datetime.combine(now.date(), market_open_time)
        cutoff = market_open_dt + timedelta(minutes=self._max_window_minutes)
        if now > cutoff:
            return None

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

        # Volume confirmation (DISABLED by default - was blocking too many signals)
        if self._volume_confirm and "volume" in df.columns:
            current_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].tail(10).mean())
            if avg_vol > 0 and current_vol < avg_vol:
                logger.debug(
                    "[%s] Volume filter blocked %s: vol=%.0f < %.0f",
                    self.strategy_id, symbol, current_vol, avg_vol,
                )
                return None

        # ATR for stop/target
        atr = self.market_data.get_indicator(symbol, Timeframe.M5, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return None
        atr_value = float(atr.iloc[-1])

        buffer = or_high * self._buffer_pct / 100

        # Range size filter (RELAXED - lower threshold)
        range_size = or_high - or_low
        if atr_value > 0 and range_size < atr_value * self._min_range_atr_ratio:
            logger.debug(
                "[%s] ORB range too narrow for %s: range=%.2f < %.2f (%.1f×ATR)",
                self.strategy_id, symbol, range_size,
                atr_value * self._min_range_atr_ratio, self._min_range_atr_ratio,
            )
            return None

        # Dynamic confidence based on range size, volume, and breakout strength
        base_confidence = 0.50  # Lower base to allow more signals

        # Bullish breakout
        if current_price > or_high + buffer:
            breakout_pct = (current_price - or_high) / or_high * 100
            confidence = self._compute_dynamic_confidence(
                base_confidence, range_size, atr_value, df, breakout_pct,
            )
            strength = SignalStrength.STRONG if confidence >= 0.75 else SignalStrength.MODERATE

            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=current_price - self._atr_stop * atr_value,
                target_price=current_price + self._atr_target * atr_value,
                metadata={
                    "or_high": or_high, "or_low": or_low, "direction": "bullish",
                    "max_hold_minutes": self._max_hold_minutes,
                },
            )
            logger.info(
                "[%s] ORB bullish breakout: %s price=%.2f > OR_high=%.2f conf=%.2f",
                self.strategy_id, symbol, current_price, or_high, confidence,
            )
            self._signals_fired[symbol] = True
            return signal

        # Bearish breakout (SELL signal) - breakdown
        if current_price < or_low - buffer:
            breakout_pct = (or_low - current_price) / or_low * 100
            confidence = self._compute_dynamic_confidence(
                base_confidence, range_size, atr_value, df, breakout_pct,
            )
            strength = SignalStrength.STRONG if confidence >= 0.75 else SignalStrength.MODERATE

            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=current_price + self._atr_stop * atr_value,
                target_price=current_price - self._atr_target * atr_value,
                metadata={
                    "or_high": or_high, "or_low": or_low, "direction": "bearish",
                    "max_hold_minutes": self._max_hold_minutes,
                },
            )
            logger.info(
                "[%s] ORB bearish breakdown: %s price=%.2f < OR_low=%.2f conf=%.2f",
                self.strategy_id, symbol, current_price, or_low, confidence,
            )
            self._signals_fired[symbol] = True
            return signal

        return None

    @staticmethod
    def _compute_dynamic_confidence(
        base: float, or_range: float, atr_value: float,
        df: Any, breakout_pct: float,
    ) -> float:
        """Compute confidence from range size, volume, and breakout strength."""
        confidence = base

        # Range quality: larger range relative to ATR = more meaningful breakout
        if atr_value > 0:
            range_ratio = or_range / atr_value
            if range_ratio >= 1.0:
                confidence += 0.1
            elif range_ratio >= 0.7:
                confidence += 0.05

        # Volume surge at breakout candle
        if "volume" in df.columns and len(df) >= 10:
            current_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].tail(10).mean())
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                if vol_ratio > 2.0:
                    confidence += 0.15
                elif vol_ratio > 1.5:
                    confidence += 0.1

        # Breakout strength
        if breakout_pct > 0.5:
            confidence += 0.05

        return min(confidence, 1.0)

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        atr = self.market_data.get_indicator(symbol, Timeframe.M5, "atr_14")
        if atr is None or atr.empty:
            return False

        atr_value = float(atr.iloc[-1])

        # Symmetric exit for long positions
        stop = entry_price - self._atr_stop * atr_value
        target = entry_price + self._atr_target * atr_value
        return current_price <= stop or current_price >= target
