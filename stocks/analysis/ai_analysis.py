"""AI-Assisted Analysis Layer.

Enhances trading signals using statistical analysis of price patterns,
volatility conditions, momentum, and historical strategy performance.

Per PRD §8: AI should assist decision making, not be the sole decision maker.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from stocks.core.enums import SignalStrength, Timeframe
from stocks.core.event_bus import EventBus
from stocks.core.models import Signal
from stocks.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class AIAnalysis:
    """Statistical signal filter and confidence adjuster."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._market_data: Optional[MarketDataEngine] = None

        ai_config = config.get("ai_analysis", {})
        self._enabled = ai_config.get("enabled", True)
        self._min_boost = ai_config.get("min_confidence_boost", 0.1)
        self._max_boost = ai_config.get("max_confidence_boost", 0.3)
        self._lookback = ai_config.get("lookback_candles", 50)

        logger.info("AIAnalysis initialized (enabled=%s)", self._enabled)

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        self._market_data = market_data

    def evaluate_signal(self, signal: Signal) -> Signal:
        """Evaluate and adjust a signal's confidence based on multiple factors.

        Returns the signal with potentially modified confidence and strength.
        """
        if not self._enabled or self._market_data is None:
            return signal

        adjustments: list[float] = []

        # Factor 1: Trend alignment
        trend_adj = self._check_trend_alignment(signal)
        adjustments.append(trend_adj)

        # Factor 2: Volatility regime
        vol_adj = self._check_volatility_regime(signal)
        adjustments.append(vol_adj)

        # Factor 3: RSI confirmation
        rsi_adj = self._check_rsi_confirmation(signal)
        adjustments.append(rsi_adj)

        # Factor 4: Volume confirmation
        vol_confirm = self._check_volume_confirmation(signal)
        adjustments.append(vol_confirm)

        # Factor 5: VWAP position
        vwap_adj = self._check_vwap_position(signal)
        adjustments.append(vwap_adj)

        # Factor 6: Support/resistance proximity
        sr_adj = self._check_support_resistance(signal)
        adjustments.append(sr_adj)

        # Factor 7: Candle pattern recognition
        pattern_adj = self._check_candle_patterns(signal)
        adjustments.append(pattern_adj)

        # Calculate total adjustment (average of factors, clamped)
        if adjustments:
            avg_adj = sum(adjustments) / len(adjustments)
            adjustment = max(-self._max_boost, min(self._max_boost, avg_adj))
        else:
            adjustment = 0.0

        # Apply adjustment
        new_confidence = max(0.0, min(1.0, signal.confidence + adjustment))

        if new_confidence != signal.confidence:
            logger.debug(
                "AI adjusted confidence for %s %s: %.2f -> %.2f (adj=%.3f)",
                signal.strategy_id, signal.symbol, signal.confidence,
                new_confidence, adjustment,
            )

        signal.confidence = new_confidence

        # Update strength based on new confidence
        if new_confidence >= 0.8:
            signal.strength = SignalStrength.STRONG
        elif new_confidence >= 0.5:
            signal.strength = SignalStrength.MODERATE
        else:
            signal.strength = SignalStrength.WEAK

        return signal

    def _check_trend_alignment(self, signal: Signal) -> float:
        """Boost if signal aligns with higher-timeframe trend."""
        ema_9 = self._market_data.get_indicator(signal.symbol, Timeframe.M15, "ema_9")
        ema_21 = self._market_data.get_indicator(signal.symbol, Timeframe.M15, "ema_21")

        if ema_9 is None or ema_21 is None or ema_9.empty or ema_21.empty:
            return 0.0

        ema_9_val = float(ema_9.iloc[-1])
        ema_21_val = float(ema_21.iloc[-1])

        from stocks.core.enums import OrderSide
        if signal.side == OrderSide.BUY and ema_9_val > ema_21_val:
            return self._min_boost  # Trend supports buy
        elif signal.side == OrderSide.SELL and ema_9_val < ema_21_val:
            return self._min_boost  # Trend supports sell
        elif signal.side == OrderSide.BUY and ema_9_val < ema_21_val:
            return -self._min_boost  # Counter-trend buy
        elif signal.side == OrderSide.SELL and ema_9_val > ema_21_val:
            return -self._min_boost  # Counter-trend sell

        return 0.0

    def _check_volatility_regime(self, signal: Signal) -> float:
        """Penalize signals during extreme volatility."""
        atr = self._market_data.get_indicator(signal.symbol, Timeframe.M5, "atr_14")
        if atr is None or len(atr) < self._lookback:
            return 0.0

        current_atr = float(atr.iloc[-1])
        mean_atr = float(atr.iloc[-self._lookback:].mean())

        if mean_atr == 0:
            return 0.0

        ratio = current_atr / mean_atr

        # High volatility (>2x normal) — penalize
        if ratio > 2.0:
            return -self._max_boost
        # Very low volatility (<0.5x) — slight penalty (choppy)
        elif ratio < 0.5:
            return -self._min_boost
        # Moderate volatility — neutral to positive
        elif 0.8 <= ratio <= 1.5:
            return self._min_boost * 0.5

        return 0.0

    def _check_rsi_confirmation(self, signal: Signal) -> float:
        """Check if RSI supports the signal direction."""
        rsi = self._market_data.get_indicator(signal.symbol, Timeframe.M5, "rsi_14")
        if rsi is None or rsi.empty or np.isnan(rsi.iloc[-1]):
            return 0.0

        rsi_val = float(rsi.iloc[-1])

        from stocks.core.enums import OrderSide
        if signal.side == OrderSide.BUY:
            if rsi_val < 30:
                return self._min_boost  # Oversold — supports buy
            elif rsi_val > 70:
                return -self._min_boost  # Overbought — penalize buy
        elif signal.side == OrderSide.SELL:
            if rsi_val > 70:
                return self._min_boost  # Overbought — supports sell
            elif rsi_val < 30:
                return -self._min_boost  # Oversold — penalize sell

        return 0.0

    def _check_volume_confirmation(self, signal: Signal) -> float:
        """Boost if current volume is above average."""
        df = self._market_data.get_dataframe(signal.symbol, Timeframe.M5)
        if df is None or "volume" not in df.columns or len(df) < 20:
            return 0.0

        current_vol = float(df["volume"].iloc[-1])
        avg_vol = float(df["volume"].iloc[-20:].mean())

        if avg_vol == 0:
            return 0.0

        ratio = current_vol / avg_vol

        if ratio > 1.5:
            return self._min_boost  # Strong volume
        elif ratio < 0.5:
            return -self._min_boost  # Weak volume

        return 0.0

    def _check_vwap_position(self, signal: Signal) -> float:
        """Boost signals aligned with VWAP (buy below, sell above)."""
        vwap = self._market_data.get_indicator(signal.symbol, Timeframe.M5, "vwap")
        if vwap is None or vwap.empty or np.isnan(vwap.iloc[-1]):
            return 0.0

        vwap_val = float(vwap.iloc[-1])
        price = signal.entry_price

        if price <= 0 or vwap_val <= 0:
            return 0.0

        from stocks.core.enums import OrderSide
        if signal.side == OrderSide.BUY and price < vwap_val:
            return self._min_boost  # Buying below VWAP — good
        elif signal.side == OrderSide.BUY and price > vwap_val * 1.01:
            return -self._min_boost * 0.5  # Buying well above VWAP — slight penalty
        elif signal.side == OrderSide.SELL and price > vwap_val:
            return self._min_boost  # Selling above VWAP — good
        elif signal.side == OrderSide.SELL and price < vwap_val * 0.99:
            return -self._min_boost * 0.5  # Selling well below VWAP — slight penalty

        return 0.0

    def _check_support_resistance(self, signal: Signal) -> float:
        """Boost if entry price is near a support (for BUY) or resistance (for SELL)."""
        df = self._market_data.get_dataframe(signal.symbol, Timeframe.M5)
        if df is None or len(df) < self._lookback:
            return 0.0

        price = signal.entry_price
        recent = df.iloc[-self._lookback:]

        # Find recent support (lowest lows) and resistance (highest highs)
        support = float(recent["low"].rolling(5).min().iloc[-1])
        resistance = float(recent["high"].rolling(5).max().iloc[-1])

        if np.isnan(support) or np.isnan(resistance):
            return 0.0

        price_range = resistance - support
        if price_range <= 0:
            return 0.0

        from stocks.core.enums import OrderSide
        if signal.side == OrderSide.BUY:
            # Boost if near support (bottom 20% of range)
            if price <= support + price_range * 0.2:
                return self._min_boost
            # Penalize if near resistance
            if price >= resistance - price_range * 0.1:
                return -self._min_boost * 0.5
        elif signal.side == OrderSide.SELL:
            # Boost if near resistance
            if price >= resistance - price_range * 0.2:
                return self._min_boost
            # Penalize if near support
            if price <= support + price_range * 0.1:
                return -self._min_boost * 0.5

        return 0.0

    def _check_candle_patterns(self, signal: Signal) -> float:
        """Detect simple reversal candle patterns."""
        df = self._market_data.get_dataframe(signal.symbol, Timeframe.M5)
        if df is None or len(df) < 3:
            return 0.0

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        o = float(latest["open"])
        h = float(latest["high"])
        l = float(latest["low"])
        c = float(latest["close"])
        body = abs(c - o)
        candle_range = h - l

        if candle_range == 0:
            return 0.0

        from stocks.core.enums import OrderSide

        # Hammer (bullish reversal): small body at top, long lower wick
        if signal.side == OrderSide.BUY:
            lower_wick = min(o, c) - l
            if body > 0 and lower_wick > body * 2 and (h - max(o, c)) < body * 0.5:
                return self._min_boost  # Hammer pattern

            # Bullish engulfing: current candle body engulfs previous bearish body
            prev_o = float(prev["open"])
            prev_c = float(prev["close"])
            if prev_c < prev_o and c > o:  # Previous bearish, current bullish
                if c > prev_o and o < prev_c:  # Current engulfs previous
                    return self._min_boost

        # Shooting star / bearish engulfing (for SELL signals)
        elif signal.side == OrderSide.SELL:
            upper_wick = h - max(o, c)
            if body > 0 and upper_wick > body * 2 and (min(o, c) - l) < body * 0.5:
                return self._min_boost  # Shooting star

            prev_o = float(prev["open"])
            prev_c = float(prev["close"])
            if prev_c > prev_o and c < o:  # Previous bullish, current bearish
                if o > prev_c and c < prev_o:  # Current engulfs previous
                    return self._min_boost

        return 0.0
