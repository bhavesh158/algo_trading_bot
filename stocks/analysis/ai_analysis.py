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
