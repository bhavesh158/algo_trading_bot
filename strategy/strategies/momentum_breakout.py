"""Momentum Breakout Strategy.

Enters when price breaks above a recent high with strong volume confirmation.
Uses ATR-based stops and targets.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from core.enums import OrderSide, SignalStrength, Timeframe
from core.models import Signal
from data.market_data_engine import MarketDataEngine
from strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy(BaseStrategy):
    """Buy on breakout above recent high with volume surge."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("momentum_breakout", config, market_data)

        strat_config = config.get("momentum_breakout", {})
        self._lookback = strat_config.get("lookback_period", 20)
        self._volume_multiplier = strat_config.get("volume_multiplier", 1.5)
        self._breakout_threshold_pct = strat_config.get("breakout_threshold_pct", 1.0)
        self._atr_stop = strat_config.get("atr_multiplier_stop", 2.0)
        self._atr_target = strat_config.get("atr_multiplier_target", 3.0)
        self.primary_timeframe = Timeframe.M5

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is None or len(df) < self._lookback + 1:
            return None

        close = df["close"]
        high = df["high"]
        volume = df.get("volume")

        current_price = float(close.iloc[-1])
        recent_high = float(high.iloc[-self._lookback - 1:-1].max())
        breakout_level = recent_high * (1 + self._breakout_threshold_pct / 100)

        # Check price breakout
        if current_price <= breakout_level:
            return None

        # Volume confirmation
        if volume is not None and not volume.empty:
            current_vol = float(volume.iloc[-1])
            avg_vol = float(volume.iloc[-self._lookback:].mean())
            if avg_vol > 0 and current_vol < avg_vol * self._volume_multiplier:
                return None  # Breakout without volume — ignore

        # ATR for stop/target calculation
        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return None

        atr_value = float(atr.iloc[-1])
        stop_loss = current_price - self._atr_stop * atr_value
        target = current_price + self._atr_target * atr_value

        # Confidence based on breakout strength and volume ratio
        breakout_strength = (current_price - recent_high) / recent_high * 100
        confidence = min(breakout_strength / 2.0, 1.0)

        strength = SignalStrength.STRONG if confidence >= 0.7 else SignalStrength.MODERATE

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=OrderSide.BUY,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target,
            metadata={
                "breakout_level": breakout_level,
                "recent_high": recent_high,
                "atr": atr_value,
            },
        )
        logger.info(
            "[%s] Momentum breakout: %s price=%.2f > high=%.2f atr=%.2f",
            self.strategy_id, symbol, current_price, recent_high, atr_value,
        )
        return signal

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty:
            return False

        atr_value = float(atr.iloc[-1])
        stop = entry_price - self._atr_stop * atr_value
        target = entry_price + self._atr_target * atr_value

        return current_price <= stop or current_price >= target
