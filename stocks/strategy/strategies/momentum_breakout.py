"""Momentum Breakout Strategy (Enhanced).

Enters when price breaks above a recent high with strong volume confirmation.
Uses ATR-based stops and targets.

Enhancements:
- Trailing stop: locks in profits on extended breakouts
- VWAP confirmation: only enter breakouts above VWAP
- Volume decay exit: exit if momentum fades (volume drops)
- Max hold duration
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from stocks.core.enums import OrderSide, SignalStrength, Timeframe
from stocks.core.models import Signal
from stocks.data.market_data_engine import MarketDataEngine
from stocks.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy(BaseStrategy):
    """Buy on breakout above recent high with volume surge.

    Enhanced with trailing stop, VWAP filter, and volume decay detection.
    """

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("momentum_breakout", config, market_data)

        strat_config = config.get("momentum_breakout", {})
        self._lookback = strat_config.get("lookback_period", 20)
        self._volume_multiplier = strat_config.get("volume_multiplier", 1.5)
        self._breakout_threshold_pct = strat_config.get("breakout_threshold_pct", 1.0)
        self._atr_stop = strat_config.get("atr_multiplier_stop", 2.0)
        self._atr_target = strat_config.get("atr_multiplier_target", 4.0)
        self._require_vwap_above = strat_config.get("require_vwap_above", True)
        self._volume_decay_ratio = strat_config.get("volume_decay_exit_ratio", 0.7)
        self.primary_timeframe = Timeframe.M5

        # Enable trailing stop and time exit from base class
        self._trailing_stop_atr = strat_config.get("trailing_stop_atr", 1.5)
        self._max_hold_minutes = strat_config.get("max_hold_minutes", 180)

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

        # VWAP confirmation: breakout must be above VWAP
        if self._require_vwap_above:
            vwap = self.market_data.get_indicator(symbol, self.primary_timeframe, "vwap")
            if vwap is not None and not vwap.empty and not np.isnan(vwap.iloc[-1]):
                if current_price < float(vwap.iloc[-1]):
                    logger.debug(
                        "[%s] VWAP filter blocked breakout for %s: price below VWAP",
                        self.strategy_id, symbol,
                    )
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

        # Boost confidence if volume is very strong
        if volume is not None and not volume.empty:
            vol_ratio = float(volume.iloc[-1]) / max(float(volume.iloc[-self._lookback:].mean()), 1)
            if vol_ratio > 2.0:
                confidence = min(confidence + 0.1, 1.0)

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
                "max_hold_minutes": self._max_hold_minutes,
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

        # Fixed stop/target check
        if current_price <= stop or current_price >= target:
            return True

        # Volume decay exit: momentum fading
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is not None and "volume" in df.columns and len(df) >= self._lookback:
            current_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].iloc[-self._lookback:].mean())
            if avg_vol > 0 and current_vol < avg_vol * self._volume_decay_ratio:
                # Only exit on volume decay if we're in profit (don't lock in losses)
                if current_price > entry_price:
                    logger.info(
                        "[%s] Volume decay exit: %s vol=%.0f < %.0f (%.1f× avg)",
                        self.strategy_id, symbol, current_vol,
                        avg_vol * self._volume_decay_ratio, self._volume_decay_ratio,
                    )
                    return True

        return False
