"""Momentum Breakout Strategy (Enhanced with SELL signals).

Enters when price breaks above a recent high with strong volume confirmation (BUY).
Enters when price breaks below a recent low with volume confirmation (SELL).

Uses ATR-based stops and targets for both directions.

Enhancements:
- Trailing stop: locks in profits on extended breakouts/breakdowns
- VWAP confirmation: breakouts above VWAP, breakdowns below VWAP
- Volume decay exit: exit if momentum fades
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
    """Breakout/breakdown strategy with both BUY and SELL signals.

    BUY: Price breaks above recent high with volume surge, above VWAP
    SELL: Price breaks below recent low with volume surge, below VWAP
    """

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("momentum_breakout", config, market_data)

        strat_config = config.get("momentum_breakout", {})
        self._lookback = strat_config.get("lookback_period", 20)
        self._volume_multiplier = strat_config.get("volume_multiplier", 1.5)
        self._breakout_threshold_pct = strat_config.get("breakout_threshold_pct", 1.0)
        self._atr_stop = strat_config.get("atr_multiplier_stop", 2.0)
        self._atr_target = strat_config.get("atr_multiplier_target", 4.0)
        self._require_vwap = strat_config.get("require_vwap", True)  # Renamed for clarity
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
        low = df["low"]
        volume = df.get("volume")

        current_price = float(close.iloc[-1])
        
        # Calculate breakout/breakdown levels
        recent_high = float(high.iloc[-self._lookback - 1:-1].max())
        recent_low = float(low.iloc[-self._lookback - 1:-1].min())
        breakout_level = recent_high * (1 + self._breakout_threshold_pct / 100)
        breakdown_level = recent_low * (1 - self._breakout_threshold_pct / 100)

        # Get VWAP for filter
        vwap_val = None
        if self._require_vwap:
            vwap = self.market_data.get_indicator(symbol, self.primary_timeframe, "vwap")
            if vwap is not None and not vwap.empty and not np.isnan(vwap.iloc[-1]):
                vwap_val = float(vwap.iloc[-1])

        # Get ATR for stop/target calculation
        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return None
        atr_value = float(atr.iloc[-1])

        # ==================== BUY SIGNAL: Breakout above recent high ====================
        if current_price > breakout_level:
            # VWAP filter: breakout must be above VWAP
            if self._require_vwap and vwap_val is not None:
                if current_price < vwap_val:
                    logger.debug(
                        "[%s] VWAP filter blocked BUY breakout %s: price below VWAP",
                        self.strategy_id, symbol,
                    )
                    return None

            # Volume confirmation
            if volume is not None and not volume.empty:
                current_vol = float(volume.iloc[-1])
                avg_vol = float(volume.iloc[-self._lookback:].mean())
                if avg_vol > 0 and current_vol < avg_vol * self._volume_multiplier:
                    logger.debug(
                        "[%s] Volume filter blocked BUY breakout %s: vol=%.0f < %.0f",
                        self.strategy_id, symbol, current_vol, avg_vol * self._volume_multiplier,
                    )
                    return None

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
                "[%s] Momentum BUY breakout: %s price=%.2f > high=%.2f atr=%.2f conf=%.2f",
                self.strategy_id, symbol, current_price, recent_high, atr_value, confidence,
            )
            return signal

        # ==================== SELL SIGNAL: Breakdown below recent low ====================
        if current_price < breakdown_level:
            # VWAP filter: breakdown must be below VWAP
            if self._require_vwap and vwap_val is not None:
                if current_price > vwap_val:
                    logger.debug(
                        "[%s] VWAP filter blocked SELL breakdown %s: price above VWAP",
                        self.strategy_id, symbol,
                    )
                    return None

            # Volume confirmation (lower multiplier for breakdowns - panic selling needs less)
            if volume is not None and not volume.empty:
                current_vol = float(volume.iloc[-1])
                avg_vol = float(volume.iloc[-self._lookback:].mean())
                # Use 80% of the normal volume requirement for breakdowns
                volume_threshold = avg_vol * self._volume_multiplier * 0.8
                if avg_vol > 0 and current_vol < volume_threshold:
                    logger.debug(
                        "[%s] Volume filter blocked SELL breakdown %s: vol=%.0f < %.0f",
                        self.strategy_id, symbol, current_vol, volume_threshold,
                    )
                    return None

            stop_loss = current_price + self._atr_stop * atr_value
            target = current_price - self._atr_target * atr_value

            # Confidence based on breakdown strength and volume ratio
            breakdown_strength = (recent_low - current_price) / recent_low * 100
            confidence = min(breakdown_strength / 2.0, 1.0)

            # Boost confidence if volume is very strong
            if volume is not None and not volume.empty:
                vol_ratio = float(volume.iloc[-1]) / max(float(volume.iloc[-self._lookback:].mean()), 1)
                if vol_ratio > 2.0:
                    confidence = min(confidence + 0.1, 1.0)

            strength = SignalStrength.STRONG if confidence >= 0.7 else SignalStrength.MODERATE

            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target,
                metadata={
                    "breakdown_level": breakdown_level,
                    "recent_low": recent_low,
                    "atr": atr_value,
                    "max_hold_minutes": self._max_hold_minutes,
                },
            )
            logger.info(
                "[%s] Momentum SELL breakdown: %s price=%.2f < low=%.2f atr=%.2f conf=%.2f",
                self.strategy_id, symbol, current_price, recent_low, atr_value, confidence,
            )
            return signal

        return None

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty:
            return False

        atr_value = float(atr.iloc[-1])
        
        # Direction-aware stop/target
        if entry_price < current_price:  # BUY position
            stop = entry_price - self._atr_stop * atr_value
            target = entry_price + self._atr_target * atr_value
        else:  # SELL position
            stop = entry_price + self._atr_stop * atr_value
            target = entry_price - self._atr_target * atr_value

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
                if entry_price < current_price:  # BUY
                    if current_price > entry_price:
                        logger.info(
                            "[%s] Volume decay exit BUY: %s vol=%.0f < %.0f (%.1f× avg)",
                            self.strategy_id, symbol, current_vol,
                            avg_vol * self._volume_decay_ratio, self._volume_decay_ratio,
                        )
                        return True
                else:  # SELL
                    if current_price < entry_price:
                        logger.info(
                            "[%s] Volume decay exit SELL: %s vol=%.0f < %.0f (%.1f× avg)",
                            self.strategy_id, symbol, current_vol,
                            avg_vol * self._volume_decay_ratio, self._volume_decay_ratio,
                        )
                        return True

        return False
