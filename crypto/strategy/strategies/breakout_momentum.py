"""Breakout Momentum Strategy v2 — Retest confirmation with volume spike.

Entry Requirements (ALL must be true):
  1. Price breaks above N-period high (or below N-period low for shorts)
  2. Volume > 2.5x average (strong institutional participation)
  3. Retest: price pulls back to breakout level and holds (confirmation candle)
  4. RSI > 60 for bullish breakout (momentum confirmation)
  
Exit: Trailing stop OR target hit OR volume decay

Why retest matters:
  - False breakouts often reverse immediately
  - Retest confirms genuine supply/demand shift
  - Better entry price = better risk:reward
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from crypto.core.enums import OrderSide, SignalStrength
from crypto.core.models import Signal
from crypto.data.market_data_engine import MarketDataEngine
from crypto.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BreakoutMomentumStrategy(BaseStrategy):
    """Range breakout with retest confirmation and volume spike."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("breakout_momentum", config, market_data)
        sc = config.get("breakout_momentum", {})
        
        # Range settings
        self._lookback = sc.get("lookback_period", 20)
        
        # Volume confirmation - much higher threshold now
        self._vol_mult = sc.get("volume_multiplier", 2.5)
        
        # Buffer to avoid false breakouts
        self._buffer_pct = sc.get("breakout_buffer_pct", 0.3) / 100
        
        # Retest settings
        self._require_retest = sc.get("require_retest", True)
        self._retest_candles = sc.get("retest_candles", 3)  # Max candles to wait for retest
        
        # RSI momentum filter
        self._rsi_min = sc.get("rsi_min_breakout", 60)
        self._rsi_max = sc.get("rsi_max_breakout", 80)  # Avoid overbought entries
        
        # Stop/target settings - improved R:R
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 2.0)
        self._atr_target_mult = sc.get("atr_multiplier_target", 4.0)
        
        # Volume decay exit
        self._vol_decay_threshold = sc.get("volume_decay_threshold", 0.5)
        self._vol_decay_candles = sc.get("volume_decay_candles", 3)
        
        self.primary_timeframe = "15m"

        # Trailing stop + max hold from base
        self._trailing_stop_atr = sc.get("trailing_stop_atr_multiplier", 1.5)
        self._max_hold_minutes = sc.get("max_hold_minutes", 360)  # 6 hours max
        
        # State tracking for retest logic
        self._breakout_detected: dict[str, dict] = {}  # symbol -> breakout info

    def _check_volume_spike(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Check if volume shows institutional participation."""
        volume = df["volume"].iloc[-1]
        vol_sma = df.get("volume_sma", pd.Series(dtype=float))
        
        if vol_sma.empty or pd.isna(vol_sma.iloc[-1]) or vol_sma.iloc[-1] == 0:
            return False, 0.0
            
        avg_volume = vol_sma.iloc[-1]
        volume_ratio = volume / avg_volume
        
        return volume_ratio >= self._vol_mult, volume_ratio

    def _check_rsi_momentum(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Check if RSI confirms momentum without being overbought."""
        rsi = df.get("rsi", pd.Series(dtype=float))
        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return False, 0.0
            
        curr_rsi = rsi.iloc[-1]
        is_valid = self._rsi_min <= curr_rsi <= self._rsi_max
        return is_valid, curr_rsi

    def _detect_breakout(self, df: pd.DataFrame) -> Optional[dict]:
        """Detect if a valid breakout has occurred.
        
        Returns breakout info dict if breakout detected, None otherwise.
        """
        high = df["high"]
        close = df["close"]
        volume = df["volume"]
        
        # Range high over lookback (excluding current candle)
        range_high = high.iloc[-(self._lookback + 1):-1].max()
        breakout_level = range_high * (1 + self._buffer_pct)
        
        current_close = close.iloc[-1]
        current_vol = volume.iloc[-1]
        
        # Check if price broke above range high
        if current_close > breakout_level:
            return {
                "type": "bullish",
                "level": breakout_level,
                "range_high": range_high,
                "breakout_candle_idx": len(df) - 1,
            }
        
        # Check for bearish breakout (range low)
        range_low = high.iloc[-(self._lookback + 1):-1].min()
        breakout_level_low = range_low * (1 - self._buffer_pct)
        
        if current_close < breakout_level_low:
            return {
                "type": "bearish",
                "level": breakout_level_low,
                "range_low": range_low,
                "breakout_candle_idx": len(df) - 1,
            }
        
        return None

    def _check_retest_confirmation(self, df: pd.DataFrame, breakout_info: dict) -> bool:
        """Check if price has successfully retested the breakout level.
        
        For bullish breakout: price should touch breakout level and bounce up
        For bearish breakout: price should touch breakout level and bounce down
        
        The retest candle should show rejection of the breakout level.
        """
        breakout_idx = breakout_info["breakout_candle_idx"]
        if len(df) <= breakout_idx + 1:
            return False  # Not enough candles since breakout
            
        # Get candles since breakout
        candles_since_breakout = df.iloc[breakout_idx + 1:]
        if len(candles_since_breakout) > self._retest_candles:
            candles_since_breakout = candles_since_breakout.iloc[:self._retest_candles]
        
        level = breakout_info["level"]
        breakout_type = breakout_info["type"]
        
        if breakout_type == "bullish":
            # Check if any candle touched/retested the level
            retest_touches = (candles_since_breakout["low"] <= level * 1.005).any()
            
            if not retest_touches:
                # Price hasn't retested yet - wait
                return False
            
            # Check if latest candle shows bullish rejection (long lower wick)
            latest = candles_since_breakout.iloc[-1]
            candle_range = latest["high"] - latest["low"]
            if candle_range == 0:
                return False
            lower_wick_pct = (latest["close"] - latest["low"]) / candle_range
            
            # Close should be in upper half of candle (bullish rejection)
            return lower_wick_pct > 0.5 and latest["close"] > level
            
        else:  # bearish
            retest_touches = (candles_since_breakout["high"] >= level * 0.995).any()
            
            if not retest_touches:
                return False
            
            # Check if latest candle shows bearish rejection (long upper wick)
            latest = candles_since_breakout.iloc[-1]
            candle_range = latest["high"] - latest["low"]
            if candle_range == 0:
                return False
            upper_wick_pct = (latest["high"] - latest["close"]) / candle_range
            
            return upper_wick_pct > 0.5 and latest["close"] < level

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < self._lookback + 5:
            return None

        close = df["close"].iloc[-1]
        atr = df.get("atr", pd.Series(dtype=float))
        
        if atr.empty or pd.isna(atr.iloc[-1]) or atr.iloc[-1] == 0:
            return None
            
        curr_atr = atr.iloc[-1]

        # Check for new breakout
        breakout = self._detect_breakout(df)
        
        if breakout:
            # New breakout detected - store it
            self._breakout_detected[symbol] = breakout
            logger.debug(
                "[breakout] %s BREAKOUT DETECTED: %s at level %.4f",
                symbol, breakout["type"].upper(), breakout["level"],
            )
            
            if not self._require_retest:
                # Immediate entry mode (original strategy behavior)
                volume_confirmed, vol_ratio = self._check_volume_spike(df)
                if not volume_confirmed:
                    return None
                    
                rsi_valid, curr_rsi = self._check_rsi_momentum(df)
                if not rsi_valid:
                    return None
                
                # Generate signal immediately
                if breakout["type"] == "bullish":
                    stop = close - self._atr_stop_mult * curr_atr
                    target = close + self._atr_target_mult * curr_atr
                    side = OrderSide.BUY
                else:
                    stop = close + self._atr_stop_mult * curr_atr
                    target = close - self._atr_target_mult * curr_atr
                    side = OrderSide.SELL
                    
                confidence = min(0.5 + (vol_ratio - self._vol_mult) / 10, 0.85)
                
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side=side,
                    strength=SignalStrength.STRONG if vol_ratio > 4 else SignalStrength.MODERATE,
                    confidence=confidence,
                    entry_price=close,
                    stop_loss=stop,
                    target_price=target,
                    metadata={"vol_ratio": vol_ratio, "rsi": curr_rsi, "entry_type": "immediate"},
                )
        
        # Check for retest entry if we have a stored breakout
        if symbol in self._breakout_detected and self._require_retest:
            breakout_info = self._breakout_detected[symbol]
            
            # Check if retest is confirmed
            retest_confirmed = self._check_retest_confirmation(df, breakout_info)
            
            if retest_confirmed:
                # Verify volume and RSI on retest candle
                volume_confirmed, vol_ratio = self._check_volume_spike(df)
                rsi_valid, curr_rsi = self._check_rsi_momentum(df)
                
                if volume_confirmed and rsi_valid:
                    breakout_type = breakout_info["type"]
                    
                    if breakout_type == "bullish":
                        stop = close - self._atr_stop_mult * curr_atr
                        target = close + self._atr_target_mult * curr_atr
                        side = OrderSide.BUY
                    else:
                        stop = close + self._atr_stop_mult * curr_atr
                        target = close - self._atr_target_mult * curr_atr
                        side = OrderSide.SELL
                    
                    confidence = min(0.55 + (vol_ratio - self._vol_mult) / 10, 0.85)
                    
                    logger.info(
                        "[breakout] %s SIGNAL: %s | Retest confirmed | Vol=%.1fx | RSI=%.1f",
                        symbol, side.name, vol_ratio, curr_rsi,
                    )
                    
                    # Clear the breakout state
                    del self._breakout_detected[symbol]
                    
                    return Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side=side,
                        strength=SignalStrength.STRONG if vol_ratio > 4 else SignalStrength.MODERATE,
                        confidence=confidence,
                        entry_price=close,
                        stop_loss=stop,
                        target_price=target,
                        metadata={"vol_ratio": vol_ratio, "rsi": curr_rsi, "entry_type": "retest"},
                    )
            
            # Check if we've waited too long for retest
            breakout_idx = breakout_info["breakout_candle_idx"]
            candles_since = len(df) - breakout_idx - 1
            if candles_since > self._retest_candles + 2:
                # Breakout failed - price ran without retest or reversed
                logger.debug(
                    "[breakout] %s BREAKOUT EXPIRED: No retest in %d candles",
                    symbol, candles_since,
                )
                del self._breakout_detected[symbol]

        return None

    def should_exit(
        self, symbol: str, entry_price: float, current_price: float,
        side: OrderSide = OrderSide.BUY,
    ) -> bool:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty:
            return False

        atr = df.get("atr", pd.Series(dtype=float))
        if atr.empty or pd.isna(atr.iloc[-1]):
            return False

        # Volume decay exit
        vol_sma = df.get("volume_sma", pd.Series(dtype=float))
        if not vol_sma.empty and len(df) >= self._vol_decay_candles:
            recent_vols = df["volume"].iloc[-self._vol_decay_candles:]
            avg_vol = vol_sma.iloc[-1]
            if avg_vol > 0 and not pd.isna(avg_vol):
                all_low = all(
                    v < self._vol_decay_threshold * avg_vol
                    for v in recent_vols
                    if not pd.isna(v)
                )
                if all_low:
                    logger.debug(
                        "[breakout] %s volume decay exit: last %d candles below %.0f%% avg",
                        symbol, self._vol_decay_candles, self._vol_decay_threshold * 100,
                    )
                    return True

        # Trailing stop
        if side == OrderSide.BUY:
            recent_high = df["high"].iloc[-self._lookback:].max()
            trailing_stop = recent_high - self._atr_stop_mult * atr.iloc[-1]
            return bool(current_price < trailing_stop)
        else:
            recent_low = df["low"].iloc[-self._lookback:].min()
            trailing_stop = recent_low + self._atr_stop_mult * atr.iloc[-1]
            return bool(current_price > trailing_stop)
