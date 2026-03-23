"""Trend Following Strategy v2 — Enhanced EMA crossover with multiple confirmations.

Entry Requirements (ALL must be true):
  1. Fast EMA crosses above/below Slow EMA
  2. ADX > 30 (strong trend present)
  3. Volume > 1.5x average (institutional participation)
  4. Price action: close in top/bottom 25% of candle range (momentum confirmation)
  5. Higher timeframe (1h) trend alignment

Exit: EMA cross against position OR trailing stop OR target hit
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


class TrendFollowingStrategy(BaseStrategy):
    """Enhanced EMA crossover with ADX, volume, and price action confirmations."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("trend_following", config, market_data)
        sc = config.get("trend_following", {})
        
        # EMA settings
        self._fast_ema = sc.get("fast_ema", 9)
        self._slow_ema = sc.get("slow_ema", 21)
        
        # ADX threshold - must be > 30 for strong trend
        self._adx_threshold = sc.get("adx_threshold", 30)
        
        # Volume confirmation - volume must be > this multiplier of average
        self._volume_mult = sc.get("volume_multiplier", 1.5)
        
        # Price action filter - close must be in top/bottom X% of candle range
        self._price_action_pct = sc.get("price_action_position_pct", 25)
        
        # Stop/target settings
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 2.5)
        self._atr_target_mult = sc.get("atr_multiplier_target", 5.0)  # Increased for better R:R
        
        # Higher timeframe alignment
        self._require_htf = sc.get("require_htf_alignment", True)
        self.primary_timeframe = "15m"
        self._htf = "1h"

        # Trailing stop + max hold from base
        self._trailing_stop_atr = sc.get("trailing_stop_atr_multiplier", 2.0)
        self._max_hold_minutes = sc.get("max_hold_minutes", 480)  # 8 hours max

    def _htf_trend_agrees(self, symbol: str, side: OrderSide) -> bool:
        """Check that the 1h EMA trend agrees with the signal direction."""
        df_1h = self.market_data.get_dataframe(symbol, self._htf)
        if df_1h.empty or len(df_1h) < self._slow_ema + 2:
            return False

        fast = df_1h.get("ema_9", pd.Series(dtype=float))
        slow = df_1h.get("ema_21", pd.Series(dtype=float))
        if fast.empty or slow.empty:
            return False

        if side == OrderSide.BUY:
            return bool(fast.iloc[-1] > slow.iloc[-1])
        else:
            return bool(fast.iloc[-1] < slow.iloc[-1])

    def _check_price_action(self, df: pd.DataFrame, side: OrderSide) -> bool:
        """Check if candle close is in favorable position for momentum.
        
        For BUY: close should be in top 25% of candle range (high - low)
        For SELL: close should be in bottom 25% of candle range
        """
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]
        close = df["close"].iloc[-1]
        
        candle_range = high - low
        if candle_range == 0:
            return False
            
        if side == OrderSide.BUY:
            # Close should be in top 25% of range
            position_in_candle = (close - low) / candle_range
            return position_in_candle >= (100 - self._price_action_pct) / 100
        else:
            # Close should be in bottom 25% of range
            position_in_candle = (close - low) / candle_range
            return position_in_candle <= self._price_action_pct / 100

    def _check_volume_confirmation(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Check if volume confirms the breakout.
        
        Returns: (is_confirmed, volume_ratio)
        """
        volume = df["volume"].iloc[-1]
        vol_sma = df.get("volume_sma", pd.Series(dtype=float))
        
        if vol_sma.empty or pd.isna(vol_sma.iloc[-1]) or vol_sma.iloc[-1] == 0:
            return False, 0.0
            
        avg_volume = vol_sma.iloc[-1]
        volume_ratio = volume / avg_volume
        
        return volume_ratio >= self._volume_mult, volume_ratio

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < self._slow_ema + 5:
            return None

        # Get indicators
        ema_fast = df[f"ema_{self._fast_ema}"] if f"ema_{self._fast_ema}" in df.columns else df["ema_9"]
        ema_slow = df[f"ema_{self._slow_ema}"] if f"ema_{self._slow_ema}" in df.columns else df["ema_21"]
        adx = df.get("adx", pd.Series(dtype=float))
        atr = df.get("atr", pd.Series(dtype=float))

        if adx.empty or atr.empty:
            return None

        curr_adx = adx.iloc[-1]
        curr_atr = atr.iloc[-1]
        if pd.isna(curr_adx) or pd.isna(curr_atr) or curr_atr == 0:
            return None

        # Check for EMA crossover
        fast_now = ema_fast.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_now = ema_slow.iloc[-1]
        slow_prev = ema_slow.iloc[-2]
        
        bullish_cross = fast_now > slow_now and fast_prev <= slow_prev
        bearish_cross = fast_now < slow_now and fast_prev >= slow_prev

        # No crossover = no signal
        if not bullish_cross and not bearish_cross:
            return None

        # FILTER 1: ADX must confirm strong trend
        if curr_adx < self._adx_threshold:
            logger.debug(
                "[trend_following] %s SKIP: ADX=%.1f < threshold %d",
                symbol, curr_adx, self._adx_threshold,
            )
            return None

        # Determine signal side
        side = OrderSide.BUY if bullish_cross else OrderSide.SELL
        price = float(df["close"].iloc[-1])

        # FILTER 2: Volume confirmation
        volume_confirmed, vol_ratio = self._check_volume_confirmation(df)
        if not volume_confirmed:
            logger.debug(
                "[trend_following] %s SKIP: Volume ratio %.1fx < required %.1fx",
                symbol, vol_ratio, self._volume_mult,
            )
            return None

        # FILTER 3: Price action confirmation
        if not self._check_price_action(df, side):
            logger.debug(
                "[trend_following] %s SKIP: Price action not favorable for %s",
                symbol, side.name,
            )
            return None

        # FILTER 4: Higher timeframe alignment
        if self._require_htf and not self._htf_trend_agrees(symbol, side):
            logger.debug(
                "[trend_following] %s BLOCKED: 1h trend disagrees with %s",
                symbol, side.name,
            )
            return None

        # All filters passed - calculate stops and targets
        if side == OrderSide.BUY:
            stop = price - self._atr_stop_mult * curr_atr
            target = price + self._atr_target_mult * curr_atr
        else:
            stop = price + self._atr_stop_mult * curr_atr
            target = price - self._atr_target_mult * curr_atr

        # Confidence based on ADX strength and volume
        adx_component = min((curr_adx - self._adx_threshold) / 20, 0.2)
        vol_component = min((vol_ratio - self._volume_mult) / 5, 0.2)
        confidence = 0.5 + adx_component + vol_component
        confidence = min(confidence, 0.9)

        ema_sep_pct = abs(fast_now - slow_now) / slow_now * 100 if slow_now > 0 else 0
        
        logger.info(
            "[trend_following] %s SIGNAL: %s | EMA sep=%.2f%% | ADX=%.1f | Vol=%.1fx | Price OK | HTF OK",
            symbol, side.name, ema_sep_pct, curr_adx, vol_ratio,
        )

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop,
            target_price=target,
            metadata={
                "adx": curr_adx,
                "atr": curr_atr,
                "vol_ratio": vol_ratio,
                "htf_aligned": True,
            },
        )

    def should_exit(
        self, symbol: str, entry_price: float, current_price: float,
        side: OrderSide = OrderSide.BUY,
    ) -> bool:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < self._slow_ema + 2:
            return False

        ema_fast = df.get("ema_9", pd.Series(dtype=float))
        ema_slow = df.get("ema_21", pd.Series(dtype=float))
        if ema_fast.empty or ema_slow.empty:
            return False

        fast_now = ema_fast.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_now = ema_slow.iloc[-1]
        slow_prev = ema_slow.iloc[-2]

        if side == OrderSide.BUY:
            return bool(fast_now < slow_now and fast_prev >= slow_prev)
        else:
            return bool(fast_now > slow_now and fast_prev <= slow_prev)
