"""Trend Following Strategy — EMA crossover with ADX confirmation.

Entry: Fast EMA crosses above slow EMA AND ADX > threshold (trend present).
Exit:  Fast EMA crosses below slow EMA OR stop/target hit.
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
    """EMA crossover with ADX confirmation and higher-timeframe alignment."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("trend_following", config, market_data)
        sc = config.get("trend_following", {})
        self._fast_ema = sc.get("fast_ema", 9)
        self._slow_ema = sc.get("slow_ema", 21)
        self._adx_threshold = sc.get("adx_threshold", 30)
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 2.5)
        self._atr_target_mult = sc.get("atr_multiplier_target", 4.0)
        self._require_htf = sc.get("require_htf_alignment", True)
        self._vol_confirm_mult = sc.get("volume_confirm_mult", 1.2)
        self.primary_timeframe = "15m"
        self._htf = "1h"

        # Trailing stop + max hold from base
        self._trailing_stop_atr = sc.get("trailing_stop_atr_multiplier", 2.0)
        self._max_hold_minutes = sc.get("max_hold_minutes", 360)

    def _htf_trend_agrees(self, symbol: str, side: OrderSide) -> bool:
        """Check that the 1h EMA trend agrees with the signal direction.

        Prevents taking 15m crossover signals that fight the higher-timeframe trend.
        """
        df_1h = self.market_data.get_dataframe(symbol, self._htf)
        if df_1h.empty or len(df_1h) < self._slow_ema + 2:
            return False  # no data = no confirmation = no trade

        fast = df_1h.get("ema_9", pd.Series(dtype=float))
        slow = df_1h.get("ema_21", pd.Series(dtype=float))
        if fast.empty or slow.empty:
            return False

        if side == OrderSide.BUY:
            return bool(fast.iloc[-1] > slow.iloc[-1])  # 1h bullish
        else:
            return bool(fast.iloc[-1] < slow.iloc[-1])  # 1h bearish

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < self._slow_ema + 5:
            return None

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

        fast_now = ema_fast.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_now = ema_slow.iloc[-1]
        slow_prev = ema_slow.iloc[-2]
        bullish_cross = fast_now > slow_now and fast_prev <= slow_prev
        bearish_cross = fast_now < slow_now and fast_prev >= slow_prev

        # No crossover = no signal (trend continuation removed — too many false entries)
        if not bullish_cross and not bearish_cross:
            return None

        # ADX must confirm a real trend is present
        if curr_adx < self._adx_threshold:
            return None

        # Volume confirmation: crossover candle must have above-average volume
        vol_sma = df.get("volume_sma", pd.Series(dtype=float))
        if not vol_sma.empty and not pd.isna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0:
            curr_vol = df["volume"].iloc[-1]
            if curr_vol < self._vol_confirm_mult * vol_sma.iloc[-1]:
                logger.debug(
                    "[trend_following] %s SKIP: low volume on crossover (%.0f < %.1f×%.0f)",
                    symbol, curr_vol, self._vol_confirm_mult, vol_sma.iloc[-1],
                )
                return None

        ema_sep_pct = abs(fast_now - slow_now) / slow_now * 100 if slow_now > 0 else 0
        price = float(df["close"].iloc[-1])

        logger.debug(
            "[trend_following] %s | EMA9=%.4f EMA21=%.4f (sep=%.2f%%) | ADX=%.1f (need>%d) | "
            "bull_cross=%s bear_cross=%s",
            symbol, fast_now, slow_now, ema_sep_pct, curr_adx, self._adx_threshold,
            bullish_cross, bearish_cross,
        )

        side = OrderSide.BUY if bullish_cross else OrderSide.SELL

        # Higher-timeframe alignment: don't fight the 1h trend
        if self._require_htf and not self._htf_trend_agrees(symbol, side):
            logger.debug(
                "[trend_following] %s | BLOCKED: 1h trend disagrees with %s",
                symbol, side.name,
            )
            return None

        if side == OrderSide.BUY:
            stop = price - self._atr_stop_mult * curr_atr
            target = price + self._atr_target_mult * curr_atr
        else:
            stop = price + self._atr_stop_mult * curr_atr
            target = price - self._atr_target_mult * curr_atr

        confidence = min(0.5 + (curr_adx - self._adx_threshold) / 50, 0.9)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop,
            target_price=target,
            metadata={"adx": curr_adx, "atr": curr_atr, "htf_aligned": True},
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
            # Long: exit on bearish crossover (fast crosses below slow)
            return bool(fast_now < slow_now and fast_prev >= slow_prev)
        else:
            # Short: exit on bullish crossover (fast crosses above slow)
            return bool(fast_now > slow_now and fast_prev <= slow_prev)
