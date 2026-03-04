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
    """EMA crossover with ADX trend confirmation."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("trend_following", config, market_data)
        sc = config.get("trend_following", {})
        self._fast_ema = sc.get("fast_ema", 9)
        self._slow_ema = sc.get("slow_ema", 21)
        self._adx_threshold = sc.get("adx_threshold", 25)
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 2.0)
        self._atr_target_mult = sc.get("atr_multiplier_target", 3.0)
        self.primary_timeframe = "15m"

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

        # Check for bullish crossover: fast crosses above slow
        if (ema_fast.iloc[-1] > ema_slow.iloc[-1] and
                ema_fast.iloc[-2] <= ema_slow.iloc[-2] and
                curr_adx >= self._adx_threshold):

            price = float(df["close"].iloc[-1])
            stop = price - self._atr_stop_mult * curr_atr
            target = price + self._atr_target_mult * curr_atr
            confidence = min(0.5 + (curr_adx - self._adx_threshold) / 50, 0.9)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=price,
                stop_loss=stop,
                target_price=target,
                metadata={"adx": curr_adx, "atr": curr_atr},
            )

        return None

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < self._slow_ema + 2:
            return False

        ema_fast = df.get("ema_9", pd.Series(dtype=float))
        ema_slow = df.get("ema_21", pd.Series(dtype=float))
        if ema_fast.empty or ema_slow.empty:
            return False

        # Exit on bearish crossover
        return bool(ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2])
