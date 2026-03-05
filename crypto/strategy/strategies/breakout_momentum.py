"""Breakout Momentum Strategy — volume-confirmed range breakout.

Entry: Price breaks above N-period high with volume > multiplier * avg volume.
Exit:  Price drops below trailing stop or target hit.
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
    """Range breakout confirmed by volume spike."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("breakout_momentum", config, market_data)
        sc = config.get("breakout_momentum", {})
        self._lookback = sc.get("lookback_period", 20)
        self._vol_mult = sc.get("volume_multiplier", 2.0)
        self._buffer_pct = sc.get("breakout_buffer_pct", 0.2) / 100
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 2.0)
        self._atr_target_mult = sc.get("atr_multiplier_target", 3.0)
        self.primary_timeframe = "1h"

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < self._lookback + 5:
            return None

        close = df["close"]
        high = df["high"]
        volume = df["volume"]
        atr = df.get("atr", pd.Series(dtype=float))
        vol_sma = df.get("volume_sma", pd.Series(dtype=float))

        if atr.empty or vol_sma.empty:
            return None

        curr_atr = atr.iloc[-1]
        curr_vol_sma = vol_sma.iloc[-1]
        if pd.isna(curr_atr) or pd.isna(curr_vol_sma) or curr_atr == 0 or curr_vol_sma == 0:
            return None

        # Range high over lookback period (excluding current candle)
        range_high = high.iloc[-(self._lookback + 1):-1].max()
        breakout_level = range_high * (1 + self._buffer_pct)
        current_close = close.iloc[-1]
        current_vol = volume.iloc[-1]

        logger.debug(
            "[breakout] %s | close=%.4f | range_high=%.4f (break=%.4f) | vol=%.0f avg=%.0f (need %.1fx)",
            symbol, current_close, range_high, breakout_level,
            current_vol, curr_vol_sma, self._vol_mult,
        )

        # Breakout: close above range high + buffer, volume confirms
        if current_close > breakout_level and current_vol > self._vol_mult * curr_vol_sma:
            stop = current_close - self._atr_stop_mult * curr_atr
            target = current_close + self._atr_target_mult * curr_atr

            vol_ratio = current_vol / curr_vol_sma
            confidence = min(0.5 + (vol_ratio - self._vol_mult) / 10, 0.9)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=SignalStrength.STRONG if vol_ratio > 3 else SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=current_close,
                stop_loss=stop,
                target_price=target,
                metadata={"range_high": range_high, "vol_ratio": vol_ratio},
            )

        return None

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty:
            return False

        atr = df.get("atr", pd.Series(dtype=float))
        if atr.empty or pd.isna(atr.iloc[-1]):
            return False

        # Trailing stop: exit if price drops more than 2x ATR from highest since entry
        recent_high = df["high"].iloc[-self._lookback:].max()
        trailing_stop = recent_high - self._atr_stop_mult * atr.iloc[-1]
        return bool(current_price < trailing_stop)
