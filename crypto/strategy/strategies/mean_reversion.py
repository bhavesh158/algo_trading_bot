"""Mean Reversion Strategy — Bollinger Band touch with RSI confirmation.

Entry: Price touches lower BB AND RSI < oversold threshold.
Exit:  Price returns to BB midline OR RSI > overbought OR stop/target hit.
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


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band reversion with RSI divergence."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("mean_reversion", config, market_data)
        sc = config.get("mean_reversion", {})
        self._rsi_oversold = sc.get("rsi_oversold", 30)
        self._rsi_overbought = sc.get("rsi_overbought", 70)
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 1.5)
        self._atr_target_mult = sc.get("atr_multiplier_target", 2.0)
        self.primary_timeframe = "15m"

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < 25:
            return None

        close = df["close"].iloc[-1]
        bb_lower = df.get("bb_lower", pd.Series(dtype=float))
        rsi = df.get("rsi", pd.Series(dtype=float))
        atr = df.get("atr", pd.Series(dtype=float))

        if bb_lower.empty or rsi.empty or atr.empty:
            return None

        curr_bb_lower = bb_lower.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        curr_atr = atr.iloc[-1]

        if pd.isna(curr_bb_lower) or pd.isna(curr_rsi) or pd.isna(curr_atr) or curr_atr == 0:
            return None

        # Buy when price touches lower BB and RSI is oversold
        if close <= curr_bb_lower and curr_rsi < self._rsi_oversold:
            stop = close - self._atr_stop_mult * curr_atr
            bb_mid = df["bb_mid"].iloc[-1] if "bb_mid" in df.columns else close
            target = bb_mid + self._atr_target_mult * curr_atr * 0.5

            # Deeper oversold = higher confidence
            confidence = min(0.5 + (self._rsi_oversold - curr_rsi) / 60, 0.85)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=close,
                stop_loss=stop,
                target_price=target,
                metadata={"rsi": curr_rsi, "bb_lower": curr_bb_lower},
            )

        return None

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty:
            return False

        bb_mid = df.get("bb_mid", pd.Series(dtype=float))
        rsi = df.get("rsi", pd.Series(dtype=float))

        if bb_mid.empty or rsi.empty:
            return False

        # Exit when price returns to BB mid or RSI goes overbought
        return bool(current_price >= bb_mid.iloc[-1] or rsi.iloc[-1] > self._rsi_overbought)
