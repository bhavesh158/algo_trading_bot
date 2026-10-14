"""Candidate 1: Volatility-expansion breakout (asymmetric payoff design).

Thesis: intraday directional commitment (wide-range bar closing on its
extreme, on high volume) continues more often than VWAP dips revert.
Target = 2x structural risk -> R:R 2.0 by construction (honest gate).
No VWAP magnet, no mean reversion.
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


class VolatilityExpansionStrategy(BaseStrategy):
    """Enter on range-expansion bars closing on their extreme + volume."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("vol_expansion", config, market_data)
        lock = (config.get("backtest_candidates", {}).get("vol_expansion", {}))
        self._range_atr_mult = lock.get("range_atr_mult", 1.5)
        self._extreme_frac = lock.get("extreme_frac", 0.75)
        self._volume_mult = lock.get("volume_mult", 1.3)
        self._target_rr = lock.get("target_rr", 2.0)
        self._stop_atr_buf = lock.get("stop_atr_buffer", 0.1)
        self.primary_timeframe = Timeframe.M15
        self._max_hold_minutes = lock.get("max_hold_minutes", 300)

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is None or len(df) < 25:
            return None
        bar = df.iloc[-1]
        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return None
        atr_value = float(atr.iloc[-1])
        if atr_value <= 0:
            return None

        o, h, low, c = (float(bar["open"]), float(bar["high"]),
                        float(bar["low"]), float(bar["close"]))
        rng = h - low
        if rng < self._range_atr_mult * atr_value:
            return None  # no expansion

        vol = float(bar.get("volume", 0))
        avg_vol = float(df["volume"].iloc[-20:].mean()) if "volume" in df.columns else 0.0
        if avg_vol > 0 and vol < self._volume_mult * avg_vol:
            return None

        long_close = (c - low) / rng if rng > 0 else 0.5
        side = None
        if long_close >= self._extreme_frac:
            side = OrderSide.BUY
        elif long_close <= 1 - self._extreme_frac:
            side = OrderSide.SELL
        else:
            return None  # mid-range close: no commitment

        buf = self._stop_atr_buf * atr_value
        if side == OrderSide.BUY:
            stop = low - buf
            risk = c - stop
            target = c + self._target_rr * risk
        else:
            stop = h + buf
            risk = stop - c
            target = c - self._target_rr * risk
        if risk <= 0:
            return None

        conf = 0.55 + (0.1 if avg_vol > 0 and vol >= 1.6 * avg_vol else 0.0)
        conf = min(conf, 0.8)
        return Signal(
            strategy_id=self.strategy_id, symbol=symbol, side=side,
            strength=SignalStrength.MODERATE, confidence=conf,
            entry_price=c, stop_loss=stop, target_price=target,
            metadata={"max_hold_minutes": self._max_hold_minutes,
                      "bar_range_atr": rng / atr_value},
        )

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        return False  # exits: stop / target / time / square-off via harness
