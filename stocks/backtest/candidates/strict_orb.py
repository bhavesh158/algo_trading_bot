"""Candidate 3: Strict opening-range breakout (honest math edition).

Thesis: ORB works only when the range is meaningful, the break has volume,
and it happens early. Accepts R:R ~1.5 openly (target 6xATR / stop 4xATR);
the test is whether win rate compensates (needs ~65%+ net of Rs.50 costs).
Range = first two 15m bars == max/min of the first thirty 1m bars (exact).
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Any, Optional

import numpy as np

from stocks.core.enums import OrderSide, SignalStrength, Timeframe
from stocks.core.models import Signal
from stocks.data.market_data_engine import MarketDataEngine
from stocks.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StrictORBStrategy(BaseStrategy):
    """First-30-min range, 09:45-10:15 breakouts with volume confirmation."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("strict_orb", config, market_data)
        lock = (config.get("backtest_candidates", {}).get("strict_orb", {}))
        self._or_end = lock.get("range_end", "09:45")
        self._cutoff = lock.get("entry_cutoff", "10:15")
        self._volume_mult = lock.get("volume_mult", 1.3)
        self._min_range_atr = lock.get("min_range_atr_ratio", 0.5)
        self._atr_stop = lock.get("atr_stop", 4.0)
        self._atr_target = lock.get("atr_target", 6.0)
        self._min_stop_pct = lock.get("min_stop_pct", 1.5)
        self.primary_timeframe = Timeframe.M15
        self._max_hold_minutes = lock.get("max_hold_minutes", 300)
        self._fired: dict[str, str] = {}  # symbol -> date string (bar-time based)

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is None or df.empty:
            return None
        last_ts = df.index[-1]
        day_str = last_ts.strftime("%Y-%m-%d")
        if self._fired.get(symbol) == day_str:
            return None  # one per symbol per day
        t = last_ts.time()
        or_end = time(*map(int, self._or_end.split(":")))
        cutoff = time(*map(int, self._cutoff.split(":")))
        if not (or_end <= t <= cutoff):
            return None

        day_bars = df[df.index.date == last_ts.date()]
        or_bars = day_bars[day_bars.index.time < or_end]
        if len(or_bars) < 2:
            return None
        or_high, or_low = float(or_bars["high"].max()), float(or_bars["low"].min())

        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return None
        atr_value = float(atr.iloc[-1])
        if atr_value <= 0 or (or_high - or_low) < self._min_range_atr * atr_value:
            return None

        bar = df.iloc[-1]
        c = float(bar["close"])
        vol = float(bar.get("volume", 0))
        avg_vol = float(df["volume"].iloc[-20:].mean()) if "volume" in df.columns else 0.0
        if avg_vol > 0 and vol < self._volume_mult * avg_vol:
            return None

        if c > or_high:
            side, stop = OrderSide.BUY, c - self._atr_stop * atr_value
            stop = self._apply_min_stop(c, stop, self._atr_stop * atr_value)
            target = c + self._atr_target * atr_value
        elif c < or_low:
            side, stop = OrderSide.SELL, c + self._atr_stop * atr_value
            stop = self._apply_min_stop(c, stop, self._atr_stop * atr_value)
            target = c - self._atr_target * atr_value
        else:
            return None

        self._fired[symbol] = day_str
        return Signal(
            strategy_id=self.strategy_id, symbol=symbol, side=side,
            strength=SignalStrength.MODERATE, confidence=0.6,
            entry_price=c, stop_loss=stop, target_price=target,
            metadata={"max_hold_minutes": self._max_hold_minutes},
        )

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        return False
