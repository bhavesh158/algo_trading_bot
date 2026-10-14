"""Candidate 2: Trend-day follower (trade WITH the index trend or sit out).

Thesis: intraday breakouts work when the market trends; chop kills them.
Filter: ^NSEI 15m EMA9/EMA21 + ADX>20 sets the allowed direction for the
day-bar; symbol must break a 20-bar extreme in that direction on volume.
Stop 2xATR / target 4xATR -> R:R 2.0 by construction.
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

INDEX_SYMBOL = "^NSEI"


INDEX_DEFAULT = "^NSEI"


class TrendFollowStrategy(BaseStrategy):
    """Index-filtered 20-bar breakout with 2R structural payoff."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("trend_follow", config, market_data)
        lock = (config.get("backtest_candidates", {}).get("trend_follow", {}))
        self._index_symbol: str = lock.get("index_symbol", INDEX_DEFAULT)
        self._lookback = lock.get("lookback_bars", 20)
        self._volume_mult = lock.get("volume_mult", 1.2)
        self._adx_min = lock.get("adx_min", 20.0)
        self._atr_stop = lock.get("atr_stop", 2.0)
        self._atr_target = lock.get("atr_target", 4.0)
        self.primary_timeframe = Timeframe.M15
        self._max_hold_minutes = lock.get("max_hold_minutes", 300)

    def _index_direction(self) -> int:
        """+1 up-trend, -1 down-trend, 0 no-trade (all replay-safe reads)."""
        for tf_getter in (lambda: self.market_data.get_indicator(self._index_symbol, self.primary_timeframe, "ema_9"),
                          lambda: self.market_data.get_indicator(self._index_symbol, self.primary_timeframe, "ema_21"),
                          lambda: self.market_data.get_indicator(self._index_symbol, self.primary_timeframe, "adx_14")):
            s = tf_getter()
            if s is None or s.empty or np.isnan(s.iloc[-1]):
                return 0
        ema9 = self.market_data.get_indicator(self._index_symbol, self.primary_timeframe, "ema_9")
        ema21 = self.market_data.get_indicator(self._index_symbol, self.primary_timeframe, "ema_21")
        adx = self.market_data.get_indicator(self._index_symbol, self.primary_timeframe, "adx_14")
        e9, e21, ax = float(ema9.iloc[-1]), float(ema21.iloc[-1]), float(adx.iloc[-1])
        if ax < self._adx_min:
            return 0
        if e9 > e21:
            return 1
        if e9 < e21:
            return -1
        return 0

    def analyze(self, symbol: str) -> Optional[Signal]:
        direction = self._index_direction()
        if direction == 0:
            return None
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is None or len(df) < self._lookback + 5:
            return None
        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return None
        atr_value = float(atr.iloc[-1])
        if atr_value <= 0:
            return None

        bar = df.iloc[-1]
        c = float(bar["close"])
        vol = float(bar.get("volume", 0))
        avg_vol = float(df["volume"].iloc[-20:].mean()) if "volume" in df.columns else 0.0
        if avg_vol > 0 and vol < self._volume_mult * avg_vol:
            return None

        if direction > 0:
            prior_high = float(df["high"].iloc[-(self._lookback + 1):-1].max())
            if c <= prior_high:
                return None
            stop = c - self._atr_stop * atr_value
            target = c + self._atr_target * atr_value
            side = OrderSide.BUY
        else:
            prior_low = float(df["low"].iloc[-(self._lookback + 1):-1].min())
            if c >= prior_low:
                return None
            stop = c + self._atr_stop * atr_value
            target = c - self._atr_target * atr_value
            side = OrderSide.SELL

        conf = 0.55 + (0.1 if avg_vol > 0 and vol >= 1.6 * avg_vol else 0.0)
        return Signal(
            strategy_id=self.strategy_id, symbol=symbol, side=side,
            strength=SignalStrength.MODERATE, confidence=min(conf, 0.8),
            entry_price=c, stop_loss=stop, target_price=target,
            metadata={"max_hold_minutes": self._max_hold_minutes},
        )

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        return False
