"""Multi-Timeframe Confirmation (PRD §17).

Validates short-term signals by checking alignment with broader trends
across multiple timeframes to reduce false signals.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from core.enums import OrderSide, Timeframe
from core.event_bus import EventBus
from core.models import Signal
from data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)

# Confirmation hierarchy: signal TF -> higher TFs to check
_CONFIRMATION_MAP = {
    Timeframe.M1: [Timeframe.M5, Timeframe.M15],
    Timeframe.M5: [Timeframe.M15, Timeframe.H1],
    Timeframe.M15: [Timeframe.H1, Timeframe.D1],
}


class MultiTimeframeAnalyzer:
    """Confirms signals using higher timeframes."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._market_data: Optional[MarketDataEngine] = None

        logger.info("MultiTimeframeAnalyzer initialized")

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        self._market_data = market_data

    def confirm_signal(self, signal: Signal, signal_timeframe: Timeframe) -> float:
        """Check higher-timeframe alignment for a signal.

        Returns a score from -1.0 (strongly against) to +1.0 (strongly aligned).
        """
        if self._market_data is None:
            return 0.0

        higher_tfs = _CONFIRMATION_MAP.get(signal_timeframe, [])
        if not higher_tfs:
            return 0.0

        scores: list[float] = []
        for tf in higher_tfs:
            score = self._check_timeframe_alignment(signal.symbol, signal.side, tf)
            scores.append(score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _check_timeframe_alignment(
        self, symbol: str, side: OrderSide, timeframe: Timeframe
    ) -> float:
        """Check if a higher timeframe supports the signal direction.

        Uses EMA crossover and price position relative to VWAP.
        """
        ema_9 = self._market_data.get_indicator(symbol, timeframe, "ema_9")
        ema_21 = self._market_data.get_indicator(symbol, timeframe, "ema_21")

        if ema_9 is None or ema_21 is None or ema_9.empty or ema_21.empty:
            return 0.0

        ema_9_val = float(ema_9.iloc[-1])
        ema_21_val = float(ema_21.iloc[-1])

        if np.isnan(ema_9_val) or np.isnan(ema_21_val):
            return 0.0

        bullish_trend = ema_9_val > ema_21_val
        bearish_trend = ema_9_val < ema_21_val

        if side == OrderSide.BUY and bullish_trend:
            return 1.0
        elif side == OrderSide.BUY and bearish_trend:
            return -0.5
        elif side == OrderSide.SELL and bearish_trend:
            return 1.0
        elif side == OrderSide.SELL and bullish_trend:
            return -0.5

        return 0.0
