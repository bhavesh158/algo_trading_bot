"""AI-Assisted Analysis (PRD §9).

Adjusts signal confidence based on momentum patterns, volatility context,
and historical trade outcomes. Acts as a decision support filter.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from crypto.core.event_bus import EventBus
from crypto.core.models import Signal
from crypto.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class AIAnalysis:
    """Adjusts signal confidence using pattern-based heuristics."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._market_data: MarketDataEngine | None = None

        ai_cfg = config.get("ai_analysis", {})
        self._enabled = ai_cfg.get("enabled", True)
        self._min_boost = ai_cfg.get("min_confidence_boost", 0.05)
        self._max_boost = ai_cfg.get("max_confidence_boost", 0.25)
        self._lookback = ai_cfg.get("lookback_candles", 50)

        logger.info("AIAnalysis initialized (enabled=%s)", self._enabled)

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        self._market_data = market_data

    def adjust_confidence(self, signal: Signal) -> Signal:
        """Adjust signal confidence based on market context."""
        if not self._enabled or self._market_data is None:
            return signal

        boost = 0.0

        # 1. Momentum alignment
        boost += self._momentum_score(signal.symbol)

        # 2. Volatility context
        boost += self._volatility_score(signal.symbol)

        # 3. Volume confirmation
        boost += self._volume_score(signal.symbol)

        # Clamp boost
        boost = max(self._min_boost, min(boost, self._max_boost))
        signal.confidence = min(signal.confidence + boost, 1.0)
        return signal

    def _momentum_score(self, symbol: str) -> float:
        """Score based on RSI momentum alignment."""
        rsi = self._market_data.get_indicator(symbol, "15m", "rsi")
        if rsi == 0:
            return 0.0
        # Neutral RSI (40-60) is favorable; extremes are risky
        if 40 <= rsi <= 60:
            return 0.05
        elif 30 <= rsi <= 70:
            return 0.02
        return -0.05

    def _volatility_score(self, symbol: str) -> float:
        """Score based on ATR relative to price."""
        df = self._market_data.get_dataframe(symbol, "15m")
        if df.empty or "atr" not in df.columns:
            return 0.0

        atr = df["atr"].iloc[-1]
        price = df["close"].iloc[-1]
        if pd.isna(atr) or price == 0:
            return 0.0

        atr_pct = atr / price * 100
        # Moderate volatility (0.5-2%) is ideal
        if 0.5 <= atr_pct <= 2.0:
            return 0.05
        elif atr_pct > 3.0:
            return -0.1  # Too volatile, reduce confidence
        return 0.0

    def _volume_score(self, symbol: str) -> float:
        """Score based on volume relative to average."""
        df = self._market_data.get_dataframe(symbol, "15m")
        if df.empty or "volume_sma" not in df.columns:
            return 0.0

        vol = df["volume"].iloc[-1]
        vol_sma = df["volume_sma"].iloc[-1]
        if pd.isna(vol_sma) or vol_sma == 0:
            return 0.0

        ratio = vol / vol_sma
        if ratio > 1.5:
            return 0.05
        elif ratio < 0.5:
            return -0.05
        return 0.0
