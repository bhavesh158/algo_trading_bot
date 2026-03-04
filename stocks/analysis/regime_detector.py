"""Market Regime Detector.

Classifies the current market into regimes (PRD §9):
- Trending up / down
- Sideways
- High / low volatility

The system activates strategies suited to the detected regime
and reduces activity when the regime is unclear.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from stocks.core.enums import MarketRegime, Timeframe
from stocks.core.event_bus import EventBus
from stocks.core.events import RegimeChangeEvent
from stocks.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects the current market regime using ADX and volatility percentiles."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._market_data: Optional[MarketDataEngine] = None

        regime_config = config.get("regime_detection", {})
        self._enabled = regime_config.get("enabled", True)
        self._adx_threshold = regime_config.get("adx_trending_threshold", 25)
        self._vol_high_pct = regime_config.get("volatility_high_percentile", 80)
        self._vol_low_pct = regime_config.get("volatility_low_percentile", 20)
        self._lookback = regime_config.get("lookback_period", 20)

        self._current_regime = MarketRegime.UNKNOWN
        self._index_symbol = "^NSEI"  # Nifty 50 as market proxy

        logger.info("RegimeDetector initialized (enabled=%s)", self._enabled)

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        self._market_data = market_data

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    def detect_regime(self) -> MarketRegime:
        """Analyze market conditions and classify the current regime."""
        if not self._enabled or self._market_data is None:
            return MarketRegime.UNKNOWN

        previous = self._current_regime

        # Detect trend strength using ADX
        adx = self._market_data.get_indicator(self._index_symbol, Timeframe.M15, "adx_14")
        ema_9 = self._market_data.get_indicator(self._index_symbol, Timeframe.M15, "ema_9")
        ema_21 = self._market_data.get_indicator(self._index_symbol, Timeframe.M15, "ema_21")
        atr = self._market_data.get_indicator(self._index_symbol, Timeframe.M15, "atr_14")

        # Default to UNKNOWN if insufficient data
        if adx is None or adx.empty or atr is None or atr.empty:
            self._current_regime = MarketRegime.UNKNOWN
            return self._current_regime

        adx_val = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0

        # Classify trend
        is_trending = adx_val >= self._adx_threshold

        # Classify direction
        trend_up = False
        trend_down = False
        if ema_9 is not None and ema_21 is not None and not ema_9.empty and not ema_21.empty:
            trend_up = float(ema_9.iloc[-1]) > float(ema_21.iloc[-1])
            trend_down = float(ema_9.iloc[-1]) < float(ema_21.iloc[-1])

        # Classify volatility
        if len(atr) >= self._lookback:
            atr_window = atr.iloc[-self._lookback:]
            current_atr = float(atr.iloc[-1])
            high_threshold = float(np.percentile(atr_window.dropna(), self._vol_high_pct))
            low_threshold = float(np.percentile(atr_window.dropna(), self._vol_low_pct))
            is_high_vol = current_atr >= high_threshold
            is_low_vol = current_atr <= low_threshold
        else:
            is_high_vol = False
            is_low_vol = False

        # Determine regime
        if is_high_vol:
            self._current_regime = MarketRegime.HIGH_VOLATILITY
        elif is_trending and trend_up:
            self._current_regime = MarketRegime.TRENDING_UP
        elif is_trending and trend_down:
            self._current_regime = MarketRegime.TRENDING_DOWN
        elif is_low_vol:
            self._current_regime = MarketRegime.LOW_VOLATILITY
        elif not is_trending:
            self._current_regime = MarketRegime.SIDEWAYS
        else:
            self._current_regime = MarketRegime.UNKNOWN

        # Publish event if regime changed
        if self._current_regime != previous:
            logger.info(
                "Market regime changed: %s -> %s (ADX=%.1f)",
                previous.name, self._current_regime.name, adx_val,
            )
            self.event_bus.publish(RegimeChangeEvent(
                previous_regime=previous,
                current_regime=self._current_regime,
            ))

        return self._current_regime
