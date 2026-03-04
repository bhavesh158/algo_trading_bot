"""Market Regime Detection (PRD §10).

Detects the current market regime (trending, sideways, high/low volatility)
to guide strategy activation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from crypto.core.enums import MarketRegime
from crypto.core.event_bus import EventBus
from crypto.core.events import RegimeChangeEvent
from crypto.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regime from ADX and volatility percentile."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._market_data: MarketDataEngine | None = None

        rc = config.get("regime_detection", {})
        self._enabled = rc.get("enabled", True)
        self._adx_threshold = rc.get("adx_trending_threshold", 25)
        self._vol_high_pct = rc.get("volatility_high_percentile", 80)
        self._vol_low_pct = rc.get("volatility_low_percentile", 20)
        self._lookback = rc.get("lookback_period", 20)

        self.current_regime = MarketRegime.UNKNOWN
        self._reference_symbol = "BTC/USDT"
        logger.info("RegimeDetector initialized")

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        self._market_data = market_data

    def detect_regime(self) -> MarketRegime:
        """Detect current market regime from the reference symbol."""
        if not self._enabled or self._market_data is None:
            return self.current_regime

        df = self._market_data.get_dataframe(self._reference_symbol, "1h")
        if df.empty or len(df) < self._lookback:
            # Fallback: try any loaded symbol
            for sym in self._market_data.symbols_loaded:
                df = self._market_data.get_dataframe(sym, "1h")
                if not df.empty and len(df) >= self._lookback:
                    break
            else:
                return self.current_regime

        adx = df.get("adx", pd.Series(dtype=float))
        atr = df.get("atr", pd.Series(dtype=float))

        if adx.empty or atr.empty:
            return self.current_regime

        curr_adx = adx.iloc[-1]
        if pd.isna(curr_adx):
            return self.current_regime

        # Determine volatility percentile
        atr_vals = atr.dropna().iloc[-self._lookback:]
        if len(atr_vals) < 5:
            return self.current_regime

        curr_atr = atr_vals.iloc[-1]
        vol_percentile = (atr_vals < curr_atr).sum() / len(atr_vals) * 100

        # Classify regime
        previous = self.current_regime

        if vol_percentile >= self._vol_high_pct:
            self.current_regime = MarketRegime.HIGH_VOLATILITY
        elif vol_percentile <= self._vol_low_pct:
            self.current_regime = MarketRegime.LOW_VOLATILITY
        elif curr_adx >= self._adx_threshold:
            # Determine trend direction from EMA
            ema_fast = df.get("ema_9", pd.Series(dtype=float))
            ema_slow = df.get("ema_21", pd.Series(dtype=float))
            if not ema_fast.empty and not ema_slow.empty:
                if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                    self.current_regime = MarketRegime.TRENDING_UP
                else:
                    self.current_regime = MarketRegime.TRENDING_DOWN
            else:
                self.current_regime = MarketRegime.TRENDING_UP
        else:
            self.current_regime = MarketRegime.SIDEWAYS

        if self.current_regime != previous:
            logger.info("Regime changed: %s -> %s", previous.name, self.current_regime.name)
            self.event_bus.publish(RegimeChangeEvent(
                previous_regime=previous,
                current_regime=self.current_regime,
            ))

        return self.current_regime
