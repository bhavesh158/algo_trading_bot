"""Volatility Protection (PRD §16).

Detects abnormal price movements and temporarily pauses trading
when market conditions become unstable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from crypto.core.event_bus import EventBus
from crypto.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class VolatilityMonitor:
    """Monitors for extreme price moves and manages trading pauses."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._market_data: MarketDataEngine | None = None

        vp = config.get("volatility_protection", {})
        self._enabled = vp.get("enabled", True)
        self._max_move_pct = vp.get("max_price_move_pct", 5.0)
        self._window_minutes = vp.get("detection_window_minutes", 15)
        self._cooldown_minutes = vp.get("cooldown_minutes", 30)

        self._paused_until: datetime | None = None
        logger.info(
            "VolatilityMonitor initialized (max_move=%.1f%%, window=%dm)",
            self._max_move_pct, self._window_minutes,
        )

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        self._market_data = market_data

    @property
    def is_trading_paused(self) -> bool:
        if self._paused_until is None:
            return False
        if datetime.now(timezone.utc) >= self._paused_until:
            logger.info("Volatility cooldown expired — trading resumed")
            self._paused_until = None
            return False
        return True

    def check_conditions(self, symbols: list[str]) -> bool:
        """Check if trading should continue.

        Returns True if conditions are safe, False if trading should pause.
        """
        if not self._enabled or self._market_data is None:
            return True

        if self.is_trading_paused:
            return False

        for symbol in symbols:
            df = self._market_data.get_dataframe(symbol, "5m")
            if df.empty or len(df) < 3:
                continue

            # Check price move over the detection window
            window_bars = max(1, self._window_minutes // 5)
            if len(df) < window_bars + 1:
                continue

            recent_close = df["close"].iloc[-1]
            window_open = df["close"].iloc[-(window_bars + 1)]
            if window_open == 0:
                continue

            move_pct = abs(recent_close - window_open) / window_open * 100

            if move_pct >= self._max_move_pct:
                self._paused_until = datetime.now(timezone.utc) + timedelta(minutes=self._cooldown_minutes)
                logger.warning(
                    "VOLATILITY ALERT: %s moved %.1f%% in %d min — trading paused until %s",
                    symbol, move_pct, self._window_minutes,
                    self._paused_until.strftime("%H:%M:%S UTC"),
                )
                return False

        return True
