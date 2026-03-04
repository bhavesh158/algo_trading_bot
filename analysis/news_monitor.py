"""News & Event Awareness + Volatility Safeguards (PRD §15, §16).

Pauses trading during major economic events and detects abnormal
market volatility (sudden spikes or crashes).
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Any, Optional

import numpy as np

from core.enums import AlertSeverity, Timeframe
from core.event_bus import EventBus
from core.events import AlertEvent
from core.models import Alert
from data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)

# Known recurring events (time in IST) — extensible via config or external API
_DEFAULT_EVENT_SCHEDULE: list[dict] = [
    {"name": "RBI Policy", "time": "10:00", "day_of_week": None},
    # Add more events here or load from config/API
]


class NewsMonitor:
    """Monitors for news events and abnormal volatility."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._market_data: Optional[MarketDataEngine] = None

        news_config = config.get("news", {})
        self._news_enabled = news_config.get("enabled", False)
        self._pause_before = news_config.get("pause_minutes_before_event", 15)
        self._pause_after = news_config.get("pause_minutes_after_event", 15)

        vol_config = config.get("volatility_safeguards", {})
        self._vol_enabled = vol_config.get("enabled", True)
        self._max_intraday_move = vol_config.get("max_intraday_move_pct", 3.0)
        self._spike_window = vol_config.get("spike_detection_window", 5)
        self._spike_threshold = vol_config.get("spike_threshold_pct", 1.5)

        self._trading_paused = False
        self._pause_reason: Optional[str] = None
        self._index_symbol = "^NSEI"

        logger.info(
            "NewsMonitor initialized (news=%s, vol_safeguards=%s)",
            self._news_enabled, self._vol_enabled,
        )

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        self._market_data = market_data

    @property
    def is_trading_paused(self) -> bool:
        return self._trading_paused

    @property
    def pause_reason(self) -> Optional[str]:
        return self._pause_reason

    def check_conditions(self) -> bool:
        """Check all conditions. Returns True if trading should continue, False if paused."""
        # Check news events
        if self._news_enabled and self._is_near_event():
            self._pause_trading("Scheduled economic event nearby")
            return False

        # Check volatility safeguards
        if self._vol_enabled and self._market_data is not None:
            if self._detect_abnormal_volatility():
                self._pause_trading("Abnormal market volatility detected")
                return False

        # All clear — resume if previously paused
        if self._trading_paused:
            logger.info("Trading resumed — conditions normalized")
            self._trading_paused = False
            self._pause_reason = None

        return True

    def _is_near_event(self) -> bool:
        """Check if current time is near a scheduled event."""
        now = datetime.now()
        for event in _DEFAULT_EVENT_SCHEDULE:
            event_time_str = event.get("time", "")
            try:
                event_time = datetime.combine(
                    now.date(),
                    datetime.strptime(event_time_str, "%H:%M").time(),
                )
                window_start = event_time - timedelta(minutes=self._pause_before)
                window_end = event_time + timedelta(minutes=self._pause_after)
                if window_start <= now <= window_end:
                    return True
            except (ValueError, TypeError):
                continue
        return False

    def _detect_abnormal_volatility(self) -> bool:
        """Detect if market is experiencing abnormal volatility."""
        df = self._market_data.get_dataframe(self._index_symbol, Timeframe.M5)
        if df is None or len(df) < self._spike_window + 1:
            return False

        close = df["close"]
        current = float(close.iloc[-1])

        # Check intraday move from day open
        if hasattr(df.index, 'date'):
            today = datetime.now().date()
            today_data = df[df.index.date == today]
            if not today_data.empty:
                day_open = float(today_data["open"].iloc[0])
                intraday_move = abs(current - day_open) / day_open * 100
                if intraday_move > self._max_intraday_move:
                    logger.warning(
                        "Intraday move %.2f%% exceeds threshold %.1f%%",
                        intraday_move, self._max_intraday_move,
                    )
                    return True

        # Check for spike (rapid move in short window)
        window_data = close.iloc[-self._spike_window:]
        window_min = float(window_data.min())
        window_max = float(window_data.max())
        if window_min > 0:
            spike_pct = (window_max - window_min) / window_min * 100
            if spike_pct > self._spike_threshold:
                logger.warning(
                    "Volatility spike %.2f%% in %d-bar window exceeds %.1f%%",
                    spike_pct, self._spike_window, self._spike_threshold,
                )
                return True

        return False

    def _pause_trading(self, reason: str) -> None:
        """Pause trading with a reason."""
        if not self._trading_paused:
            logger.warning("Trading PAUSED: %s", reason)
            self._trading_paused = True
            self._pause_reason = reason
            self.event_bus.publish(AlertEvent(alert=Alert(
                severity=AlertSeverity.WARNING,
                source="NewsMonitor",
                message=f"Trading paused: {reason}",
            )))
