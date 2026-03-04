"""Event definitions for the trading system event bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from stocks.core.enums import AlertSeverity, MarketRegime, TradingPhase, Timeframe
from stocks.core.models import Candle, Order, Signal, Alert


class Event:
    """Base event class."""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketDataEvent(Event):
    """New market data received (candle or tick)."""
    symbol: str = ""
    candle: Optional[Candle] = None
    price: float = 0.0
    volume: float = 0.0
    timeframe: Timeframe = Timeframe.M1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SignalEvent(Event):
    """A strategy has generated a trading signal."""
    signal: Optional[Signal] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderEvent(Event):
    """An order has been placed, filled, or cancelled."""
    order: Optional[Order] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskEvent(Event):
    """A risk threshold has been breached."""
    rule: str = ""
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    action: str = ""  # e.g., "reduce_size", "pause_trading", "close_positions"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScheduleEvent(Event):
    """A trading phase transition has occurred."""
    phase: TradingPhase = TradingPhase.MARKET_CLOSED
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeChangeEvent(Event):
    """Market regime has changed."""
    previous_regime: MarketRegime = MarketRegime.UNKNOWN
    current_regime: MarketRegime = MarketRegime.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlertEvent(Event):
    """System alert triggered."""
    alert: Optional[Alert] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioUpdateEvent(Event):
    """Portfolio state has changed."""
    total_capital: float = 0.0
    available_capital: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
