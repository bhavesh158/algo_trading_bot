"""Event definitions for the crypto trading system event bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from crypto.core.enums import AlertSeverity, MarketRegime, Timeframe
from crypto.core.models import Candle, Order, Signal, Alert


class Event:
    """Base event class."""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketDataEvent(Event):
    """New market data received."""
    symbol: str = ""
    candle: Optional[Candle] = None
    price: float = 0.0
    volume: float = 0.0
    timeframe: Timeframe = Timeframe.M5
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SignalEvent(Event):
    """A strategy has generated a trading signal."""
    signal: Optional[Signal] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderEvent(Event):
    """An order has been placed, filled, or cancelled."""
    order: Optional[Order] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskEvent(Event):
    """A risk threshold has been breached."""
    rule: str = ""
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    action: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegimeChangeEvent(Event):
    """Market regime has changed."""
    previous_regime: MarketRegime = MarketRegime.UNKNOWN
    current_regime: MarketRegime = MarketRegime.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertEvent(Event):
    """System alert triggered."""
    alert: Optional[Alert] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioUpdateEvent(Event):
    """Portfolio state has changed."""
    total_capital: float = 0.0
    available_capital: float = 0.0
    rolling_pnl: float = 0.0
    open_positions: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExchangeConnectionEvent(Event):
    """Exchange connection status changed."""
    exchange: str = ""
    connected: bool = False
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
