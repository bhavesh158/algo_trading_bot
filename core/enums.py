"""Shared enumerations for the trading system."""

from enum import Enum, auto


class TradingMode(Enum):
    """Operating mode of the trading system."""
    PAPER = auto()
    LIVE = auto()


class OrderSide(Enum):
    """Direction of a trade."""
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    """Type of order to place."""
    MARKET = auto()
    LIMIT = auto()
    STOP_LOSS = auto()
    STOP_LOSS_LIMIT = auto()


class OrderStatus(Enum):
    """Lifecycle status of an order."""
    PENDING = auto()
    SUBMITTED = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class PositionStatus(Enum):
    """Status of a position."""
    OPEN = auto()
    CLOSED = auto()


class MarketRegime(Enum):
    """Detected market condition."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    SIDEWAYS = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    UNKNOWN = auto()


class TradingPhase(Enum):
    """Intraday trading session phases."""
    PRE_MARKET = auto()
    MARKET_OPEN = auto()
    MARKET_HOURS = auto()
    PRE_CLOSE = auto()
    MARKET_CLOSED = auto()


class Timeframe(Enum):
    """Candle timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"


class SignalStrength(Enum):
    """Confidence level of a trading signal."""
    WEAK = auto()
    MODERATE = auto()
    STRONG = auto()


class AlertSeverity(Enum):
    """Severity level for system alerts."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
