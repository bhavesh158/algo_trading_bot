"""Abstract base class for all crypto trading strategies.

Every strategy must subclass BaseStrategy and implement `analyze()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from crypto.core.enums import OrderSide, SignalStrength, Timeframe
from crypto.core.models import Signal
from crypto.data.market_data_engine import MarketDataEngine


class BaseStrategy(ABC):
    """Interface for a pluggable trading strategy."""

    def __init__(self, strategy_id: str, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        self.strategy_id = strategy_id
        self.config = config
        self.market_data = market_data
        self.enabled: bool = True
        self.primary_timeframe: str = "5m"

    @abstractmethod
    def analyze(self, symbol: str) -> Optional[Signal]:
        """Analyze a symbol and return a Signal if a trade opportunity exists."""
        ...

    @abstractmethod
    def should_exit(
        self, symbol: str, entry_price: float, current_price: float,
        side: OrderSide = OrderSide.BUY,
    ) -> bool:
        """Determine whether an open position should be exited."""
        ...

    def get_exit_signal(
        self, symbol: str, entry_price: float, current_price: float,
        side: OrderSide = OrderSide.BUY,
    ) -> Optional[Signal]:
        if self.should_exit(symbol, entry_price, current_price, side=side):
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                entry_price=current_price,
            )
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.strategy_id}, enabled={self.enabled})"
