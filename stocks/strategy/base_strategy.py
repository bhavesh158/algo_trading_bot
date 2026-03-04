"""Abstract base class for all trading strategies.

Every strategy must subclass BaseStrategy and implement `analyze()`.
Strategies are pluggable — add new ones by creating a subclass and
registering it in the strategy engine config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from stocks.core.enums import Timeframe
from stocks.core.models import Candle, Signal
from stocks.data.market_data_engine import MarketDataEngine


class BaseStrategy(ABC):
    """Interface for a pluggable trading strategy."""

    def __init__(self, strategy_id: str, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        self.strategy_id = strategy_id
        self.config = config
        self.market_data = market_data
        self.enabled: bool = True
        self.primary_timeframe: Timeframe = Timeframe.M5

    @abstractmethod
    def analyze(self, symbol: str) -> Optional[Signal]:
        """Analyze a symbol and return a Signal if a trade opportunity exists.

        Returns None if no valid signal is found.
        """
        ...

    @abstractmethod
    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        """Determine whether an open position should be exited."""
        ...

    def get_exit_signal(self, symbol: str, entry_price: float, current_price: float) -> Optional[Signal]:
        """Generate an exit signal if exit conditions are met.

        Override for custom exit logic. Default returns a SELL signal if should_exit() is True.
        """
        if self.should_exit(symbol, entry_price, current_price):
            from stocks.core.enums import OrderSide, SignalStrength
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
