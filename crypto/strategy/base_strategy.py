"""Abstract base class for all crypto trading strategies.

Every strategy must subclass BaseStrategy and implement `analyze()`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from crypto.core.enums import OrderSide, SignalStrength, Timeframe
from crypto.core.models import Position, Signal
from crypto.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Interface for a pluggable trading strategy."""

    def __init__(self, strategy_id: str, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        self.strategy_id = strategy_id
        self.config = config
        self.market_data = market_data
        self.enabled: bool = True
        self.primary_timeframe: str = "5m"

        # Trailing stop (subclasses set > 0 to enable)
        self._trailing_stop_atr: float = 0.0
        # Max hold duration (subclasses set > 0 to enable)
        self._max_hold_minutes: int = 0

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

    def check_trailing_stop(self, symbol: str, position: Position) -> bool:
        """Check if the trailing stop has been hit.

        Uses ATR-based trailing stop: for BUY positions, stop trails at
        (highest_since_entry - trailing_atr * ATR).
        """
        if self._trailing_stop_atr <= 0:
            return False

        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty:
            return False

        atr = df.get("atr", pd.Series(dtype=float))
        if atr.empty or pd.isna(atr.iloc[-1]):
            return False

        atr_value = float(atr.iloc[-1])

        if position.side == OrderSide.BUY:
            trailing_stop = position.highest_since_entry - self._trailing_stop_atr * atr_value
            if position.current_price <= trailing_stop and position.current_price < position.highest_since_entry:
                logger.info(
                    "[%s] Trailing stop hit: %s price=%.4f < trail=%.4f (peak=%.4f)",
                    self.strategy_id, symbol, position.current_price,
                    trailing_stop, position.highest_since_entry,
                )
                return True
        else:  # SELL
            trailing_stop = position.lowest_since_entry + self._trailing_stop_atr * atr_value
            if position.current_price >= trailing_stop and position.current_price > position.lowest_since_entry:
                logger.info(
                    "[%s] Trailing stop hit: %s price=%.4f > trail=%.4f (trough=%.4f)",
                    self.strategy_id, symbol, position.current_price,
                    trailing_stop, position.lowest_since_entry,
                )
                return True

        return False

    def check_time_exit(self, position: Position) -> bool:
        """Check if max hold duration has been exceeded."""
        if self._max_hold_minutes <= 0:
            return False
        if position.hold_duration_minutes >= self._max_hold_minutes:
            logger.info(
                "[%s] Time exit: %s held %.0f min (max=%d)",
                self.strategy_id, position.symbol,
                position.hold_duration_minutes, self._max_hold_minutes,
            )
            return True
        return False

    def get_exit_signal(
        self, symbol: str, entry_price: float, current_price: float,
        side: OrderSide = OrderSide.BUY,
        position: Optional[Position] = None,
    ) -> Optional[Signal]:
        """Generate an exit signal if exit conditions are met.

        Checks: strategy-specific exit, trailing stop, and time-based exit.
        """
        should_close = self.should_exit(symbol, entry_price, current_price, side=side)
        exit_reason = "strategy"

        # Check trailing stop
        if not should_close and position is not None:
            if self.check_trailing_stop(symbol, position):
                should_close = True
                exit_reason = "trailing_stop"

        # Check time-based exit
        if not should_close and position is not None:
            if self.check_time_exit(position):
                should_close = True
                exit_reason = "time_exit"

        if should_close:
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                entry_price=current_price,
                metadata={"exit_reason": exit_reason},
            )
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.strategy_id}, enabled={self.enabled})"
