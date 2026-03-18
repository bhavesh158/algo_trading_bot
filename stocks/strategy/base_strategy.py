"""Abstract base class for all trading strategies.

Every strategy must subclass BaseStrategy and implement `analyze()`.
Strategies are pluggable — add new ones by creating a subclass and
registering it in the strategy engine config.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from stocks.core.enums import Timeframe
from stocks.core.models import Candle, Position, Signal
from stocks.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Interface for a pluggable trading strategy."""

    def __init__(self, strategy_id: str, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        self.strategy_id = strategy_id
        self.config = config
        self.market_data = market_data
        self.enabled: bool = True
        self.primary_timeframe: Timeframe = Timeframe.M5

        # Trailing stop (subclasses set > 0 to enable)
        self._trailing_stop_atr: float = 0.0
        # Percentage-based trailing stop (subclasses set > 0 to enable)
        self._trailing_stop_pct: float = 0.0
        # Max hold duration (subclasses set > 0 to enable)
        self._max_hold_minutes: int = 0

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

    def check_stop_loss(self, position: Position) -> bool:
        """Check if the hard stop-loss price has been breached.

        Uses the stop_loss price stored on the Position at entry time.
        This fires for all strategies — stop-losses should always be respected.
        """
        if position.stop_loss <= 0:
            return False
        from stocks.core.enums import OrderSide
        if position.side == OrderSide.BUY:
            return position.current_price <= position.stop_loss
        else:
            return position.current_price >= position.stop_loss

    def check_trailing_stop_pct(self, position: Position) -> bool:
        """Percentage-based trailing stop: exit when price drops `_trailing_stop_pct`%
        from the peak (or rises that % from the trough for short positions).

        Requires `_trailing_stop_pct > 0`. A position must already have moved in
        our favour (highest_since_entry > entry for BUY, etc.) for this to fire.
        """
        if self._trailing_stop_pct <= 0:
            return False
        from stocks.core.enums import OrderSide
        trail = self._trailing_stop_pct / 100
        if position.side == OrderSide.BUY:
            if position.highest_since_entry <= 0:
                return False
            trail_stop = position.highest_since_entry * (1 - trail)
            return (
                position.current_price <= trail_stop
                and position.current_price < position.highest_since_entry
            )
        else:
            if position.lowest_since_entry <= 0:
                return False
            trail_stop = position.lowest_since_entry * (1 + trail)
            return (
                position.current_price >= trail_stop
                and position.current_price > position.lowest_since_entry
            )

    def check_trailing_stop(self, symbol: str, position: Position) -> bool:
        """Check if the trailing stop has been hit.

        Uses ATR-based trailing stop: for BUY positions, stop trails at
        (highest_since_entry - trailing_atr * ATR).
        """
        if self._trailing_stop_atr <= 0:
            return False

        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return False

        atr_value = float(atr.iloc[-1])
        from stocks.core.enums import OrderSide

        if position.side == OrderSide.BUY:
            trailing_stop = position.highest_since_entry - self._trailing_stop_atr * atr_value
            if position.current_price <= trailing_stop and position.current_price < position.highest_since_entry:
                logger.info(
                    "[%s] Trailing stop hit: %s price=%.2f < trail=%.2f (peak=%.2f)",
                    self.strategy_id, symbol, position.current_price,
                    trailing_stop, position.highest_since_entry,
                )
                return True
        else:  # SELL
            trailing_stop = position.lowest_since_entry + self._trailing_stop_atr * atr_value
            if position.current_price >= trailing_stop and position.current_price > position.lowest_since_entry:
                logger.info(
                    "[%s] Trailing stop hit: %s price=%.2f > trail=%.2f (trough=%.2f)",
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
        position: Optional[Position] = None,
    ) -> Optional[Signal]:
        """Generate an exit signal if exit conditions are met.

        Checks (in priority order):
        1. Hard stop-loss
        2. Strategy-specific exit (should_exit)
        3. ATR-based trailing stop
        4. Percentage-based trailing stop
        5. Time-based exit
        """
        should_close = False
        exit_reason = "strategy"

        # 1. Hard stop-loss — highest priority, always fires
        if position is not None and self.check_stop_loss(position):
            should_close = True
            exit_reason = "stop_loss"

        # 2. Strategy-specific exit (e.g. VWAP touch, target hit)
        if not should_close:
            should_close = self.should_exit(symbol, entry_price, current_price)

        # 3. ATR-based trailing stop
        if not should_close and position is not None:
            if self.check_trailing_stop(symbol, position):
                should_close = True
                exit_reason = "trailing_stop"

        # 4. Percentage-based trailing stop
        if not should_close and position is not None:
            if self.check_trailing_stop_pct(position):
                should_close = True
                exit_reason = "trailing_stop"

        # 5. Time-based exit
        if not should_close and position is not None:
            if self.check_time_exit(position):
                should_close = True
                exit_reason = "time_exit"

        if should_close:
            from stocks.core.enums import OrderSide, SignalStrength
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
