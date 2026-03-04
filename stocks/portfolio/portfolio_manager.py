"""Portfolio Manager (PRD §18).

Tracks open positions, available capital, profit and loss, and exposure.
Provides the PortfolioState used by risk management and reporting.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from stocks.core.enums import OrderSide, PositionStatus
from stocks.core.event_bus import EventBus
from stocks.core.events import PortfolioUpdateEvent
from stocks.core.models import Order, Position, PortfolioState, Trade

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages positions, capital, and portfolio state."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus

        initial_capital = config.get("account", {}).get("initial_capital", 100_000.0)
        self._total_capital = initial_capital
        self._available_capital = initial_capital
        self._peak_capital = initial_capital

        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._trades: list[Trade] = []
        self._daily_pnl = 0.0

        logger.info(
            "PortfolioManager initialized (capital=%.2f)", initial_capital
        )

    def open_position(self, order: Order) -> Position:
        """Create a new position from a filled order."""
        position = Position(
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            entry_price=order.filled_price,
            current_price=order.filled_price,
            stop_loss=order.stop_price,
            strategy_id=order.strategy_id,
            entry_order_id=order.id,
        )

        self._positions[order.symbol] = position
        value = order.filled_price * order.filled_quantity
        self._available_capital -= value

        logger.info(
            "Position opened: %s %s %d @ %.2f (capital remaining: %.2f)",
            order.side.name, order.symbol, order.filled_quantity,
            order.filled_price, self._available_capital,
        )

        self._publish_update()
        return position

    def close_position(self, symbol: str, exit_price: float, commission: float = 0.0) -> Trade | None:
        """Close an open position and record the completed trade."""
        position = self._positions.get(symbol)
        if not position or position.status != PositionStatus.OPEN:
            logger.warning("No open position for %s", symbol)
            return None

        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.now()
        position.current_price = exit_price

        # Record trade
        trade = Trade(
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            strategy_id=position.strategy_id,
            entry_time=position.opened_at,
            exit_time=datetime.now(),
            commission=commission,
        )
        self._trades.append(trade)

        # Restore capital
        entry_value = position.entry_price * position.quantity
        exit_value = exit_price * position.quantity
        self._available_capital += exit_value
        self._daily_pnl += trade.pnl

        # Update total capital (realized)
        self._total_capital += trade.pnl

        # Track peak for drawdown
        if self._total_capital > self._peak_capital:
            self._peak_capital = self._total_capital

        # Remove from active positions
        del self._positions[symbol]

        logger.info(
            "Position closed: %s %s %d @ %.2f PnL=%.2f (capital: %.2f)",
            position.side.name, symbol, position.quantity,
            exit_price, trade.pnl, self._total_capital,
        )

        self._publish_update()
        return trade

    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update the current price of an open position."""
        position = self._positions.get(symbol)
        if position and position.status == PositionStatus.OPEN:
            position.current_price = current_price

    def get_state(self) -> PortfolioState:
        """Get a snapshot of the current portfolio state."""
        open_positions = [
            p for p in self._positions.values() if p.status == PositionStatus.OPEN
        ]
        total_exposure = sum(
            abs(p.entry_price * p.quantity) for p in open_positions
        )
        unrealized = sum(p.unrealized_pnl for p in open_positions)

        current_drawdown = 0.0
        if self._peak_capital > 0:
            effective_capital = self._total_capital + unrealized
            current_drawdown = max(
                0, (self._peak_capital - effective_capital) / self._peak_capital * 100
            )

        return PortfolioState(
            total_capital=self._total_capital,
            available_capital=self._available_capital,
            positions=open_positions,
            daily_pnl=self._daily_pnl + unrealized,
            daily_pnl_pct=(self._daily_pnl + unrealized) / self._total_capital * 100
            if self._total_capital > 0 else 0,
            total_exposure=total_exposure,
            current_drawdown=current_drawdown,
            peak_capital=self._peak_capital,
        )

    def get_position(self, symbol: str) -> Position | None:
        pos = self._positions.get(symbol)
        return pos if pos and pos.status == PositionStatus.OPEN else None

    def get_open_positions(self) -> dict[str, Position]:
        return {
            s: p for s, p in self._positions.items() if p.status == PositionStatus.OPEN
        }

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    @property
    def daily_trades(self) -> list[Trade]:
        today = datetime.now().date()
        return [t for t in self._trades if t.entry_time.date() == today]

    def reset_daily_state(self) -> None:
        """Reset daily P&L counters (called at start of each day)."""
        self._daily_pnl = 0.0
        logger.info("Daily portfolio state reset")

    def close_all_positions(self, get_price_fn) -> int:
        """Close all open positions (used at end of day)."""
        closed = 0
        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            if pos.status == PositionStatus.OPEN:
                price = get_price_fn(symbol)
                if price > 0:
                    self.close_position(symbol, price)
                    closed += 1
        if closed:
            logger.info("Closed %d positions at market close", closed)
        return closed

    def _publish_update(self) -> None:
        state = self.get_state()
        self.event_bus.publish(PortfolioUpdateEvent(
            total_capital=state.total_capital,
            available_capital=state.available_capital,
            daily_pnl=state.daily_pnl,
            open_positions=state.open_position_count,
        ))
