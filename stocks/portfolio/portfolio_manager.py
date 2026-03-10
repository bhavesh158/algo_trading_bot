"""Portfolio Manager (PRD §18).

Tracks open positions, available capital, profit and loss, and exposure.
Provides the PortfolioState used by risk management and reporting.
Includes fee/tax accounting, capital ledger, and state persistence.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from stocks.core.enums import OrderSide, PositionStatus
from stocks.core.event_bus import EventBus
from stocks.core.events import PortfolioUpdateEvent
from stocks.core.models import Order, Position, PortfolioState, Trade
from stocks.portfolio.state_manager import StateManager
from stocks.reporting.capital_ledger import CapitalLedger

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages positions, capital, and portfolio state."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self.state_manager = StateManager(config)
        self.ledger = CapitalLedger(config)

        acct = config.get("account", {})
        initial_capital = acct.get("initial_capital", 100_000.0)
        self._profit_tax_rate = acct.get("profit_tax_pct", 15.0) / 100

        self._total_capital = initial_capital
        self._available_capital = initial_capital
        self._peak_capital = initial_capital

        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._trades: list[Trade] = []
        self._daily_pnl = 0.0

        # Attempt to restore from saved state (mid-day restart)
        restored = self._restore_from_disk(initial_capital)
        if not restored:
            self.ledger.log_initial(initial_capital)

        logger.info(
            "PortfolioManager initialized (capital=%.2f, tax_rate=%.0f%%)",
            self._total_capital, self._profit_tax_rate * 100,
        )

    def _restore_from_disk(self, default_capital: float) -> bool:
        """Restore portfolio state from disk if available."""
        saved = self.state_manager.load_state()
        if not saved:
            return False

        capital = saved.get("capital", {})
        if not capital:
            return False

        self._total_capital = capital.get("total", default_capital)
        self._available_capital = capital.get("available", default_capital)
        self._peak_capital = capital.get("peak", default_capital)
        self._daily_pnl = capital.get("daily_pnl", 0.0)

        positions = saved.get("positions", {})
        if positions:
            self._positions = self.state_manager.restore_positions(saved)

        self.ledger.log_restore(
            len(self._positions), self._available_capital, self._total_capital,
        )
        logger.info(
            "Restored state from disk (capital=%.2f, positions=%d)",
            self._total_capital, len(self._positions),
        )
        return True

    def open_position(self, order: Order, entry_commission: float = 0.0) -> Position:
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
            entry_commission=entry_commission,
        )

        notional = order.filled_price * order.filled_quantity
        self._available_capital -= notional + entry_commission
        self._positions[order.symbol] = position

        self.ledger.log_open(
            symbol=order.symbol,
            side=order.side.name,
            notional=notional,
            commission=entry_commission,
            available_after=self._available_capital,
            total_capital=self._total_capital,
        )

        logger.info(
            "Position opened: %s %s %d @ %.2f (capital remaining: %.2f)",
            order.side.name, order.symbol, order.filled_quantity,
            order.filled_price, self._available_capital,
        )

        self._publish_update()
        self._save_state()
        return position

    def close_position(self, symbol: str, exit_price: float, exit_commission: float = 0.0) -> Optional[Trade]:
        """Close an open position and record the completed trade."""
        position = self._positions.get(symbol)
        if not position or position.status != PositionStatus.OPEN:
            logger.warning("No open position for %s", symbol)
            return None

        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.now()
        position.current_price = exit_price

        total_commission = position.entry_commission + exit_commission

        # Calculate gross P&L (before fees/tax)
        multiplier = 1 if position.side == OrderSide.BUY else -1
        gross_pnl = multiplier * (exit_price - position.entry_price) * position.quantity

        # Tax on profit AFTER commission — taxable = gross - costs
        taxable = max(0.0, gross_pnl - total_commission)
        tax = taxable * self._profit_tax_rate

        trade = Trade(
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            strategy_id=position.strategy_id,
            entry_time=position.opened_at,
            exit_time=position.closed_at,
            commission=total_commission,
            tax=tax,
        )
        self._trades.append(trade)

        # Return capital + net P&L
        # entry_commission was already deducted on open, add it back and let
        # trade.pnl handle the full deduction.
        notional = position.entry_price * position.quantity
        self._available_capital += notional + position.entry_commission + trade.pnl
        self._total_capital += trade.pnl
        self._daily_pnl += trade.pnl

        # Track peak for drawdown
        if self._total_capital > self._peak_capital:
            self._peak_capital = self._total_capital

        # Remove from active positions
        del self._positions[symbol]

        self.ledger.log_close(
            symbol=symbol,
            side=trade.side.name,
            notional_returned=notional,
            gross_pnl=trade.gross_pnl,
            commission=total_commission,
            tax=tax,
            net_pnl=trade.pnl,
            available_after=self._available_capital,
            total_capital=self._total_capital,
        )

        logger.info(
            "Position closed: %s %s %d @ %.2f gross=%.2f comm=%.2f tax=%.2f net=%.2f",
            position.side.name, symbol, position.quantity,
            exit_price, trade.gross_pnl, total_commission, tax, trade.pnl,
        )

        self._publish_update()
        self._save_state()
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
        total_exposure = sum(p.notional_value for p in open_positions)
        unrealized = sum(p.unrealized_pnl for p in open_positions)

        # Track peak using equity (cash + position value)
        current_capital = self._available_capital + sum(
            p.current_price * p.quantity for p in open_positions
        )
        self._peak_capital = max(self._peak_capital, current_capital)

        current_drawdown = 0.0
        if self._peak_capital > 0:
            current_drawdown = max(
                0, (self._peak_capital - current_capital) / self._peak_capital * 100
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

    def get_open_position_symbols(self) -> set[str]:
        """Return symbols with open positions."""
        return {s for s, p in self._positions.items() if p.status == PositionStatus.OPEN}

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

    def close_all_positions(self, get_price_fn: Callable[[str], float], exit_commission_fn: Callable[[str, float], float] | None = None) -> int:
        """Close all open positions (used at end of day)."""
        closed = 0
        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            if pos.status == PositionStatus.OPEN:
                price = get_price_fn(symbol)
                if price > 0:
                    comm = exit_commission_fn(symbol, price * pos.quantity) if exit_commission_fn else 0.0
                    self.close_position(symbol, price, comm)
                    closed += 1
        if closed:
            logger.info("Closed %d positions at market close", closed)
        return closed

    def _save_state(self) -> None:
        """Persist current state to disk."""
        self.state_manager.save_state(
            positions=self._positions,
            total_capital=self._total_capital,
            available_capital=self._available_capital,
            peak_capital=self._peak_capital,
            daily_pnl=self._daily_pnl,
        )

    def _publish_update(self) -> None:
        state = self.get_state()
        self.event_bus.publish(PortfolioUpdateEvent(
            total_capital=state.total_capital,
            available_capital=state.available_capital,
            daily_pnl=state.daily_pnl,
            open_positions=state.open_position_count,
        ))
