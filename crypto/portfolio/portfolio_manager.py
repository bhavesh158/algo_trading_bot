"""Portfolio Manager (PRD §14).

Tracks open positions, capital, exposure, and P&L.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from crypto.core.enums import OrderSide, PositionStatus
from crypto.core.event_bus import EventBus
from crypto.core.events import PortfolioUpdateEvent
from crypto.core.models import Order, Position, PortfolioState, Trade
from crypto.portfolio.state_manager import StateManager
from crypto.reporting.capital_ledger import CapitalLedger

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Tracks positions, capital, and portfolio state."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self.state_manager = StateManager(config)
        self.ledger = CapitalLedger(config)

        acct = config.get("account", {})
        initial = acct.get("initial_capital", 1000.0)
        self._profit_tax_rate = acct.get("profit_tax_pct", 30.0) / 100  # e.g. 30%
        self._total_capital = initial
        self._available_capital = initial
        self._peak_capital = initial

        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._closed_trades: list[Trade] = []
        self._rolling_pnl = 0.0
        # Track recent P&L entries for rolling window
        self._pnl_entries: deque = deque(maxlen=10000)

        # Attempt to restore from saved state
        restored = self._restore_from_disk(initial)
        if not restored:
            self.ledger.log_initial(initial)

        logger.info("PortfolioManager initialized (capital=%.2f)", self._total_capital)

    def _restore_from_disk(self, default_capital: float) -> bool:
        """Restore portfolio state from disk if available. Returns True if restored."""
        saved = self.state_manager.load_state()
        if not saved:
            return False

        # Always restore capital if we have a valid state file — even with
        # no open positions, the capital reflects accumulated P&L from
        # previous trades and must not be reset to initial_capital.
        capital = saved.get("capital", {})
        if not capital:
            return False

        self._total_capital = capital.get("total", default_capital)
        self._available_capital = capital.get("available", default_capital)
        self._peak_capital = capital.get("peak", default_capital)
        self._rolling_pnl = capital.get("rolling_pnl", 0.0)

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

    def get_state(self) -> PortfolioState:
        """Get current portfolio snapshot."""
        open_positions = [p for p in self._positions.values() if p.status == PositionStatus.OPEN]
        total_exposure = sum(p.notional_value for p in open_positions)

        # Total equity = cash + current value of all open positions
        current_capital = self._available_capital + sum(
            p.current_price * p.quantity for p in open_positions
        )
        self._peak_capital = max(self._peak_capital, current_capital)
        current_dd = 0.0
        if self._peak_capital > 0:
            current_dd = (self._peak_capital - current_capital) / self._peak_capital * 100

        return PortfolioState(
            total_capital=self._total_capital,
            available_capital=self._available_capital,
            positions=list(self._positions.values()),
            rolling_pnl=self._rolling_pnl,
            rolling_pnl_pct=(self._rolling_pnl / self._total_capital * 100) if self._total_capital > 0 else 0,
            total_exposure=total_exposure,
            current_drawdown=current_dd,
            peak_capital=self._peak_capital,
        )

    def open_position(self, order: Order, entry_commission: float = 0.0) -> Position:
        """Create a new position from a filled order."""
        position = Position(
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            entry_price=order.filled_price,
            current_price=order.filled_price,
            stop_loss=order.stop_price,
            target_price=order.target_price,
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
            "Position opened: %s %s qty=%.6f @ %.4f",
            order.side.name, order.symbol, order.filled_quantity, order.filled_price,
        )
        self._publish_update()
        self._save_state()
        return position

    def close_position(self, symbol: str, exit_price: float, exit_commission: float = 0.0) -> Optional[Trade]:
        """Close an open position and record the trade."""
        pos = self._positions.get(symbol)
        if not pos or pos.status != PositionStatus.OPEN:
            return None

        pos.status = PositionStatus.CLOSED
        pos.closed_at = datetime.now(timezone.utc)
        pos.current_price = exit_price

        total_commission = pos.entry_commission + exit_commission

        # Calculate gross P&L (before fees/tax)
        multiplier = 1 if pos.side == OrderSide.BUY else -1
        gross_pnl = multiplier * (exit_price - pos.entry_price) * pos.quantity

        # Tax on gross profit only (before commission), applied only if profitable
        tax = max(0.0, gross_pnl) * self._profit_tax_rate

        trade = Trade(
            symbol=symbol,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            strategy_id=pos.strategy_id,
            entry_time=pos.opened_at,
            exit_time=pos.closed_at,
            commission=total_commission,
            tax=tax,
        )

        # Return capital + net P&L (gross - commission - tax)
        # Note: entry_commission was already deducted on open, so we add it
        # back into notional return, and let trade.pnl handle the full deduction.
        notional = pos.entry_price * pos.quantity
        self._available_capital += notional + pos.entry_commission + trade.pnl
        self._total_capital += trade.pnl
        self._rolling_pnl += trade.pnl
        self._pnl_entries.append((datetime.now(timezone.utc), trade.pnl))

        self._closed_trades.append(trade)
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
            "Position closed: %s %s gross=%.4f comm=%.4f tax=%.4f net=%.4f (%.2f%%)",
            symbol, trade.side.name, trade.gross_pnl, total_commission,
            tax, trade.pnl, trade.pnl_pct,
        )
        self._publish_update()
        self._save_state()
        return trade

    def update_position_price(self, symbol: str, price: float) -> None:
        pos = self._positions.get(symbol)
        if pos and pos.status == PositionStatus.OPEN:
            pos.current_price = price

    def get_open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._positions.items() if p.status == PositionStatus.OPEN}

    def close_all_positions(self, price_fn: Callable[[str], float]) -> None:
        """Close all open positions using provided price function."""
        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            if pos.status == PositionStatus.OPEN:
                price = price_fn(symbol)
                if price > 0:
                    self.close_position(symbol, price)

    def reset_rolling_pnl(self) -> None:
        self._rolling_pnl = 0.0
        self._pnl_entries.clear()
        logger.info("Rolling P&L reset")

    def _save_state(self, entered_symbols: set[str] | None = None) -> None:
        """Persist current state to disk."""
        self.state_manager.save_state(
            positions=self._positions,
            total_capital=self._total_capital,
            available_capital=self._available_capital,
            peak_capital=self._peak_capital,
            rolling_pnl=self._rolling_pnl,
            entered_symbols=entered_symbols,
        )

    def get_open_position_symbols(self) -> set[str]:
        """Return symbols with open positions (used by pair selector)."""
        return {s for s, p in self._positions.items() if p.status == PositionStatus.OPEN}

    def _publish_update(self) -> None:
        state = self.get_state()
        self.event_bus.publish(PortfolioUpdateEvent(
            total_capital=state.total_capital,
            available_capital=state.available_capital,
            rolling_pnl=state.rolling_pnl,
            open_positions=state.open_position_count,
        ))
