"""Capital Ledger — append-only record of every capital movement.

Columns:
    timestamp     — IST ISO timestamp
    event         — INITIAL | RESTORE | POSITION_OPEN | POSITION_CLOSE
    symbol        — trading symbol (or blank for non-trade events)
    side          — BUY/SELL (or blank)
    debit         — capital going out (notional + entry fee on OPEN)
    credit        — capital coming in (notional + net_pnl on CLOSE)
    gross_pnl     — P&L before fees and tax
    commission    — total trading fees (entry + exit)
    tax           — profit tax (only on gross profit after costs)
    net_pnl       — gross_pnl - commission - tax
    balance       — available capital after this event
    total_capital — total capital (available + allocated) after this event
    details       — human-readable note
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEADERS = [
    "timestamp", "event", "symbol", "side",
    "debit", "credit", "gross_pnl", "commission", "tax", "net_pnl",
    "balance", "total_capital", "details",
]


class CapitalLedger:
    """Append-only CSV ledger for capital movements."""

    def __init__(self, config: dict[str, Any]) -> None:
        data_dir = config.get("system", {}).get("data_dir", "./data_store")
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "capital_ledger.csv"
        self._init_csv()
        logger.info("CapitalLedger initialized (%s)", self._path)

    def _init_csv(self) -> None:
        if not self._path.exists():
            with open(self._path, "w", newline="") as f:
                csv.writer(f).writerow(_HEADERS)

    def log_initial(self, capital: float) -> None:
        """Log the starting capital."""
        self._append("INITIAL", "", "",
                      debit=0, credit=capital,
                      gross_pnl=0, commission=0, tax=0, net_pnl=0,
                      balance=capital, total_capital=capital,
                      details=f"Starting capital: {capital:.2f}")

    def log_restore(self, positions_count: int, available: float, total: float) -> None:
        """Log state restoration from crash recovery."""
        self._append("RESTORE", "", "",
                      debit=0, credit=0,
                      gross_pnl=0, commission=0, tax=0, net_pnl=0,
                      balance=available, total_capital=total,
                      details=f"Restored {positions_count} positions from saved state")

    def log_open(
        self, symbol: str, side: str, notional: float,
        commission: float,
        available_after: float, total_capital: float,
    ) -> None:
        """Log capital allocated to a new position (debit = notional + entry fee)."""
        total_debit = notional + commission
        details = f"Allocated {notional:.2f} to {side} {symbol}"
        if commission > 0:
            details += f", entry_fee={commission:.2f}"
        self._append("POSITION_OPEN", symbol, side,
                      debit=total_debit, credit=0,
                      gross_pnl=0, commission=commission, tax=0, net_pnl=0,
                      balance=available_after, total_capital=total_capital,
                      details=details)

    def log_close(
        self, symbol: str, side: str,
        notional_returned: float,
        gross_pnl: float, commission: float, tax: float, net_pnl: float,
        available_after: float, total_capital: float,
    ) -> None:
        """Log capital returned from a closed position."""
        credit = notional_returned + net_pnl
        details = (
            f"Closed {side} {symbol}: gross={gross_pnl:+.2f},"
            f" comm={commission:.2f}, tax={tax:.2f}, net={net_pnl:+.2f}"
        )
        self._append("POSITION_CLOSE", symbol, side,
                      debit=0, credit=credit,
                      gross_pnl=gross_pnl, commission=commission,
                      tax=tax, net_pnl=net_pnl,
                      balance=available_after, total_capital=total_capital,
                      details=details)

    def _append(
        self, event: str, symbol: str, side: str,
        debit: float, credit: float,
        gross_pnl: float, commission: float, tax: float, net_pnl: float,
        balance: float, total_capital: float, details: str,
    ) -> None:
        row = [
            datetime.now().isoformat(),
            event,
            symbol,
            side,
            f"{debit:.2f}" if debit else "",
            f"{credit:.2f}" if credit else "",
            f"{gross_pnl:+.2f}" if gross_pnl else "",
            f"{commission:.2f}" if commission else "",
            f"{tax:.2f}" if tax else "",
            f"{net_pnl:+.2f}" if net_pnl else "",
            f"{balance:.2f}",
            f"{total_capital:.2f}",
            details,
        ]
        try:
            with open(self._path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception:
            logger.exception("Failed to write to capital ledger")
