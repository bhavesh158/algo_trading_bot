"""Capital Ledger — append-only record of every capital movement.

Every debit (capital allocated to a position) and credit (capital returned
on close, including P&L) is logged here. The ledger can always be replayed
to reconcile the current balance.

Columns:
    timestamp   — UTC ISO timestamp
    event       — INITIAL | POSITION_OPEN | POSITION_CLOSE | COMMISSION | ADJUSTMENT
    symbol      — trading pair (or blank for non-trade events)
    side        — BUY/SELL (or blank)
    debit       — capital going out (positive number)
    credit      — capital coming in (positive number)
    pnl         — realized P&L on this event (blank for opens)
    balance     — available capital after this event
    total_capital — total capital (available + allocated) after this event
    details     — human-readable note
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEADERS = [
    "timestamp", "event", "symbol", "side",
    "debit", "credit", "pnl",
    "balance", "total_capital", "details",
]


class CapitalLedger:
    """Append-only CSV ledger for capital movements."""

    def __init__(self, config: dict[str, Any]) -> None:
        data_dir = config.get("system", {}).get("data_dir", "./crypto_data")
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
        self._append("INITIAL", "", "", 0, capital, 0, capital, capital,
                      f"Starting capital: {capital:.2f}")

    def log_restore(self, positions_count: int, available: float, total: float) -> None:
        """Log state restoration from crash recovery."""
        self._append("RESTORE", "", "", 0, 0, 0, available, total,
                      f"Restored {positions_count} positions from saved state")

    def log_open(
        self, symbol: str, side: str, notional: float,
        available_after: float, total_capital: float,
    ) -> None:
        """Log capital allocated to a new position."""
        self._append("POSITION_OPEN", symbol, side,
                      notional, 0, 0,
                      available_after, total_capital,
                      f"Allocated {notional:.4f} to {side} {symbol}")

    def log_close(
        self, symbol: str, side: str,
        notional_returned: float, pnl: float, commission: float,
        available_after: float, total_capital: float,
    ) -> None:
        """Log capital returned from a closed position."""
        credit = notional_returned + pnl
        details = f"Closed {side} {symbol}: returned {notional_returned:.4f}, pnl={pnl:+.4f}"
        if commission > 0:
            details += f", commission={commission:.4f}"
        self._append("POSITION_CLOSE", symbol, side,
                      0, credit, pnl,
                      available_after, total_capital, details)

    def _append(
        self, event: str, symbol: str, side: str,
        debit: float, credit: float, pnl: float,
        balance: float, total_capital: float, details: str,
    ) -> None:
        row = [
            datetime.now(timezone.utc).isoformat(),
            event,
            symbol,
            side,
            f"{debit:.4f}" if debit else "",
            f"{credit:.4f}" if credit else "",
            f"{pnl:+.4f}" if pnl else "",
            f"{balance:.4f}",
            f"{total_capital:.4f}",
            details,
        ]
        try:
            with open(self._path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception:
            logger.exception("Failed to write to capital ledger")
