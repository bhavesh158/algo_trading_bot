"""Trade Journal — logs all position opens and closes to a CSV file.

Provides a persistent, human-readable record of every trade for analysis.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from stocks.core.models import Order, Position, Trade

logger = logging.getLogger(__name__)


class TradeJournal:
    """Appends trade events to a CSV file."""

    def __init__(self, config: dict[str, Any]) -> None:
        data_dir = config.get("system", {}).get("data_dir", "./data_store")
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._path = self._dir / "trade_journal.csv"
        self._init_csv()
        logger.info("TradeJournal initialized (%s)", self._path)

    def _init_csv(self) -> None:
        """Create CSV with headers if it doesn't exist."""
        if not self._path.exists():
            with open(self._path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "event", "symbol", "side", "quantity",
                    "entry_price", "exit_price", "stop_loss", "target_price",
                    "pnl", "pnl_pct", "commission", "strategy_id", "hold_duration",
                    "gross_pnl", "tax",
                ])

    def log_open(self, order: Order, position: Position) -> None:
        """Log a position open event."""
        row = [
            datetime.now().isoformat(),
            "OPEN",
            position.symbol,
            position.side.name,
            str(position.quantity),
            f"{position.entry_price:.2f}",
            "",  # exit_price
            f"{position.stop_loss:.2f}" if position.stop_loss else "",
            f"{position.target_price:.2f}" if position.target_price else "",
            "",  # pnl
            "",  # pnl_pct
            "",  # commission
            position.strategy_id,
            "",  # hold_duration
            "",  # gross_pnl
            "",  # tax
        ]
        self._append_row(row)

    def log_close(self, trade: Trade) -> None:
        """Log a position close event."""
        hold = ""
        if trade.exit_time and trade.entry_time:
            delta = trade.exit_time - trade.entry_time
            minutes = delta.total_seconds() / 60
            if minutes >= 60:
                hold = f"{minutes / 60:.1f}h"
            else:
                hold = f"{minutes:.0f}m"

        row = [
            datetime.now().isoformat(),
            "CLOSE",
            trade.symbol,
            trade.side.name,
            str(trade.quantity),
            f"{trade.entry_price:.2f}",
            f"{trade.exit_price:.2f}",
            "",  # stop_loss
            "",  # target_price
            f"{trade.pnl:.2f}",
            f"{trade.pnl_pct:.2f}%",
            f"{trade.commission:.2f}",
            trade.strategy_id,
            hold,
            f"{trade.gross_pnl:.2f}",
            f"{trade.tax:.2f}",
        ]
        self._append_row(row)

    def _append_row(self, row: list) -> None:
        try:
            with open(self._path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception:
            logger.exception("Failed to write to trade journal")
