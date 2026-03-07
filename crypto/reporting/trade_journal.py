"""Trade Journal — logs all position opens and closes to a CSV file.

Provides a persistent, human-readable record of every trade for analysis.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crypto.core.models import Order, Position, Trade

logger = logging.getLogger(__name__)


class TradeJournal:
    """Appends trade events to a CSV file."""

    OPEN_HEADERS = [
        "timestamp", "event", "symbol", "side", "quantity",
        "entry_price", "stop_loss", "target_price", "strategy_id",
    ]
    CLOSE_HEADERS = [
        "timestamp", "event", "symbol", "side", "quantity",
        "entry_price", "exit_price", "pnl", "pnl_pct",
        "commission", "strategy_id", "hold_duration",
        "gross_pnl", "tax",
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        data_dir = config.get("system", {}).get("data_dir", "./crypto_data")
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._open_path = self._dir / "trade_journal.csv"
        self._init_csv()
        logger.info("TradeJournal initialized (%s)", self._open_path)

    def _init_csv(self) -> None:
        """Create CSV with headers if it doesn't exist."""
        if not self._open_path.exists():
            with open(self._open_path, "w", newline="") as f:
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
            datetime.now(timezone.utc).isoformat(),
            "OPEN",
            position.symbol,
            position.side.name,
            f"{position.quantity:.8f}",
            f"{position.entry_price:.6f}",
            "",  # exit_price
            f"{position.stop_loss:.6f}" if position.stop_loss else "",
            f"{position.target_price:.6f}" if position.target_price else "",
            "",  # pnl
            "",  # pnl_pct
            "",  # commission
            position.strategy_id,
            "",  # hold_duration
        ]
        self._append_row(row)

    def log_close(self, trade: Trade) -> None:
        """Log a position close event."""
        hold = ""
        if trade.exit_time and trade.entry_time:
            delta = trade.exit_time - trade.entry_time
            hours = delta.total_seconds() / 3600
            hold = f"{hours:.1f}h"

        row = [
            datetime.now(timezone.utc).isoformat(),
            "CLOSE",
            trade.symbol,
            trade.side.name,
            f"{trade.quantity:.8f}",
            f"{trade.entry_price:.6f}",
            f"{trade.exit_price:.6f}",
            "",  # stop_loss
            "",  # target_price
            f"{trade.pnl:.4f}",
            f"{trade.pnl_pct:.2f}%",
            f"{trade.commission:.4f}",
            trade.strategy_id,
            hold,
            f"{trade.gross_pnl:.4f}",
            f"{trade.tax:.4f}",
        ]
        self._append_row(row)

    def _append_row(self, row: list) -> None:
        with open(self._open_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
