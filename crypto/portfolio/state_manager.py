"""State Persistence Manager.

Saves and restores portfolio state (open positions, capital, P&L) to a JSON
file on disk. This ensures:
- Open positions survive graceful restarts
- Crash recovery can detect and handle orphaned positions
- No state is silently lost
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from crypto.core.enums import OrderSide, PositionStatus
from crypto.core.models import Position

logger = logging.getLogger(__name__)

STATE_VERSION = 1


class StateManager:
    """Persists trading state to a JSON file."""

    def __init__(self, config: dict[str, Any]) -> None:
        data_dir = config.get("system", {}).get("data_dir", "./crypto_data")
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._dir / "state.json"
        self._backup_path = self._dir / "state.backup.json"
        logger.info("StateManager initialized (%s)", self._state_path)

    def save_state(
        self,
        positions: dict[str, Position],
        total_capital: float,
        available_capital: float,
        peak_capital: float,
        rolling_pnl: float,
        entered_symbols: set[str] | None = None,
    ) -> None:
        """Save current state to disk. Writes atomically via temp + rename."""
        state = {
            "version": STATE_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "capital": {
                "total": total_capital,
                "available": available_capital,
                "peak": peak_capital,
                "rolling_pnl": rolling_pnl,
            },
            "positions": {
                symbol: _position_to_dict(pos)
                for symbol, pos in positions.items()
                if pos.status == PositionStatus.OPEN
            },
            "entered_symbols": sorted(entered_symbols) if entered_symbols else [],
        }

        # Backup existing state before overwriting
        if self._state_path.exists():
            try:
                self._state_path.rename(self._backup_path)
            except OSError:
                pass

        tmp_path = self._state_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            tmp_path.rename(self._state_path)
        except Exception:
            logger.exception("Failed to save state")
            # Restore backup
            if self._backup_path.exists():
                try:
                    self._backup_path.rename(self._state_path)
                except OSError:
                    pass

    def load_state(self) -> Optional[dict[str, Any]]:
        """Load state from disk. Returns None if no state file exists."""
        path = self._state_path
        if not path.exists():
            # Try backup
            path = self._backup_path
            if not path.exists():
                logger.info("No saved state found — starting fresh")
                return None

        try:
            with open(path) as f:
                state = json.load(f)

            if state.get("version") != STATE_VERSION:
                logger.warning(
                    "State version mismatch (got %s, expected %s) — ignoring",
                    state.get("version"), STATE_VERSION,
                )
                return None

            logger.info(
                "Loaded saved state from %s (%d open positions, capital=%.2f)",
                state.get("saved_at", "unknown"),
                len(state.get("positions", {})),
                state.get("capital", {}).get("total", 0),
            )
            return state

        except (json.JSONDecodeError, KeyError):
            logger.exception("Corrupted state file — starting fresh")
            return None

    def save_cooldowns(self, recently_closed: dict[str, datetime]) -> None:
        """Persist close cooldowns to a separate file."""
        path = self._dir / "cooldowns.json"
        try:
            data = {sym: ts.isoformat() for sym, ts in recently_closed.items()}
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            logger.exception("Failed to save cooldowns")

    def load_cooldowns(self) -> dict[str, datetime]:
        """Restore close cooldowns from disk."""
        path = self._dir / "cooldowns.json"
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
            cooldowns = {sym: datetime.fromisoformat(ts) for sym, ts in data.items()}
            if cooldowns:
                logger.info("Restored %d cooldown entries from disk", len(cooldowns))
            return cooldowns
        except Exception:
            logger.exception("Failed to load cooldowns")
            return {}

    def restore_positions(self, state: dict[str, Any]) -> dict[str, Position]:
        """Convert saved position dicts back to Position objects."""
        positions = {}
        for symbol, data in state.get("positions", {}).items():
            try:
                pos = _dict_to_position(data)
                positions[symbol] = pos
                logger.info(
                    "Restored position: %s %s qty=%.6f entry=%.4f",
                    pos.side.name, symbol, pos.quantity, pos.entry_price,
                )
            except Exception:
                logger.exception("Failed to restore position for %s", symbol)
        return positions

    def clear_state(self) -> None:
        """Remove state file (e.g. after all positions are closed cleanly)."""
        for path in (self._state_path, self._backup_path):
            if path.exists():
                path.unlink()
        logger.info("State file cleared")

    @property
    def has_saved_state(self) -> bool:
        return self._state_path.exists() or self._backup_path.exists()


def _position_to_dict(pos: Position) -> dict:
    return {
        "id": pos.id,
        "symbol": pos.symbol,
        "side": pos.side.name,
        "quantity": pos.quantity,
        "entry_price": pos.entry_price,
        "current_price": pos.current_price,
        "stop_loss": pos.stop_loss,
        "target_price": pos.target_price,
        "strategy_id": pos.strategy_id,
        "entry_order_id": pos.entry_order_id,
        "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
        "entry_commission": pos.entry_commission,
    }


def _dict_to_position(data: dict) -> Position:
    opened_at = datetime.fromisoformat(data["opened_at"]) if data.get("opened_at") else datetime.now(timezone.utc)
    return Position(
        id=data.get("id", ""),
        symbol=data["symbol"],
        side=OrderSide[data["side"]],
        quantity=data["quantity"],
        entry_price=data["entry_price"],
        current_price=data.get("current_price", data["entry_price"]),
        stop_loss=data.get("stop_loss", 0.0),
        target_price=data.get("target_price", 0.0),
        strategy_id=data.get("strategy_id", ""),
        entry_order_id=data.get("entry_order_id", ""),
        opened_at=opened_at,
        status=PositionStatus.OPEN,
        entry_commission=data.get("entry_commission", 0.0),
    )
