"""Lightweight Status API Server (stdlib only — zero extra dependencies).

Runs in a daemon thread alongside the trading system. Provides read-only
endpoints for monitoring:

    GET /health  — liveness check
    GET /status  — open positions, capital, P&L
    GET /trades  — recent entries from trade journal CSV
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from datetime import datetime, timezone
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from crypto.portfolio.portfolio_manager import PortfolioManager
    from crypto.selection.pair_selector import PairSelector

logger = logging.getLogger(__name__)

_start_time: float = 0.0
_trading_system: Any = None


class _Handler(BaseHTTPRequestHandler):
    """Handles incoming HTTP requests."""

    def do_GET(self) -> None:
        routes = {
            "/health": self._health,
            "/status": self._status,
            "/trades": self._trades,
            "/ledger": self._ledger,
        }

        handler = routes.get(self.path.split("?")[0])
        if handler:
            try:
                data = handler()
                self._json_response(200, data)
            except Exception as e:
                logger.exception("API error on %s", self.path)
                self._json_response(500, {"error": str(e)})
        else:
            self._json_response(404, {
                "error": "not found",
                "endpoints": ["/health", "/status", "/trades", "/ledger"],
            })

    def _health(self) -> dict:
        uptime_s = time.time() - _start_time
        hours, rem = divmod(int(uptime_s), 3600)
        mins, secs = divmod(rem, 60)

        system = _trading_system
        return {
            "status": "ok",
            "mode": system.mode.name if system else "unknown",
            "uptime": f"{hours}h {mins}m {secs}s",
            "uptime_seconds": round(uptime_s, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _status(self) -> dict:
        system = _trading_system
        if not system:
            return {"error": "system not initialized"}

        pm: PortfolioManager = system.portfolio_manager
        state = pm.get_state()

        positions = []
        for pos in state.positions:
            if pos.status.name != "OPEN":
                continue
            pnl = (pos.current_price - pos.entry_price) * pos.quantity
            if pos.side.name == "SELL":
                pnl = -pnl
            pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 else 0

            positions.append({
                "symbol": pos.symbol,
                "side": pos.side.name,
                "quantity": round(pos.quantity, 8),
                "entry_price": round(pos.entry_price, 6),
                "current_price": round(pos.current_price, 6),
                "pnl": round(pnl, 4),
                "pnl_pct": round(pnl_pct, 2),
                "stop_loss": round(pos.stop_loss, 6) if pos.stop_loss else None,
                "target_price": round(pos.target_price, 6) if pos.target_price else None,
                "strategy": pos.strategy_id,
                "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
            })

        total_unrealized = sum(p["pnl"] for p in positions)

        ps: PairSelector = system.pair_selector
        return {
            "capital": {
                "total": round(state.total_capital, 2),
                "available": round(state.available_capital, 2),
                "equity": round(state.total_capital + total_unrealized, 2),
                "peak": round(state.peak_capital, 2),
                "drawdown_pct": round(state.current_drawdown, 2),
            },
            "pnl": {
                "realized": round(state.rolling_pnl, 4),
                "unrealized": round(total_unrealized, 4),
                "total": round(state.rolling_pnl + total_unrealized, 4),
            },
            "positions": {
                "count": len(positions),
                "details": positions,
            },
            "active_pairs": ps.active_pairs if ps else [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _trades(self) -> dict:
        system = _trading_system
        if not system:
            return {"error": "system not initialized"}

        data_dir = system.config.get("system", {}).get("data_dir", "./crypto_data")
        journal_path = Path(data_dir) / "trade_journal.csv"

        if not journal_path.exists():
            return {"trades": [], "count": 0}

        # Read last 50 entries
        trades = []
        with open(journal_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows[-50:]:
            trades.append({k: v for k, v in row.items() if v != ""})

        return {
            "trades": trades,
            "count": len(trades),
            "total_in_journal": len(rows),
        }

    def _ledger(self) -> dict:
        system = _trading_system
        if not system:
            return {"error": "system not initialized"}

        data_dir = system.config.get("system", {}).get("data_dir", "./crypto_data")
        ledger_path = Path(data_dir) / "capital_ledger.csv"

        if not ledger_path.exists():
            return {"entries": [], "count": 0}

        entries = []
        with open(ledger_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows[-100:]:
            entries.append({k: v for k, v in row.items() if v != ""})

        return {
            "entries": entries,
            "count": len(entries),
            "total_in_ledger": len(rows),
        }

    def _json_response(self, status: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Suppress default stderr logging — use our logger instead
        logger.debug("API %s %s", self.command, self.path)


def start_status_server(system: Any, host: str = "0.0.0.0", port: int = 8599) -> threading.Thread:
    """Start the status API server in a daemon thread.

    Returns the thread (already started). Thread is daemon so it
    dies automatically when the main process exits.
    """
    global _start_time, _trading_system
    _start_time = time.time()
    _trading_system = system

    server = HTTPServer((host, port), _Handler)

    thread = threading.Thread(
        target=server.serve_forever,
        name="status-api",
        daemon=True,
    )
    thread.start()
    logger.info("Status API server started on http://%s:%d", host, port)
    return thread
