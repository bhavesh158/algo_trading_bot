"""Reporting System (PRD §21).

Generates daily trade reports with performance metrics.
Reports are saved as JSON files and logged to console.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from stocks.core.event_bus import EventBus
from stocks.core.models import Trade

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates and persists trading reports."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus

        report_config = config.get("reporting", {})
        self._report_dir = Path(report_config.get("report_dir", "./reports"))
        self._format = report_config.get("format", "json")
        self._daily_report_enabled = report_config.get("daily_report", True)

        self._report_dir.mkdir(parents=True, exist_ok=True)

        # Will be set by TradingSystem
        self._portfolio_manager = None
        self._performance_monitor = None

        logger.info("ReportGenerator initialized (dir=%s)", self._report_dir)

    def set_dependencies(self, portfolio_manager: Any, performance_monitor: Any) -> None:
        self._portfolio_manager = portfolio_manager
        self._performance_monitor = performance_monitor

    def generate_daily_report(self) -> dict[str, Any]:
        """Generate the end-of-day trading report."""
        if not self._daily_report_enabled:
            return {}

        today = datetime.now().strftime("%Y-%m-%d")
        trades = self._get_daily_trades()

        # Basic statistics
        total_trades = len(trades)
        winning = [t for t in trades if t.is_winner]
        losing = [t for t in trades if not t.is_winner]
        total_pnl = sum(t.pnl for t in trades)
        total_commission = sum(t.commission for t in trades)

        report = {
            "date": today,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_trades": total_trades,
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": round(len(winning) / total_trades * 100, 1) if total_trades > 0 else 0,
                "total_pnl": round(total_pnl, 2),
                "total_commission": round(total_commission, 2),
                "net_pnl": round(total_pnl - total_commission, 2),
                "avg_win": round(sum(t.pnl for t in winning) / len(winning), 2) if winning else 0,
                "avg_loss": round(sum(t.pnl for t in losing) / len(losing), 2) if losing else 0,
                "largest_win": round(max((t.pnl for t in winning), default=0), 2),
                "largest_loss": round(min((t.pnl for t in losing), default=0), 2),
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "side": t.side.name,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 2),
                    "strategy": t.strategy_id,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                }
                for t in trades
            ],
        }

        # Add strategy performance if available
        if self._performance_monitor:
            report["strategy_performance"] = self._performance_monitor.get_summary()

        # Add portfolio state if available
        if self._portfolio_manager:
            state = self._portfolio_manager.get_state()
            report["portfolio"] = {
                "total_capital": round(state.total_capital, 2),
                "available_capital": round(state.available_capital, 2),
                "daily_pnl": round(state.daily_pnl, 2),
                "current_drawdown": round(state.current_drawdown, 2),
            }

        # Save report
        self._save_report(report, today)

        # Log summary
        logger.info("=" * 50)
        logger.info("  Daily Report: %s", today)
        logger.info("  Trades: %d (W:%d L:%d)", total_trades, len(winning), len(losing))
        logger.info("  Net P&L: ₹%.2f", total_pnl)
        if total_trades > 0:
            logger.info("  Win Rate: %.1f%%", len(winning) / total_trades * 100)
        logger.info("=" * 50)

        return report

    def _get_daily_trades(self) -> list[Trade]:
        """Get today's trades from the portfolio manager."""
        if self._portfolio_manager:
            return self._portfolio_manager.daily_trades
        return []

    def _save_report(self, report: dict, date_str: str) -> None:
        """Persist the report to disk."""
        try:
            filepath = self._report_dir / f"report_{date_str}.json"
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Report saved to %s", filepath)
        except Exception:
            logger.exception("Failed to save report")
