"""Reporting Engine (PRD §19).

Generates periodic performance reports in JSON format.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from crypto.portfolio.portfolio_manager import PortfolioManager
    from crypto.portfolio.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates periodic trading performance reports."""

    def __init__(self, config: dict[str, Any], event_bus: Any) -> None:
        self.config = config
        self.event_bus = event_bus
        self._portfolio: PortfolioManager | None = None
        self._performance: PerformanceMonitor | None = None

        self._report_dir = Path(config.get("reporting", {}).get("report_dir", "./crypto_reports"))
        self._report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ReportGenerator initialized (dir=%s)", self._report_dir)

    def set_dependencies(self, portfolio: PortfolioManager, performance: PerformanceMonitor) -> None:
        self._portfolio = portfolio
        self._performance = performance

    def generate_report(self) -> None:
        """Generate and save a performance report."""
        if not self._portfolio or not self._performance:
            return

        state = self._portfolio.get_state()
        perfs = self._performance.get_all_performances()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": {
                "total_capital": state.total_capital,
                "available_capital": state.available_capital,
                "rolling_pnl": state.rolling_pnl,
                "total_exposure": state.total_exposure,
                "open_positions": state.open_position_count,
                "current_drawdown_pct": state.current_drawdown,
                "peak_capital": state.peak_capital,
            },
            "strategies": {},
        }

        for sid, perf in perfs.items():
            report["strategies"][sid] = {
                "total_trades": perf.total_trades,
                "winning_trades": perf.winning_trades,
                "losing_trades": perf.losing_trades,
                "win_rate_pct": round(perf.win_rate, 2),
                "total_pnl": round(perf.total_pnl, 6),
                "avg_pnl_per_trade": round(perf.avg_pnl_per_trade, 6),
                "max_drawdown_pct": round(perf.max_drawdown, 2),
                "enabled": perf.is_enabled,
            }

        filename = f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self._report_dir / filename

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Report generated: %s", filepath)
