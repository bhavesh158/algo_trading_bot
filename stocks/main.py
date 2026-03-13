#!/usr/bin/env python3
"""Automated AI Trading System — CLI Entrypoint.

Usage:
    python main.py --mode paper
    python main.py --mode live --config my_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Any

from stocks.config.settings import load_config, get_nested
from stocks.core.enums import TradingMode
from stocks.core.event_bus import EventBus
from stocks.utils.logger import setup_logging
from stocks.utils.security import validate_live_trading_prerequisites

logger = logging.getLogger(__name__)


class TradingSystem:
    """Top-level orchestrator that initializes and runs all components."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mode = TradingMode.LIVE if config["system"]["mode"] == "live" else TradingMode.PAPER
        self.event_bus = EventBus()
        self._running = False
        self._components: list[Any] = []

    def initialize(self) -> None:
        """Initialize all trading system components."""
        logger.info("=" * 60)
        logger.info("  Automated AI Trading System")
        logger.info("  Mode: %s", self.mode.name)
        logger.info("=" * 60)

        # Safety check for live trading
        if self.mode == TradingMode.LIVE:
            broker_name = get_nested(self.config, "broker", "adapter") or "angelone"
            issues = validate_live_trading_prerequisites(broker=broker_name)
            if issues:
                for issue in issues:
                    logger.error("Live trading prerequisite failed: %s", issue)
                raise RuntimeError("Cannot start live trading — prerequisites not met.")

        # Import and initialize components
        from stocks.data.market_data_engine import MarketDataEngine
        from stocks.selection.stock_selector import StockSelector
        from stocks.analysis.regime_detector import RegimeDetector
        from stocks.analysis.ai_analysis import AIAnalysis
        from stocks.analysis.multi_timeframe import MultiTimeframeAnalyzer
        from stocks.analysis.news_monitor import NewsMonitor
        from stocks.strategy.strategy_engine import StrategyEngine
        from stocks.risk.risk_manager import RiskManager
        from stocks.risk.position_sizer import PositionSizer
        from stocks.risk.drawdown_monitor import DrawdownMonitor
        from stocks.execution.order_executor import OrderExecutor
        from stocks.portfolio.portfolio_manager import PortfolioManager
        from stocks.portfolio.performance_monitor import PerformanceMonitor
        from stocks.scheduler.trading_scheduler import TradingScheduler
        from stocks.reporting.report_generator import ReportGenerator
        from stocks.reporting.alert_manager import AlertManager
        from stocks.reporting.trade_journal import TradeJournal

        # Instantiate components (order matters for dependency chain)
        self.trade_journal = TradeJournal(self.config)
        self.alert_manager = AlertManager(self.config, self.event_bus)
        self.portfolio_manager = PortfolioManager(self.config, self.event_bus)
        self.risk_manager = RiskManager(self.config, self.event_bus, self.portfolio_manager)
        self.position_sizer = PositionSizer(self.config, self.event_bus, self.portfolio_manager)
        self.drawdown_monitor = DrawdownMonitor(self.config, self.event_bus, self.portfolio_manager)
        self.market_data_engine = MarketDataEngine(self.config, self.event_bus)
        self.stock_selector = StockSelector(self.config, self.event_bus, self.market_data_engine)
        self.regime_detector = RegimeDetector(self.config, self.event_bus)
        self.ai_analysis = AIAnalysis(self.config, self.event_bus)
        self.multi_tf = MultiTimeframeAnalyzer(self.config, self.event_bus)
        self.news_monitor = NewsMonitor(self.config, self.event_bus)
        self.order_executor = OrderExecutor(self.config, self.event_bus, self.mode)
        self.strategy_engine = StrategyEngine(
            self.config, self.event_bus, self.risk_manager,
            self.position_sizer, self.ai_analysis, self.regime_detector,
            multi_tf=self.multi_tf,
        )
        self.performance_monitor = PerformanceMonitor(self.config, self.event_bus)
        self.report_generator = ReportGenerator(self.config, self.event_bus)
        self.scheduler = TradingScheduler(self.config, self.event_bus)

        self._components = [
            self.trade_journal,
            self.alert_manager,
            self.portfolio_manager,
            self.risk_manager,
            self.position_sizer,
            self.drawdown_monitor,
            self.market_data_engine,
            self.stock_selector,
            self.regime_detector,
            self.ai_analysis,
            self.multi_tf,
            self.news_monitor,
            self.order_executor,
            self.strategy_engine,
            self.performance_monitor,
            self.report_generator,
            self.scheduler,
        ]

        # Wire broker adapter for live trading
        if self.mode == TradingMode.LIVE:
            self._setup_broker_adapter()

        # Give scheduler a reference to this system for orchestration
        self.scheduler.set_trading_system(self)

        logger.info("All components initialized (%d total)", len(self._components))
        logger.info("Event bus: %d subscriptions active", self.event_bus.subscriber_count)

    def _setup_broker_adapter(self) -> None:
        """Initialize the broker adapter for live trading."""
        import os
        broker_name = get_nested(self.config, "broker", "adapter") or "zerodha"

        if broker_name == "angelone":
            from stocks.execution.broker_adapters.angelone_adapter import AngelOneAdapter
            adapter = AngelOneAdapter(
                api_key=os.environ.get("BROKER_API_KEY", ""),
                client_id=os.environ.get("BROKER_CLIENT_ID", ""),
                password=os.environ.get("BROKER_PASSWORD", ""),
                totp_secret=os.environ.get("BROKER_TOTP_SECRET", ""),
            )
        else:
            from stocks.execution.broker_adapters.zerodha_adapter import ZerodhaAdapter
            adapter = ZerodhaAdapter(
                api_key=os.environ.get("BROKER_API_KEY", ""),
                api_secret=os.environ.get("BROKER_API_SECRET", ""),
                access_token=os.environ.get("BROKER_ACCESS_TOKEN", ""),
            )

        if adapter.connect():
            self.order_executor.set_broker_adapter(adapter)
            logger.info("Broker adapter connected: %s", broker_name)
        else:
            raise RuntimeError(f"Failed to connect broker adapter: {broker_name}")

    def run(self) -> None:
        """Start the trading system main loop."""
        self._running = True
        logger.info("Trading system starting...")
        try:
            self.scheduler.run(self._components)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown — save state for mid-day recovery."""
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down trading system...")

        # Save state so mid-day restarts can resume
        open_count = len(self.portfolio_manager.get_open_positions())
        self.portfolio_manager._save_state()
        logger.info("Saved state with %d open position(s) for restart", open_count)

        self.report_generator.generate_daily_report()
        self.event_bus.clear()
        logger.info("Trading system stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated AI Trading System")
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to user config YAML file (overrides defaults)",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI args override config file
    if args.mode:
        config["system"]["mode"] = args.mode
    if args.log_level:
        config["system"]["log_level"] = args.log_level

    # Setup logging
    setup_logging(
        level=config["system"]["log_level"],
        log_dir=get_nested(config, "system", "data_dir"),
    )

    # Create and run the trading system
    system = TradingSystem(config)

    # Handle SIGTERM gracefully
    def _signal_handler(signum, frame):
        logger.info("Signal %s received, shutting down...", signum)
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)

    system.initialize()
    system.run()


if __name__ == "__main__":
    main()
