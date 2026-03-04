#!/usr/bin/env python3
"""Crypto Automated Trading System — CLI Entrypoint.

Usage:
    python -m crypto.crypto_main --mode paper
    python -m crypto.crypto_main --mode live --exchange binance
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from typing import Any

from crypto.config.settings import load_config, get_nested
from crypto.core.enums import TradingMode
from crypto.core.event_bus import EventBus
from crypto.utils.logger import setup_logging
from crypto.utils.security import validate_live_trading_prerequisites

logger = logging.getLogger(__name__)


class CryptoTradingSystem:
    """Top-level orchestrator that initializes and runs all components."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mode = TradingMode.LIVE if config["system"]["mode"] == "live" else TradingMode.PAPER
        self.event_bus = EventBus()
        self._running = False

    def initialize(self) -> None:
        """Initialize all trading system components."""
        logger.info("=" * 60)
        logger.info("  Crypto Automated Trading System")
        logger.info("  Mode: %s", self.mode.name)
        logger.info("  Exchange: %s", get_nested(self.config, "exchange", "name"))
        logger.info("=" * 60)

        # Safety check for live trading
        if self.mode == TradingMode.LIVE:
            exchange_name = get_nested(self.config, "exchange", "name") or "binance"
            issues = validate_live_trading_prerequisites(exchange=exchange_name)
            if issues:
                for issue in issues:
                    logger.error("Live trading prerequisite failed: %s", issue)
                raise RuntimeError("Cannot start live trading — prerequisites not met.")

        # Import components
        from crypto.data.providers.ccxt_provider import CcxtProvider
        from crypto.data.market_data_engine import MarketDataEngine
        from crypto.selection.pair_selector import PairSelector
        from crypto.analysis.ai_analysis import AIAnalysis
        from crypto.analysis.regime_detector import RegimeDetector
        from crypto.analysis.volatility_monitor import VolatilityMonitor
        from crypto.strategy.strategy_engine import StrategyEngine
        from crypto.risk.risk_manager import RiskManager
        from crypto.risk.position_sizer import PositionSizer
        from crypto.risk.drawdown_monitor import DrawdownMonitor
        from crypto.execution.order_executor import OrderExecutor
        from crypto.portfolio.portfolio_manager import PortfolioManager
        from crypto.portfolio.performance_monitor import PerformanceMonitor
        from crypto.scheduler.continuous_scheduler import ContinuousScheduler
        from crypto.reporting.report_generator import ReportGenerator
        from crypto.reporting.alert_manager import AlertManager

        # Connect to exchange data provider
        exchange_name = get_nested(self.config, "exchange", "name") or "binance"
        self.provider = CcxtProvider(exchange_name, self.config)

        if self.mode == TradingMode.LIVE:
            connected = self.provider.connect(
                api_key=os.environ.get("EXCHANGE_API_KEY", ""),
                api_secret=os.environ.get("EXCHANGE_API_SECRET", ""),
                password=os.environ.get("EXCHANGE_PASSWORD", ""),
            )
        else:
            connected = self.provider.connect()

        if not connected:
            raise RuntimeError(f"Failed to connect to exchange: {exchange_name}")

        # Instantiate all components
        self.alert_manager = AlertManager(self.config, self.event_bus)
        self.portfolio_manager = PortfolioManager(self.config, self.event_bus)
        self.risk_manager = RiskManager(self.config, self.event_bus, self.portfolio_manager)
        self.position_sizer = PositionSizer(self.config, self.event_bus, self.portfolio_manager)
        self.drawdown_monitor = DrawdownMonitor(self.config, self.event_bus, self.portfolio_manager)
        self.market_data_engine = MarketDataEngine(self.config, self.event_bus, self.provider)
        self.pair_selector = PairSelector(self.config, self.event_bus, self.provider)
        self.regime_detector = RegimeDetector(self.config, self.event_bus)
        self.ai_analysis = AIAnalysis(self.config, self.event_bus)
        self.volatility_monitor = VolatilityMonitor(self.config, self.event_bus)
        self.order_executor = OrderExecutor(self.config, self.event_bus, self.mode)
        self.strategy_engine = StrategyEngine(
            self.config, self.event_bus, self.risk_manager,
            self.position_sizer, self.ai_analysis, self.regime_detector,
        )
        self.performance_monitor = PerformanceMonitor(self.config, self.event_bus)
        self.report_generator = ReportGenerator(self.config, self.event_bus)
        self.scheduler = ContinuousScheduler(self.config, self.event_bus)

        # Wire exchange adapter for live mode
        if self.mode == TradingMode.LIVE:
            from crypto.execution.exchange_adapters.ccxt_adapter import CcxtExchangeAdapter
            adapter = CcxtExchangeAdapter(exchange_name, self.config)
            if adapter.connect():
                self.order_executor.set_exchange_adapter(adapter)
            else:
                raise RuntimeError("Failed to connect exchange adapter for live trading")

        self.scheduler.set_trading_system(self)

        logger.info("All components initialized")
        logger.info("Event bus: %d subscriptions active", self.event_bus.subscriber_count)

    def run(self) -> None:
        """Start the continuous trading loop."""
        self._running = True
        logger.info("Crypto trading system starting...")
        try:
            self.scheduler.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown."""
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down crypto trading system...")
        self.report_generator.generate_report()
        self.event_bus.clear()
        logger.info("Crypto trading system stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto Automated Trading System")
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--exchange", type=str, default=None,
        help="Exchange name (e.g. binance, coinbase, kraken)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to user config YAML (overrides defaults)",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.mode:
        config["system"]["mode"] = args.mode
    if args.exchange:
        config.setdefault("exchange", {})["name"] = args.exchange
    if args.log_level:
        config["system"]["log_level"] = args.log_level

    setup_logging(
        level=config["system"]["log_level"],
        log_dir=get_nested(config, "system", "data_dir"),
    )

    system = CryptoTradingSystem(config)

    def _signal_handler(signum, frame):
        logger.info("Signal %s received, shutting down...", signum)
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)

    system.initialize()
    system.run()


if __name__ == "__main__":
    main()
