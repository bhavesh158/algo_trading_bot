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
        from crypto.reporting.trade_journal import TradeJournal

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
        self.trade_journal = TradeJournal(self.config)
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

        # Crash recovery: check for restored positions and ensure pairs are tracked
        restored_symbols = self.portfolio_manager.get_open_position_symbols()
        if restored_symbols:
            logger.info(
                "Crash recovery: %d positions restored from state — %s",
                len(restored_symbols), ", ".join(restored_symbols),
            )
            self.pair_selector.set_protected_pairs(restored_symbols)
            # Register restored symbols with order executor to prevent duplicate orders
            for sym in restored_symbols:
                self.order_executor.register_active_symbol(sym)

        # Start status API if enabled
        api_conf = config.get("api", {})
        if api_conf.get("enabled", False):
            from crypto.api.status_server import start_status_server
            self._api_thread = start_status_server(
                system=self,
                host=api_conf.get("host", "0.0.0.0"),
                port=int(api_conf.get("port", 8599)),
            )

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
        """Graceful shutdown — close all open positions, save state, generate report."""
        if not self._running:
            return
        self._running = False
        logger.info("="*60)
        logger.info("  GRACEFUL SHUTDOWN INITIATED")
        logger.info("="*60)

        # Close all open positions at market price
        try:
            self._close_all_positions()
        except Exception:
            logger.exception("Error closing positions during shutdown")

        # Final report
        try:
            self.report_generator.generate_report()
        except Exception:
            logger.exception("Error generating final report")

        self.event_bus.clear()
        logger.info("Crypto trading system stopped.")

    def _close_all_positions(self) -> None:
        """Close every open position at current market price during shutdown."""
        open_positions = self.portfolio_manager.get_open_positions()
        if not open_positions:
            logger.info("No open positions to close.")
            return

        # Safety save — snapshot current state BEFORE attempting closes
        # so if shutdown hangs/crashes, positions are still on disk
        self.portfolio_manager._save_state()

        logger.info("Closing %d open position(s)...", len(open_positions))
        for symbol, pos in list(open_positions.items()):
            try:
                current_price = self.market_data_engine.get_current_price(symbol)
                if current_price <= 0:
                    # Try fetching fresh price from provider
                    ticker = self.provider.fetch_ticker(symbol)
                    current_price = ticker.get("last", 0) if ticker else 0

                if current_price <= 0:
                    logger.error("Cannot get price for %s — position kept in state file for recovery", symbol)
                    continue

                commission = self.order_executor.get_commission(current_price * pos.quantity)
                trade = self.portfolio_manager.close_position(symbol, current_price, commission)
                if trade:
                    self.trade_journal.log_close(trade)
                    self.order_executor.release_symbol(symbol)
                    logger.info(
                        "  Closed %s: pnl=%.4f (%.2f%%)",
                        symbol, trade.pnl, trade.pnl_pct,
                    )
            except Exception:
                logger.exception("Failed to close position for %s", symbol)

        # Only clear state if ALL positions were closed successfully
        remaining = self.portfolio_manager.get_open_positions()
        if remaining:
            logger.warning(
                "%d position(s) could NOT be closed — preserved in state.json for crash recovery: %s",
                len(remaining), ", ".join(remaining.keys()),
            )
        else:
            self.portfolio_manager.state_manager.clear_state()
            logger.info("All positions closed and state cleared.")


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
    signal.signal(signal.SIGHUP, _signal_handler)  # terminal/SSH disconnect

    system.initialize()
    system.run()


if __name__ == "__main__":
    main()
