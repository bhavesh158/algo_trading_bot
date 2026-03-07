"""Continuous 24/7 Scheduler (PRD §4).

Runs the main trading loop continuously. Unlike the stock system's session-based
scheduler, this runs indefinitely with periodic maintenance tasks.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from crypto.core.enums import OrderSide, OrderType
from crypto.core.event_bus import EventBus
from crypto.core.models import Order

logger = logging.getLogger(__name__)


class ContinuousScheduler:
    """Manages the 24/7 continuous trading loop."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus

        sched = config.get("scheduler", {})
        self._loop_interval = sched.get("main_loop_interval_seconds", 10)
        self._pair_refresh_hours = sched.get("pair_refresh_interval_hours", 4)
        self._report_hours = sched.get("report_interval_hours", 24)
        self._risk_reset_hours = sched.get("risk_reset_interval_hours", 24)
        self._heartbeat_cycles = sched.get("heartbeat_interval_cycles", 30)

        self._running = False
        self._cycle_count = 0
        self._last_pair_refresh = datetime.min.replace(tzinfo=timezone.utc)
        self._last_report = datetime.min.replace(tzinfo=timezone.utc)
        self._last_risk_reset = datetime.min.replace(tzinfo=timezone.utc)
        self._system = None

        logger.info("ContinuousScheduler initialized (interval=%ds)", self._loop_interval)

    def set_trading_system(self, system: Any) -> None:
        self._system = system

    def run(self) -> None:
        """Main 24/7 loop."""
        self._running = True
        logger.info("Continuous scheduler starting — 24/7 mode")

        # Initial setup
        self._run_initial_setup()

        while self._running:
            try:
                self._cycle_count += 1
                now = datetime.now(timezone.utc)

                # Periodic tasks
                self._check_periodic_tasks(now)

                # Main trading cycle
                self._run_trading_cycle()

                # Heartbeat
                if self._cycle_count % self._heartbeat_cycles == 0:
                    self._log_heartbeat()

                time.sleep(self._loop_interval)

            except KeyboardInterrupt:
                logger.info("Scheduler interrupted")
                self._running = False
            except Exception:
                logger.exception("Error in main loop — will retry next cycle")
                time.sleep(self._loop_interval)

    def _run_initial_setup(self) -> None:
        """Run once at startup: connect, load data, build watchlist."""
        if not self._system:
            return

        system = self._system
        logger.info("=== INITIAL SETUP ===")

        # Build trading pair watchlist
        system.pair_selector.build_watchlist()
        pairs = system.pair_selector.active_pairs

        if not pairs:
            logger.warning("No pairs passed selection filters — will retry on next refresh")
            return

        # Load historical data
        system.market_data_engine.load_historical_data(pairs)

        # Initialize analysis
        system.regime_detector.set_market_data(system.market_data_engine)
        system.regime_detector.detect_regime()
        system.ai_analysis.set_market_data(system.market_data_engine)
        system.volatility_monitor.set_market_data(system.market_data_engine)

        # Load strategies
        system.strategy_engine.load_strategies(system.market_data_engine)

        # Wire report generator
        system.report_generator.set_dependencies(
            system.portfolio_manager, system.performance_monitor,
        )

        now = datetime.now(timezone.utc)
        self._last_pair_refresh = now
        self._last_report = now
        self._last_risk_reset = now

        logger.info("Initial setup complete. Trading %d pairs", len(pairs))

    def _check_periodic_tasks(self, now: datetime) -> None:
        """Run periodic maintenance tasks."""
        if not self._system:
            return

        # Pair refresh
        if now - self._last_pair_refresh >= timedelta(hours=self._pair_refresh_hours):
            logger.info("--- PAIR REFRESH ---")
            self._system.pair_selector.build_watchlist()
            new_pairs = self._system.pair_selector.active_pairs
            self._system.market_data_engine.load_historical_data(new_pairs)
            self._last_pair_refresh = now

        # Rolling risk reset
        if now - self._last_risk_reset >= timedelta(hours=self._risk_reset_hours):
            logger.info("--- ROLLING RISK RESET ---")
            self._system.risk_manager.reset_rolling_state()
            self._system.portfolio_manager.reset_rolling_pnl()
            self._last_risk_reset = now

        # Periodic report
        if now - self._last_report >= timedelta(hours=self._report_hours):
            logger.info("--- PERIODIC REPORT ---")
            self._system.report_generator.generate_report()
            self._last_report = now

    def _run_trading_cycle(self) -> None:
        """One iteration of the trading loop."""
        if not self._system:
            return

        system = self._system
        pairs = system.pair_selector.active_pairs
        if not pairs:
            return

        # Check safety conditions
        if not system.volatility_monitor.check_conditions(pairs):
            return
        if system.drawdown_monitor.is_trading_paused:
            return
        if system.risk_manager.is_loss_limit_breached:
            return

        # Update market data — fetch only the timeframes needed this cycle.
        # 5m: every cycle (for volatility monitor / price updates)
        # 15m: every 3 cycles (~48s — strategies use 15m candles)
        # 1h:  every 12 cycles (~3min — breakout uses 1h candles)
        tfs = ["5m"]
        if self._cycle_count % 3 == 0:
            tfs.append("15m")
        if self._cycle_count % 12 == 0:
            tfs.append("1h")
        system.market_data_engine.update_data(pairs, timeframes=tfs)

        # Update position prices
        for symbol, pos in system.portfolio_manager.get_open_positions().items():
            price = system.market_data_engine.get_current_price(symbol)
            if price > 0:
                system.portfolio_manager.update_position_price(symbol, price)

        # Check drawdown
        dd_level = system.drawdown_monitor.check_drawdown()
        if dd_level == "reduce_size":
            system.position_sizer.set_drawdown_factor(system.drawdown_monitor.size_reduction_factor)

        # Detect market regime
        system.regime_detector.detect_regime()

        # Run strategies
        signals = system.strategy_engine.run_strategies(pairs)

        # Execute signals
        for signal in signals:
            # Re-check position limit before each order (signals batch from same cycle)
            if not system.risk_manager.can_take_trade(signal):
                logger.info(
                    "BLOCKED by risk_manager: %s %s conf=%.2f",
                    signal.side.name, signal.symbol, signal.confidence,
                )
                continue

            quantity = system.position_sizer.calculate_quantity(signal)
            if quantity <= 0:
                logger.info(
                    "BLOCKED zero quantity: %s %s entry=%.4f stop=%.4f",
                    signal.side.name, signal.symbol, signal.entry_price, signal.stop_loss,
                )
                continue

            order = Order(
                symbol=signal.symbol,
                side=signal.side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=signal.entry_price,
                stop_price=signal.stop_loss,
                target_price=signal.target_price,
                strategy_id=signal.strategy_id,
                signal_id=signal.id,
            )

            current_price = system.market_data_engine.get_current_price(signal.symbol)
            filled_order = system.order_executor.execute_order(order, current_price)

            if filled_order.status.name == "FILLED":
                pos = system.portfolio_manager.open_position(filled_order)
                system.trade_journal.log_open(filled_order, pos)
            else:
                logger.info(
                    "ORDER %s: %s %s qty=%.6f price=%.4f",
                    filled_order.status.name, signal.side.name,
                    signal.symbol, quantity, current_price,
                )

        # Check exits for open positions
        open_positions = system.portfolio_manager.get_open_positions()
        for symbol, pos in open_positions.items():
            current_price = system.market_data_engine.get_current_price(symbol)
            is_long = pos.side == OrderSide.BUY

            # Stop loss check (long: price falls to stop; short: price rises to stop)
            if pos.stop_loss > 0:
                stop_hit = current_price <= pos.stop_loss if is_long else current_price >= pos.stop_loss
                if stop_hit:
                    commission = system.order_executor.get_commission(current_price * pos.quantity)
                    trade = system.portfolio_manager.close_position(symbol, current_price, commission)
                    if trade:
                        system.trade_journal.log_close(trade)
                        system.performance_monitor.record_trade(trade)
                        system.order_executor.release_symbol(symbol)
                    continue

            # Target check (long: price rises to target; short: price falls to target)
            if pos.target_price > 0:
                target_hit = current_price >= pos.target_price if is_long else current_price <= pos.target_price
                if target_hit:
                    commission = system.order_executor.get_commission(current_price * pos.quantity)
                    trade = system.portfolio_manager.close_position(symbol, current_price, commission)
                    if trade:
                        system.trade_journal.log_close(trade)
                        system.performance_monitor.record_trade(trade)
                        system.order_executor.release_symbol(symbol)
                    continue

            # Strategy exit
            exit_signals = system.strategy_engine.check_exits(
                [symbol],
                {symbol: {
                    "strategy_id": pos.strategy_id,
                    "entry_price": pos.entry_price,
                    "current_price": current_price,
                }},
            )
            for _ in exit_signals:
                commission = system.order_executor.get_commission(current_price * pos.quantity)
                trade = system.portfolio_manager.close_position(symbol, current_price, commission)
                if trade:
                    system.trade_journal.log_close(trade)
                    system.performance_monitor.record_trade(trade)
                    system.order_executor.release_symbol(symbol)

        # Keep pair selector in sync with open positions
        system.pair_selector.set_protected_pairs(
            system.portfolio_manager.get_open_position_symbols()
        )

        # Evaluate strategy performance periodically
        if self._cycle_count % (self._heartbeat_cycles * 10) == 0:
            underperformers = system.performance_monitor.evaluate_strategies()
            for sid in underperformers:
                system.strategy_engine.disable_strategy(sid)

    def _log_heartbeat(self) -> None:
        if not self._system:
            return
        state = self._system.portfolio_manager.get_state()
        regime = self._system.regime_detector.current_regime
        pairs = self._system.pair_selector.active_pairs
        unrealized = state.total_unrealized_pnl
        logger.info(
            "[heartbeat] cycle=%d | regime=%s | positions=%d | capital=%.2f | "
            "realized=%.4f | unrealized=%.4f | equity=%.2f | pairs=%d",
            self._cycle_count, regime.name, state.open_position_count,
            state.total_capital, state.rolling_pnl, unrealized,
            state.total_capital + unrealized, len(pairs),
        )

    def stop(self) -> None:
        self._running = False
        logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running
