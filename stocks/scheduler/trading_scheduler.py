"""Trading Scheduler / Session Controller (PRD §20).

Controls the trading day lifecycle:
1. Pre-market: analysis, watchlist building, strategy preparation
2. Market hours: data monitoring, signal generation, trade execution
3. Pre-close: close positions, cancel orders, generate reports

The scheduler runs the main loop and delegates to components.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, time, timedelta
from typing import Any

from stocks.core.enums import OrderSide, OrderType, TradingPhase
from stocks.core.event_bus import EventBus
from stocks.core.events import ScheduleEvent
from stocks.core.models import Order
from datetime import timedelta

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Manages the intraday trading session lifecycle."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus

        sched = config.get("schedule", {})
        self._pre_market_start = self._parse_time(sched.get("pre_market_start", "08:45"))
        self._market_open = self._parse_time(sched.get("market_open", "09:15"))
        self._pre_close_start = self._parse_time(sched.get("pre_close_start", "15:15"))
        self._market_close = self._parse_time(sched.get("market_close", "15:30"))
        self._post_market_end = self._parse_time(sched.get("post_market_end", "16:00"))

        self._update_interval = config.get("market_data", {}).get("update_interval_seconds", 5)
        self._current_phase = TradingPhase.MARKET_CLOSED
        self._running = False
        self._cycle_count = 0
        self._heartbeat_interval = 12  # Log status every 12 cycles (~60s at 5s interval)
        self._last_trading_date = None  # Track date for daily resets

        # Anti-churn / scheduler guards
        sched_config = config.get("scheduler", {})
        self._max_entries_per_cycle = sched_config.get("max_entries_per_cycle", 1)
        self._min_expected_profit_pct = sched_config.get("min_expected_profit_pct", 0.5)
        self._max_open_positions = config.get("risk", {}).get("max_open_positions", 3)

        # Post-close cooldown: prevent re-entering a symbol for N minutes after close
        self._cooldown_minutes = sched_config.get("trade_cooldown_minutes", 15)
        self._symbol_cooldown: dict[str, datetime] = {}  # symbol -> cooldown_end

        logger.info("TradingScheduler initialized (open=%s, close=%s)",
                     self._market_open.strftime("%H:%M"), self._market_close.strftime("%H:%M"))

    @staticmethod
    def _parse_time(time_str: str) -> time:
        """Parse HH:MM string into a time object."""
        return datetime.strptime(time_str, "%H:%M").time()

    def _get_current_phase(self) -> TradingPhase:
        """Determine the current trading phase based on clock time."""
        now = datetime.now()
        today = now.date()

        # Reset daily flags when the date rolls over
        if self._last_trading_date != today:
            if self._last_trading_date is not None:
                logger.info("New trading day detected (%s -> %s), resetting daily state",
                            self._last_trading_date, today)
            self._last_trading_date = today
            self._pre_market_done = False
            self._pre_close_done = False
            self._cycle_count = 0

        # Compare using today's date so boundaries are always current
        pre_market_start = datetime.combine(today, self._pre_market_start)
        market_open = datetime.combine(today, self._market_open)
        pre_close_start = datetime.combine(today, self._pre_close_start)
        market_close = datetime.combine(today, self._market_close)

        if now < pre_market_start:
            return TradingPhase.MARKET_CLOSED
        elif now < market_open:
            return TradingPhase.PRE_MARKET
        elif market_open <= now < market_open + timedelta(minutes=5):
            return TradingPhase.MARKET_OPEN
        elif now < pre_close_start:
            return TradingPhase.MARKET_HOURS
        elif now < market_close:
            return TradingPhase.PRE_CLOSE
        else:
            return TradingPhase.MARKET_CLOSED

    def set_trading_system(self, trading_system: Any) -> None:
        """Set reference to the parent TradingSystem for orchestration."""
        self._system = trading_system

    def run(self, components: list[Any]) -> None:
        """Main loop: runs the trading day from pre-market through close."""
        self._running = True
        self._pre_market_done = False
        self._pre_close_done = False
        logger.info("Scheduler starting main loop...")

        while self._running:
            try:
                new_phase = self._get_current_phase()

                # Handle phase transitions
                if new_phase != self._current_phase:
                    self._on_phase_transition(new_phase)

                # Execute phase logic
                self._execute_phase()

                # Sleep until next cycle
                _time.sleep(self._update_interval)

            except KeyboardInterrupt:
                logger.info("Scheduler interrupted")
                self._running = False
            except Exception:
                logger.exception("Error in scheduler main loop")
                _time.sleep(self._update_interval)

    def _on_phase_transition(self, new_phase: TradingPhase) -> None:
        """Handle transition from one phase to another."""
        old_phase = self._current_phase
        self._current_phase = new_phase
        logger.info("Phase transition: %s -> %s", old_phase.name, new_phase.name)
        self.event_bus.publish(ScheduleEvent(phase=new_phase))

    def _execute_phase(self) -> None:
        """Delegate to the TradingSystem based on the current phase."""
        if not hasattr(self, '_system') or self._system is None:
            return

        system = self._system

        if self._current_phase == TradingPhase.PRE_MARKET:
            if not self._pre_market_done:
                self._run_pre_market(system)
                self._pre_market_done = True

        elif self._current_phase in (TradingPhase.MARKET_OPEN, TradingPhase.MARKET_HOURS):
            # If we joined mid-session, run pre-market setup first
            if not self._pre_market_done:
                logger.info("Late start detected — running pre-market setup before trading")
                self._run_pre_market(system)
                self._pre_market_done = True
                return  # Start trading on the next cycle
            self._run_market_hours(system)

        elif self._current_phase == TradingPhase.PRE_CLOSE:
            if not self._pre_close_done:
                self._run_pre_close(system)
                self._pre_close_done = True

    def _run_pre_market(self, system: Any) -> None:
        """Pre-market phase: load data, build watchlist, prepare strategies."""
        logger.info("--- PRE-MARKET PHASE ---")

        # Reset daily state
        system.portfolio_manager.reset_daily_state()
        system.risk_manager.reset_daily_state()

        # Load daily data for stock selection scoring
        from stocks.selection.stock_selector import NIFTY50_SYMBOLS
        logger.info("Loading daily data for %d candidates...", len(NIFTY50_SYMBOLS))
        system.market_data_engine.load_daily_data(NIFTY50_SYMBOLS)

        # Build watchlist
        system.stock_selector.build_watchlist()
        watchlist = system.stock_selector.watchlist

        # Load intraday data for watchlist + index symbols for regime detection
        index_symbols = self.config.get("selection", {}).get("index_symbols", ["^NSEI"])
        system.market_data_engine.load_historical_data(watchlist + index_symbols)

        # Detect initial market regime
        system.regime_detector.set_market_data(system.market_data_engine)
        system.regime_detector.detect_regime()

        # Initialize analysis components
        system.ai_analysis.set_market_data(system.market_data_engine)
        system.multi_tf.set_market_data(system.market_data_engine)
        system.news_monitor.set_market_data(system.market_data_engine)

        # Load strategies
        system.strategy_engine.load_strategies(system.market_data_engine)

        # Wire report generator
        system.report_generator.set_dependencies(
            system.portfolio_manager, system.performance_monitor
        )

        # Initial macro refresh at pre-market (non-blocking)
        if hasattr(system, "macro_analyst") and system.macro_analyst is not None:
            try:
                system.macro_analyst.refresh()
            except Exception:
                logger.exception("Pre-market macro refresh failed — continuing")

        logger.info("Pre-market complete. Watchlist: %d symbols", len(watchlist))

    def _run_market_hours(self, system: Any) -> None:
        """Market hours: update data, run strategies, execute trades."""
        self._cycle_count += 1

        # Check if trading should be paused
        if not system.news_monitor.check_conditions():
            return
        if system.drawdown_monitor.is_trading_paused:
            return
        if system.risk_manager.is_daily_loss_breached:
            return

        watchlist = system.stock_selector.watchlist
        if not watchlist:
            return

        # Update market data (include index symbols for regime detection)
        index_symbols = self.config.get("selection", {}).get("index_symbols", ["^NSEI"])
        all_symbols = list(set(watchlist + index_symbols))
        system.market_data_engine.update_data(all_symbols)

        # Periodic heartbeat
        if self._cycle_count % self._heartbeat_interval == 0:
            state = system.portfolio_manager.get_state()
            regime = system.regime_detector.current_regime
            logger.info(
                "[heartbeat] cycle=%d | regime=%s | positions=%d | capital=%.0f | daily_pnl=%.2f | scanning %d symbols",
                self._cycle_count, regime.name, state.open_position_count,
                state.total_capital, state.daily_pnl, len(watchlist),
            )

        # Update position prices
        for symbol, pos in system.portfolio_manager.get_open_positions().items():
            price = system.market_data_engine.get_current_price(symbol)
            if price > 0:
                system.portfolio_manager.update_position_price(symbol, price)

        # Check drawdown
        dd_level = system.drawdown_monitor.check_drawdown()
        if dd_level == "reduce_size":
            system.position_sizer.set_drawdown_factor(
                system.drawdown_monitor.size_reduction_factor
            )

        # Detect market regime
        system.regime_detector.detect_regime()

        # Periodic macro refresh during market hours
        if hasattr(system, "macro_analyst") and system.macro_analyst is not None:
            if system.macro_analyst.should_refresh():
                try:
                    system.macro_analyst.refresh()
                except Exception:
                    logger.exception("Intraday macro refresh failed — keeping previous context")

        # Run strategies to generate signals
        signals = system.strategy_engine.run_strategies(watchlist)

        # Execute signals (with per-entry position limit + throttle)
        entries_this_cycle = 0
        commission_per_trade = system.order_executor.commission

        for signal in signals:
            # Hard limit: re-check position count before EACH entry
            open_count = len(system.portfolio_manager.get_open_positions())
            if open_count >= self._max_open_positions:
                break

            # Throttle: max entries per cycle
            if entries_this_cycle >= self._max_entries_per_cycle:
                break

            # Profitability filter: reject signals where target doesn't cover fees
            if signal.target_price and signal.entry_price > 0:
                if signal.side == OrderSide.BUY:
                    expected_pct = (signal.target_price - signal.entry_price) / signal.entry_price * 100
                else:
                    expected_pct = (signal.entry_price - signal.target_price) / signal.entry_price * 100
                if expected_pct < self._min_expected_profit_pct:
                    logger.debug(
                        "Signal filtered (low profit): %s %s expected=%.2f%% < %.2f%%",
                        signal.strategy_id, signal.symbol, expected_pct, self._min_expected_profit_pct,
                    )
                    continue

            # Commission-aware filter: expected gross profit must exceed 2× round-trip commission
            if signal.target_price and signal.entry_price > 0 and commission_per_trade > 0:
                quantity = system.position_sizer.calculate_quantity(signal)
                if quantity > 0:
                    if signal.side == OrderSide.BUY:
                        expected_gross = (signal.target_price - signal.entry_price) * quantity
                    else:
                        expected_gross = (signal.entry_price - signal.target_price) * quantity
                    round_trip_comm = commission_per_trade * 2
                    if expected_gross < round_trip_comm * 2:
                        logger.debug(
                            "Signal filtered (comm drag): %s %s expected_gross=%.2f < 2×comm=%.2f",
                            signal.strategy_id, signal.symbol, expected_gross, round_trip_comm * 2,
                        )
                        continue

            # Skip if symbol is in post-close cooldown
            cooldown_end = self._symbol_cooldown.get(signal.symbol)
            if cooldown_end is not None:
                if datetime.now() < cooldown_end:
                    logger.debug(
                        "Signal suppressed (cooldown): %s until %s",
                        signal.symbol, cooldown_end.strftime("%H:%M:%S"),
                    )
                    continue
                else:
                    del self._symbol_cooldown[signal.symbol]

            # Skip if already have a position in this symbol
            if signal.symbol in system.portfolio_manager.get_open_position_symbols():
                continue

            quantity = system.position_sizer.calculate_quantity(signal)
            if quantity <= 0:
                continue

            order = Order(
                symbol=signal.symbol,
                side=signal.side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=signal.entry_price,
                stop_price=signal.stop_loss,
                strategy_id=signal.strategy_id,
                signal_id=signal.id,
            )

            current_price = system.market_data_engine.get_current_price(signal.symbol)
            filled_order = system.order_executor.execute_order(order, current_price)

            if filled_order.status.name == "FILLED":
                entry_comm = system.order_executor.commission
                pos = system.portfolio_manager.open_position(filled_order, entry_commission=entry_comm)
                # Propagate target_price and max hold from signal to position
                if signal.target_price > 0:
                    pos.target_price = signal.target_price
                max_hold = signal.metadata.get("max_hold_minutes", 0)
                if max_hold > 0:
                    pos.max_hold_minutes = max_hold
                system.trade_journal.log_open(filled_order, pos)
                entries_this_cycle += 1

        # Check exit conditions for open positions
        open_positions = system.portfolio_manager.get_open_positions()
        for symbol, pos in list(open_positions.items()):
            current_price = system.market_data_engine.get_current_price(symbol)

            # Update position price extremes for trailing stops
            if current_price > 0:
                pos.update_extremes(current_price)

            exit_signals = system.strategy_engine.check_exits(
                [symbol],
                {symbol: {
                    "strategy_id": pos.strategy_id,
                    "entry_price": pos.entry_price,
                    "current_price": current_price,
                    "position": pos,  # Pass full Position for trailing/time exits
                }},
            )
            for exit_sig in exit_signals:
                exit_comm = system.order_executor.commission
                exit_reason = exit_sig.metadata.get("exit_reason", "strategy")

                # Commission-aware holdback: if the strategy-target exit would net a
                # loss after round-trip fees, hold a bit longer so price can run past
                # the target. Always exit on time / deep-red / non-target exits.
                if exit_reason == "strategy" and exit_comm > 0:
                    round_trip_comm = exit_comm * 2
                    gross_pnl = pos.unrealized_pnl  # direction-aware
                    if gross_pnl < round_trip_comm:
                        # Hard bail-out: position has gone well into the red
                        deeply_negative = gross_pnl < -round_trip_comm
                        # Bail-out: nearly expired (>80% of max hold elapsed)
                        near_expiry = (
                            pos.max_hold_minutes > 0
                            and pos.hold_duration_minutes >= pos.max_hold_minutes * 0.80
                        )
                        if not deeply_negative and not near_expiry:
                            logger.debug(
                                "Holding %s — gross %.2f < round-trip comm %.2f "
                                "(held %.0f/%.0f min)",
                                symbol, gross_pnl, round_trip_comm,
                                pos.hold_duration_minutes, pos.max_hold_minutes or 0,
                            )
                            continue

                trade = system.portfolio_manager.close_position(
                    symbol, current_price, exit_comm
                )
                if trade:
                    exit_reason = exit_sig.metadata.get("exit_reason", "strategy")
                    logger.info(
                        "Position closed [%s]: %s net=%.2f (reason=%s)",
                        pos.strategy_id, symbol, trade.pnl, exit_reason,
                    )
                    system.trade_journal.log_close(trade)
                    system.performance_monitor.record_trade(trade)
                    system.order_executor.release_symbol(symbol)
                    # Apply cooldown: prevent re-entering this symbol for N minutes
                    self._symbol_cooldown[symbol] = datetime.now() + timedelta(
                        minutes=self._cooldown_minutes
                    )
                    logger.debug(
                        "Cooldown set for %s: %d min", symbol, self._cooldown_minutes
                    )

        # Evaluate strategy performance periodically
        underperforming = system.performance_monitor.evaluate_strategies()
        for sid in underperforming:
            system.strategy_engine.disable_strategy(sid)

    def _run_pre_close(self, system: Any) -> None:
        """Pre-close phase: close positions, cancel orders, generate report."""
        logger.info("--- PRE-CLOSE PHASE ---")

        # Close all open positions (with commission)
        def _get_exit_comm(sym: str, notional: float) -> float:
            return system.order_executor.commission

        system.portfolio_manager.close_all_positions(
            lambda sym: system.market_data_engine.get_current_price(sym),
            exit_commission_fn=_get_exit_comm,
        )

        # Cancel pending orders
        system.order_executor.cancel_all_pending()

        # Generate daily report
        system.report_generator.generate_daily_report()

        logger.info("Pre-close complete. System will idle until next session.")

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        logger.info("Scheduler stopped")

    @property
    def current_phase(self) -> TradingPhase:
        return self._current_phase

    @property
    def is_market_hours(self) -> bool:
        return self._current_phase in (
            TradingPhase.MARKET_OPEN, TradingPhase.MARKET_HOURS
        )

    @property
    def is_running(self) -> bool:
        return self._running
