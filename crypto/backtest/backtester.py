#!/usr/bin/env python3
"""Crypto Strategy Backtester.

Runs historical simulation of strategies using CCXT historical data.
Generates performance metrics including commissions, slippage, and tax.

Usage:
    python -m crypto.backtest.backtester --symbols BTC/USDT,ETH/USDT --days 30
    python -m crypto.backtest.backtester --config user_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from crypto.config.settings import load_config
from crypto.core.enums import OrderSide, PositionStatus, SignalStrength, Timeframe
from crypto.core.models import Position, Signal, Trade
from crypto.data.providers.ccxt_provider import CcxtProvider
from crypto.strategy.base_strategy import BaseStrategy
from crypto.strategy.strategies.trend_following import TrendFollowingStrategy
from crypto.strategy.strategies.mean_reversion import MeanReversionStrategy
from crypto.strategy.strategies.breakout_momentum import BreakoutMomentumStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on an OHLCV DataFrame."""
    if df.empty or len(df) < 2:
        return df

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["sma_20"] = close.rolling(20).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Bollinger Bands
    df["bb_mid"] = df["sma_20"]
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # ADX
    df["adx"] = _compute_adx(high, low, close, period=14)

    # Volume SMA
    df["volume_sma"] = volume.rolling(20).mean()

    return df


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index)."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Only keep the larger of the two
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return adx


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 1000.0
    commission_pct: float = 0.1  # 0.1% per side
    slippage_pct: float = 0.05   # 0.05% slippage
    profit_tax_pct: float = 30.0  # 30% tax on profits
    days_to_test: int = 30
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "15m"


@dataclass
class BacktestResult:
    """Backtest performance metrics."""
    total_return_pct: float
    total_return_abs: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    avg_hold_time_hours: float
    trades: list[Trade] = field(default_factory=list)


class MockMarketData:
    """Mock market data provider for backtesting."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._current_idx = 0

    def set_index(self, idx: int) -> None:
        self._current_idx = max(0, min(idx, len(self._df) - 1))

    def get_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # Return data up to current index (simulate real-time)
        return self._df.iloc[:self._current_idx + 1].copy()

    def get_current_price(self) -> float:
        if self._current_idx < len(self._df):
            return float(self._df["close"].iloc[self._current_idx])
        return 0.0


class BacktestEngine:
    """Single-symbol backtest engine."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._capital = config.initial_capital
        self._peak_capital = config.initial_capital
        self._position: Optional[Position] = None
        self._trades: list[Trade] = []
        self._equity_curve: list[float] = []
        self._daily_returns: list[float] = []

    def _apply_commission(self, notional: float) -> float:
        return notional * (self.config.commission_pct / 100)

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return price * (1 + self.config.slippage_pct / 100)
        return price * (1 - self.config.slippage_pct / 100)

    def _open_position(self, signal: Signal, price: float, timestamp: datetime) -> None:
        fill_price = self._apply_slippage(price, signal.side)
        commission = self._apply_commission(fill_price * signal.confidence)  # confidence used as qty proxy

        # Calculate quantity based on available capital (use 10% of capital per trade)
        position_value = self._capital * 0.10  # 10% position sizing
        quantity = position_value / fill_price

        self._position = Position(
            symbol=signal.symbol,
            side=signal.side,
            quantity=quantity,
            entry_price=fill_price,
            current_price=fill_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            strategy_id=signal.strategy_id,
            opened_at=timestamp,
        )

        self._capital -= self._apply_commission(fill_price * quantity)
        logger.debug(f"[{timestamp}] OPEN {signal.side.name} {signal.symbol} @ {fill_price:.4f}")

    def _close_position(self, price: float, timestamp: datetime, reason: str = "exit") -> None:
        if not self._position:
            return

        fill_price = self._apply_slippage(price, OrderSide.SELL if self._position.side == OrderSide.BUY else OrderSide.BUY)
        commission = self._apply_commission(fill_price * self._position.quantity)

        # Calculate P&L
        if self._position.side == OrderSide.BUY:
            gross_pnl = (fill_price - self._position.entry_price) * self._position.quantity
        else:
            gross_pnl = (self._position.entry_price - fill_price) * self._position.quantity

        # Apply tax on profits
        taxable = max(0.0, gross_pnl - commission)
        tax = taxable * (self.config.profit_tax_pct / 100)
        net_pnl = gross_pnl - commission - tax

        pnl_pct = (gross_pnl / (self._position.entry_price * self._position.quantity)) * 100

        trade = Trade(
            symbol=self._position.symbol,
            side=self._position.side,
            quantity=self._position.quantity,
            entry_price=self._position.entry_price,
            exit_price=fill_price,
            strategy_id=self._position.strategy_id,
            entry_time=self._position.opened_at,
            exit_time=timestamp,
            commission=commission,
            tax=tax,
        )

        self._trades.append(trade)
        self._capital += net_pnl
        self._position = None

        logger.debug(f"[{timestamp}] CLOSE {trade.symbol} | gross={gross_pnl:.4f} tax={tax:.4f} net={net_pnl:.4f} ({pnl_pct:.2f}%) [{reason}]")

    def run(self, df: pd.DataFrame, strategy: BaseStrategy, symbol: str) -> BacktestResult:
        """Run backtest on historical data."""
        if df.empty:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        self._capital = self.config.initial_capital
        self._peak_capital = self.config.initial_capital
        self._trades = []
        self._equity_curve = [self._capital]
        self._daily_returns = []

        mock_data = MockMarketData(df)
        strategy.market_data = mock_data

        prev_day = None
        prev_equity = self._capital

        for idx in range(100, len(df)):  # Skip first 100 bars for indicator warmup
            mock_data.set_index(idx)
            row = df.iloc[idx]
            timestamp = df.index[idx] if hasattr(df.index[idx], 'to_pydatetime') else datetime.now()
            if hasattr(df.index[idx], 'to_pydatetime'):
                timestamp = df.index[idx].to_pydatetime()
            else:
                timestamp = datetime.fromtimestamp(df.index[idx]) if isinstance(df.index[idx], (int, float)) else datetime.now(timezone.utc)

            current_price = float(row["close"])

            # Track daily returns
            current_day = timestamp.date() if hasattr(timestamp, 'date') else timestamp
            if prev_day and current_day != prev_day:
                daily_ret = (self._capital - prev_equity) / prev_equity
                self._daily_returns.append(daily_ret)
                prev_equity = self._capital
            prev_day = current_day

            # Check for entry signal
            if not self._position:
                signal = strategy.analyze(symbol)
                if signal and signal.confidence >= 0.55:
                    self._open_position(signal, current_price, timestamp)
            else:
                # Update position price
                self._position.current_price = current_price

                # Check stop loss
                if self._position.side == OrderSide.BUY:
                    if current_price <= self._position.stop_loss:
                        self._close_position(current_price, timestamp, "stop_loss")
                        continue
                else:
                    if current_price >= self._position.stop_loss:
                        self._close_position(current_price, timestamp, "stop_loss")
                        continue

                # Check target
                if self._position.side == OrderSide.BUY:
                    if current_price >= self._position.target_price:
                        self._close_position(current_price, timestamp, "target")
                        continue
                else:
                    if current_price <= self._position.target_price:
                        self._close_position(current_price, timestamp, "target")
                        continue

                # Check strategy exit
                if strategy.should_exit(symbol, self._position.entry_price, current_price, self._position.side):
                    self._close_position(current_price, timestamp, "strategy_exit")
                    continue

            self._equity_curve.append(self._capital)
            self._peak_capital = max(self._peak_capital, self._capital)

        # Close any open position at end
        if self._position and len(df) > 100:
            final_price = float(df["close"].iloc[-1])
            self._close_position(final_price, datetime.now(timezone.utc), "end_of_test")

        return self._calculate_metrics()

    def _calculate_metrics(self) -> BacktestResult:
        if not self._trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        total_return = self._capital - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100

        # Win/Loss stats
        winning = [t for t in self._trades if t.pnl > 0]
        losing = [t for t in self._trades if t.pnl <= 0]

        win_rate = len(winning) / len(self._trades) * 100 if self._trades else 0

        gross_wins = sum(t.pnl for t in winning)
        gross_losses = abs(sum(t.pnl for t in losing))

        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        avg_win_pct = sum(t.pnl_pct for t in winning) / len(winning) if winning else 0
        avg_loss_pct = sum(t.pnl_pct for t in losing) / len(losing) if losing else 0

        # Max drawdown
        max_dd = 0
        peak = self.config.initial_capital
        for equity in self._equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)

        # Sharpe ratio (annualized)
        if len(self._daily_returns) > 1:
            import numpy as np
            sharpe = (np.mean(self._daily_returns) / np.std(self._daily_returns)) * np.sqrt(252) if np.std(self._daily_returns) > 0 else 0
        else:
            sharpe = 0

        # Average hold time
        hold_times = []
        for t in self._trades:
            if t.entry_time and t.exit_time:
                try:
                    delta = t.exit_time - t.entry_time
                    hold_times.append(delta.total_seconds() / 3600)
                except:
                    pass
        avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0

        return BacktestResult(
            total_return_pct=total_return_pct,
            total_return_abs=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            total_trades=len(self._trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            avg_hold_time_hours=avg_hold,
            trades=self._trades,
        )


def load_historical_data(provider: CcxtProvider, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch historical OHLCV data from exchange using pagination."""
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}...")
    
    # Calculate candles needed
    tf_minutes = int(timeframe.rstrip('m'))
    candles_per_day = (24 * 60) // tf_minutes
    total_candles_needed = days * candles_per_day
    
    # CCXT max limit is typically 500-1000 per request, so we paginate
    all_candles = []
    limit = 500  # Safe limit for most exchanges
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    
    current_since = since_ms
    max_iterations = (total_candles_needed // limit) + 2
    
    for _ in range(max_iterations):
        df = provider.fetch_ohlcv(symbol, timeframe, limit=limit, since_ms=current_since)
        if df.empty:
            break
        
        all_candles.append(df)
        
        # If we got fewer than limit, we've reached the end (most recent data)
        if len(df) < limit:
            break
        
        # Update since to get the next batch (most recent first, so we go forward)
        last_timestamp = int(df.index[-1].timestamp() * 1000)
        current_since = last_timestamp + 1
        
        # Check if we have enough data
        total_collected = sum(len(c) for c in all_candles)
        if total_collected >= total_candles_needed:
            break
    
    if all_candles:
        result = pd.concat(all_candles).drop_duplicates()
        logger.info(f"Fetched {len(result)} candles for {symbol}")
        return result
    return pd.DataFrame()


def run_backtest(config: BacktestConfig) -> dict[str, BacktestResult]:
    """Run backtest across all symbols."""
    provider = CcxtProvider("binance", {})
    provider.connect()

    results = {}

    for symbol in config.symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtesting: {symbol}")
        logger.info(f"{'='*60}")

        df = load_historical_data(provider, symbol, config.timeframe, config.days_to_test)

        if df.empty:
            logger.warning(f"No data for {symbol}")
            continue

        logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        # Compute technical indicators
        logger.info("Computing technical indicators...")
        df = compute_indicators(df)

        # Create strategy instances
        strategy_config = {
            "trend_following": {},
            "mean_reversion": {},
            "breakout_momentum": {},
        }

        strategies = [
            TrendFollowingStrategy(strategy_config, None),
            MeanReversionStrategy(strategy_config, None),
            BreakoutMomentumStrategy(strategy_config, None),
        ]

        for strategy in strategies:
            engine = BacktestEngine(config)
            result = engine.run(df.copy(), strategy, symbol)

            logger.info(f"\n{strategy.strategy_id.upper()} Results:")
            logger.info(f"  Total Return: {result.total_return_pct:.2f}% (${result.total_return_abs:.2f})")
            logger.info(f"  Win Rate: {result.win_rate:.1f}% ({result.winning_trades}/{result.total_trades})")
            logger.info(f"  Profit Factor: {result.profit_factor:.2f}")
            logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
            logger.info(f"  Avg Win: {result.avg_win_pct:.2f}% | Avg Loss: {result.avg_loss_pct:.2f}%")
            logger.info(f"  Avg Hold: {result.avg_hold_time_hours:.1f} hours")

            results[f"{symbol}_{strategy.strategy_id}"] = result

    return results


def main():
    parser = argparse.ArgumentParser(description="Crypto Strategy Backtester")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT",
                        help="Comma-separated symbols to test")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe (5m, 15m, 1h)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital")
    parser.add_argument("--tax", type=float, default=30.0, help="Profit tax percentage")
    parser.add_argument("--config", type=str, help="Path to config YAML")

    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}

    bt_config = BacktestConfig(
        initial_capital=args.capital,
        profit_tax_pct=args.tax,
        days_to_test=args.days,
        symbols=args.symbols.split(","),
        timeframe=args.timeframe,
    )

    logger.info("="*60)
    logger.info("  CRYPTO STRATEGY BACKTESTER")
    logger.info("="*60)
    logger.info(f"Capital: ${bt_config.initial_capital}")
    logger.info(f"Symbols: {', '.join(bt_config.symbols)}")
    logger.info(f"Days: {bt_config.days_to_test}")
    logger.info(f"Timeframe: {bt_config.timeframe}")
    logger.info(f"Tax: {bt_config.profit_tax_pct}%")
    logger.info("="*60)

    results = run_backtest(bt_config)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("  SUMMARY")
    logger.info("="*60)

    for key, result in results.items():
        status = "✓ PROFITABLE" if result.total_return_abs > 0 else "✗ LOSS"
        logger.info(f"{key}: {result.total_return_pct:.2f}% [{status}]")


if __name__ == "__main__":
    main()
