"""Mean Reversion Strategy (Enhanced).

Trades when price deviates significantly from its moving average (measured by
z-score of the Bollinger Bands). Enters when price is oversold and exits when
it reverts to the mean.

Enhancements over original:
- Higher-timeframe trend filter: Only BUY in uptrend/sideways, never in downtrend
- VWAP filter: Only BUY below VWAP (genuine undervaluation)
- RSI capitulation floor: Skip if RSI < 20 (falling knife)
- Min exit profit: Don't exit until gross profit covers fees
- Max hold duration: Force exit to prevent prolonged bleeding
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from stocks.core.enums import OrderSide, SignalStrength, Timeframe
from stocks.core.models import Signal
from stocks.data.market_data_engine import MarketDataEngine
from stocks.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Buy when price drops below lower Bollinger Band; exit at mean.

    Filtered by trend alignment, VWAP, and RSI to avoid buying dips
    in actively downtrending stocks.
    """

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("mean_reversion", config, market_data)

        strat_config = config.get("mean_reversion", {})
        self._lookback = strat_config.get("lookback_period", 20)
        self._entry_zscore = strat_config.get("entry_zscore", -2.2)
        self._exit_zscore = strat_config.get("exit_zscore", 0.0)
        self._stop_zscore = strat_config.get("stop_zscore", -3.0)
        self._min_exit_profit_pct = strat_config.get("min_exit_profit_pct", 0.5)
        self._require_trend = strat_config.get("require_trend_alignment", True)
        self._require_vwap_below = strat_config.get("require_vwap_below", True)
        self._rsi_floor = strat_config.get("rsi_capitulation_floor", 20)
        self._max_hold_minutes = strat_config.get("max_hold_minutes", 90)
        self.primary_timeframe = Timeframe.M5

    def _is_trend_aligned(self, symbol: str) -> bool:
        """Check that higher-timeframe trend is NOT bearish.

        Uses 15m EMA9 vs EMA21. Returns True if trend is up or sideways.
        """
        ema_9 = self.market_data.get_indicator(symbol, Timeframe.M15, "ema_9")
        ema_21 = self.market_data.get_indicator(symbol, Timeframe.M15, "ema_21")

        if ema_9 is None or ema_21 is None or ema_9.empty or ema_21.empty:
            return True  # No data — allow (don't block on missing data)

        ema_9_val = float(ema_9.iloc[-1])
        ema_21_val = float(ema_21.iloc[-1])

        if np.isnan(ema_9_val) or np.isnan(ema_21_val):
            return True

        # Block if clearly bearish (EMA9 meaningfully below EMA21)
        if ema_21_val > 0:
            gap_pct = (ema_9_val - ema_21_val) / ema_21_val * 100
            if gap_pct < -0.15:  # EMA9 more than 0.15% below EMA21 = downtrend
                logger.debug(
                    "[%s] Trend filter blocked %s: EMA9=%.2f < EMA21=%.2f (gap=%.2f%%)",
                    self.strategy_id, symbol, ema_9_val, ema_21_val, gap_pct,
                )
                return False

        return True

    def _is_below_vwap(self, symbol: str, price: float) -> bool:
        """Check that price is below VWAP (genuine undervaluation)."""
        vwap = self.market_data.get_indicator(symbol, self.primary_timeframe, "vwap")
        if vwap is None or vwap.empty or np.isnan(vwap.iloc[-1]):
            return True  # No VWAP data — allow

        vwap_val = float(vwap.iloc[-1])
        return price < vwap_val

    def _check_rsi_floor(self, symbol: str) -> bool:
        """Return False if RSI is in capitulation territory (< floor)."""
        rsi = self.market_data.get_indicator(symbol, self.primary_timeframe, "rsi_14")
        if rsi is None or rsi.empty or np.isnan(rsi.iloc[-1]):
            return True

        rsi_val = float(rsi.iloc[-1])
        if rsi_val < self._rsi_floor:
            logger.debug(
                "[%s] RSI capitulation filter blocked %s: RSI=%.1f < %d",
                self.strategy_id, symbol, rsi_val, self._rsi_floor,
            )
            return False
        return True

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is None or len(df) < self._lookback:
            return None

        close = df["close"]
        sma = close.rolling(window=self._lookback).mean()
        std = close.rolling(window=self._lookback).std()

        if std.iloc[-1] == 0 or np.isnan(std.iloc[-1]):
            return None

        current_price = float(close.iloc[-1])
        current_sma = float(sma.iloc[-1])
        current_std = float(std.iloc[-1])
        zscore = (current_price - current_sma) / current_std

        # Entry condition: price significantly below mean
        if zscore <= self._entry_zscore:
            # --- FILTER 1: Higher-timeframe trend alignment ---
            if self._require_trend and not self._is_trend_aligned(symbol):
                return None

            # --- FILTER 2: Price must be below VWAP ---
            if self._require_vwap_below and not self._is_below_vwap(symbol, current_price):
                logger.debug(
                    "[%s] VWAP filter blocked %s: price above VWAP",
                    self.strategy_id, symbol,
                )
                return None

            # --- FILTER 3: RSI not in capitulation ---
            if not self._check_rsi_floor(symbol):
                return None

            # Calculate stop and target using z-scores
            stop_loss = current_sma + self._stop_zscore * current_std
            target = current_sma  # Revert to mean

            # Ensure target is far enough to cover commissions
            expected_move_pct = (target - current_price) / current_price * 100
            if expected_move_pct < self._min_exit_profit_pct:
                logger.debug(
                    "[%s] Expected move too small for %s: %.2f%% < %.2f%%",
                    self.strategy_id, symbol, expected_move_pct, self._min_exit_profit_pct,
                )
                return None

            # Confidence scales with how extreme the deviation is
            confidence = min(abs(zscore) / 3.0, 1.0)

            if zscore <= self._entry_zscore * 1.5:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE

            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target,
                metadata={
                    "zscore": zscore,
                    "sma": current_sma,
                    "max_hold_minutes": self._max_hold_minutes,
                },
            )
            logger.info(
                "[%s] Mean reversion signal: %s zscore=%.2f price=%.2f sma=%.2f (trend✓ vwap✓ rsi✓)",
                self.strategy_id, symbol, zscore, current_price, current_sma,
            )
            return signal

        return None

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is None or len(df) < self._lookback:
            return False

        close = df["close"]
        sma = close.rolling(window=self._lookback).mean()
        std = close.rolling(window=self._lookback).std()

        if std.iloc[-1] == 0 or np.isnan(std.iloc[-1]):
            return False

        zscore = (current_price - float(sma.iloc[-1])) / float(std.iloc[-1])

        # Exit when price reverts to mean — BUT only if profit covers fees
        if zscore >= self._exit_zscore:
            gross_profit_pct = (current_price - entry_price) / entry_price * 100
            if gross_profit_pct >= self._min_exit_profit_pct:
                return True
            # If at mean but profit too small, hold a bit longer (up to time exit)
            logger.debug(
                "[%s] %s at mean (z=%.2f) but profit %.2f%% < min %.2f%%, holding",
                self.strategy_id, symbol, zscore, gross_profit_pct, self._min_exit_profit_pct,
            )
            # However, if price is ABOVE mean (overshoot), take profit regardless
            if zscore >= 0.5:
                return True

        return False
