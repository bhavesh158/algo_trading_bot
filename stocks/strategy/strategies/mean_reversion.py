"""Mean Reversion Strategy.

Trades when price deviates significantly from its moving average (measured by
z-score of the Bollinger Bands). Enters when price is oversold and exits when
it reverts to the mean.
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
    """Buy when price drops below lower Bollinger Band; exit at mean."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("mean_reversion", config, market_data)

        strat_config = config.get("mean_reversion", {})
        self._lookback = strat_config.get("lookback_period", 20)
        self._entry_zscore = strat_config.get("entry_zscore", -2.0)
        self._exit_zscore = strat_config.get("exit_zscore", 0.0)
        self._stop_zscore = strat_config.get("stop_zscore", -3.0)
        self.primary_timeframe = Timeframe.M5

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
            # Calculate stop and target using z-scores
            stop_loss = current_sma + self._stop_zscore * current_std
            target = current_sma  # Revert to mean

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
                metadata={"zscore": zscore, "sma": current_sma},
            )
            logger.info(
                "[%s] Mean reversion signal: %s zscore=%.2f price=%.2f sma=%.2f",
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

        # Exit when price reverts to mean
        return zscore >= self._exit_zscore
