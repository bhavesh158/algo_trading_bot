"""Trend Following Strategy — EMA crossover with ADX confirmation.

Entry: Fast EMA crosses above slow EMA AND ADX > threshold (trend present).
Exit:  Fast EMA crosses below slow EMA OR stop/target hit.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from crypto.core.enums import OrderSide, SignalStrength
from crypto.core.models import Signal
from crypto.data.market_data_engine import MarketDataEngine
from crypto.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """EMA crossover with ADX trend confirmation."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("trend_following", config, market_data)
        sc = config.get("trend_following", {})
        self._fast_ema = sc.get("fast_ema", 9)
        self._slow_ema = sc.get("slow_ema", 21)
        self._adx_threshold = sc.get("adx_threshold", 25)
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 2.0)
        self._atr_target_mult = sc.get("atr_multiplier_target", 3.0)
        self._trend_cont_adx = sc.get("trend_continuation_adx", 40)
        self._entered_symbols: set[str] = set()  # track to avoid duplicate entries
        self.primary_timeframe = "15m"

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < self._slow_ema + 5:
            return None

        ema_fast = df[f"ema_{self._fast_ema}"] if f"ema_{self._fast_ema}" in df.columns else df["ema_9"]
        ema_slow = df[f"ema_{self._slow_ema}"] if f"ema_{self._slow_ema}" in df.columns else df["ema_21"]
        adx = df.get("adx", pd.Series(dtype=float))
        atr = df.get("atr", pd.Series(dtype=float))

        if adx.empty or atr.empty:
            return None

        curr_adx = adx.iloc[-1]
        curr_atr = atr.iloc[-1]
        if pd.isna(curr_adx) or pd.isna(curr_atr) or curr_atr == 0:
            return None

        fast_now = ema_fast.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_now = ema_slow.iloc[-1]
        slow_prev = ema_slow.iloc[-2]
        bullish_cross = fast_now > slow_now and fast_prev <= slow_prev
        bearish_cross = fast_now < slow_now and fast_prev >= slow_prev

        # EMA separation as % of price
        ema_sep_pct = abs(fast_now - slow_now) / slow_now * 100 if slow_now > 0 else 0

        logger.debug(
            "[trend_following] %s | EMA9=%.4f EMA21=%.4f (sep=%.2f%%) | ADX=%.1f (need>%d) | "
            "bull_cross=%s bear_cross=%s",
            symbol, fast_now, slow_now, ema_sep_pct, curr_adx, self._adx_threshold,
            bullish_cross, bearish_cross,
        )

        price = float(df["close"].iloc[-1])

        # Bullish crossover
        if bullish_cross and curr_adx >= self._adx_threshold:
            self._entered_symbols.add(symbol)
            stop = price - self._atr_stop_mult * curr_atr
            target = price + self._atr_target_mult * curr_atr
            confidence = min(0.5 + (curr_adx - self._adx_threshold) / 50, 0.9)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=price,
                stop_loss=stop,
                target_price=target,
                metadata={"adx": curr_adx, "atr": curr_atr},
            )

        # Bearish crossover
        if bearish_cross and curr_adx >= self._adx_threshold:
            self._entered_symbols.add(symbol)
            stop = price + self._atr_stop_mult * curr_atr
            target = price - self._atr_target_mult * curr_atr
            confidence = min(0.5 + (curr_adx - self._adx_threshold) / 50, 0.9)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=price,
                stop_loss=stop,
                target_price=target,
                metadata={"adx": curr_adx, "atr": curr_atr},
            )

        # Trend continuation: strong existing trend, not yet entered
        if symbol not in self._entered_symbols and curr_adx >= self._trend_cont_adx:
            if fast_now > slow_now:  # bullish trend in progress
                stop = price - self._atr_stop_mult * curr_atr
                target = price + self._atr_target_mult * curr_atr
                confidence = min(0.45 + (curr_adx - self._trend_cont_adx) / 80, 0.75)
                self._entered_symbols.add(symbol)

                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    entry_price=price,
                    stop_loss=stop,
                    target_price=target,
                    metadata={"adx": curr_adx, "atr": curr_atr, "type": "continuation"},
                )
            elif fast_now < slow_now:  # bearish trend in progress
                stop = price + self._atr_stop_mult * curr_atr
                target = price - self._atr_target_mult * curr_atr
                confidence = min(0.45 + (curr_adx - self._trend_cont_adx) / 80, 0.75)
                self._entered_symbols.add(symbol)

                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    entry_price=price,
                    stop_loss=stop,
                    target_price=target,
                    metadata={"adx": curr_adx, "atr": curr_atr, "type": "continuation"},
                )

        return None

    def should_exit(
        self, symbol: str, entry_price: float, current_price: float,
        side: OrderSide = OrderSide.BUY,
    ) -> bool:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < self._slow_ema + 2:
            return False

        ema_fast = df.get("ema_9", pd.Series(dtype=float))
        ema_slow = df.get("ema_21", pd.Series(dtype=float))
        if ema_fast.empty or ema_slow.empty:
            return False

        fast_now = ema_fast.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_now = ema_slow.iloc[-1]
        slow_prev = ema_slow.iloc[-2]

        if side == OrderSide.BUY:
            # Long: exit on bearish crossover (fast crosses below slow)
            return bool(fast_now < slow_now and fast_prev >= slow_prev)
        else:
            # Short: exit on bullish crossover (fast crosses above slow)
            return bool(fast_now > slow_now and fast_prev <= slow_prev)
