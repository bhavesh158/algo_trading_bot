"""Mean Reversion Strategy — Bollinger Band touch with RSI confirmation.

Entry: Price touches lower BB AND RSI < oversold threshold.
Exit:  Price returns to BB midline OR RSI > overbought OR stop/target hit.
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


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band reversion with RSI divergence, EMA trend filter, and volume confirmation."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("mean_reversion", config, market_data)
        sc = config.get("mean_reversion", {})
        self._rsi_oversold = sc.get("rsi_oversold", 25)
        self._rsi_overbought = sc.get("rsi_overbought", 75)
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 1.5)
        self._atr_target_mult = sc.get("atr_multiplier_target", 3.0)
        self._min_exit_profit_pct = sc.get("min_exit_profit_pct", 1.0) / 100  # 1.0%
        self._trend_filter_enabled = sc.get("trend_filter_enabled", True)
        self._rsi_capitulation_floor = sc.get("rsi_capitulation_floor", 15)
        self._rsi_capitulation_ceiling = sc.get("rsi_capitulation_ceiling", 85)
        self.primary_timeframe = "15m"

        # Trailing stop + max hold from base
        self._trailing_stop_atr = sc.get("trailing_stop_atr_multiplier", 1.5)
        self._max_hold_minutes = sc.get("max_hold_minutes", 120)

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < 25:
            return None

        close = df["close"].iloc[-1]
        bb_lower = df.get("bb_lower", pd.Series(dtype=float))
        rsi = df.get("rsi", pd.Series(dtype=float))
        atr = df.get("atr", pd.Series(dtype=float))

        if bb_lower.empty or rsi.empty or atr.empty:
            return None

        curr_bb_lower = bb_lower.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        curr_atr = atr.iloc[-1]

        if pd.isna(curr_bb_lower) or pd.isna(curr_rsi) or pd.isna(curr_atr) or curr_atr == 0:
            return None

        # RSI capitulation filter: skip free-falls and vertical pumps
        if curr_rsi < self._rsi_capitulation_floor or curr_rsi > self._rsi_capitulation_ceiling:
            logger.debug(
                "[mean_reversion] %s SKIP: RSI=%.1f in capitulation zone (<%d or >%d)",
                symbol, curr_rsi, self._rsi_capitulation_floor, self._rsi_capitulation_ceiling,
            )
            return None

        # EMA trend filter: don't buy dips in downtrends or sell rips in uptrends
        ema_trend_bullish = None
        if self._trend_filter_enabled:
            ema_9 = df.get("ema_9", pd.Series(dtype=float))
            ema_21 = df.get("ema_21", pd.Series(dtype=float))
            if not ema_9.empty and not ema_21.empty and not pd.isna(ema_9.iloc[-1]) and not pd.isna(ema_21.iloc[-1]):
                ema_trend_bullish = bool(ema_9.iloc[-1] > ema_21.iloc[-1])

        bb_upper = df.get("bb_upper", pd.Series(dtype=float))
        curr_bb_upper = bb_upper.iloc[-1] if not bb_upper.empty else 0

        logger.debug(
            "[mean_reversion] %s | close=%.4f | BB_low=%.4f BB_up=%.4f | RSI=%.1f (need<%d or >%d)",
            symbol, close, curr_bb_lower, curr_bb_upper, curr_rsi,
            self._rsi_oversold, self._rsi_overbought,
        )

        bb_mid = df["bb_mid"].iloc[-1] if "bb_mid" in df.columns else (curr_bb_lower + curr_bb_upper) / 2
        bb_width = curr_bb_upper - curr_bb_lower
        lower_zone = curr_bb_lower + bb_width * 0.25  # lower 25% of band
        upper_zone = curr_bb_upper - bb_width * 0.25  # upper 25% of band

        # BUY: price in lower 25% of BB + RSI deeply oversold
        if close <= lower_zone and curr_rsi < self._rsi_oversold:
            # Trend filter: don't buy dips in a downtrend
            if self._trend_filter_enabled and ema_trend_bullish is False:
                logger.debug(
                    "[mean_reversion] %s SKIP BUY: EMA downtrend (RSI=%.1f)",
                    symbol, curr_rsi,
                )
                return None

            stop = close - self._atr_stop_mult * curr_atr
            target = bb_mid + self._atr_target_mult * curr_atr * 0.5

            # Deeper oversold = higher confidence
            confidence = min(0.5 + (self._rsi_oversold - curr_rsi) / 30, 0.85)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=close,
                stop_loss=stop,
                target_price=target,
                metadata={"rsi": curr_rsi, "bb_lower": curr_bb_lower, "ema_trend": "bullish" if ema_trend_bullish else "bearish"},
            )

        # SELL: price in upper 25% of BB + RSI deeply overbought
        if close >= upper_zone and curr_rsi > self._rsi_overbought:
            # Trend filter: don't sell rips in an uptrend
            if self._trend_filter_enabled and ema_trend_bullish is True:
                logger.debug(
                    "[mean_reversion] %s SKIP SELL: EMA uptrend (RSI=%.1f)",
                    symbol, curr_rsi,
                )
                return None

            stop = close + self._atr_stop_mult * curr_atr
            target = bb_mid - self._atr_target_mult * curr_atr * 0.5

            confidence = min(0.5 + (curr_rsi - self._rsi_overbought) / 30, 0.85)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=close,
                stop_loss=stop,
                target_price=target,
                metadata={"rsi": curr_rsi, "bb_upper": curr_bb_upper, "ema_trend": "bullish" if ema_trend_bullish else "bearish"},
            )

        return None

    def should_exit(
        self, symbol: str, entry_price: float, current_price: float,
        side: OrderSide = OrderSide.BUY,
    ) -> bool:
        """Exit when RSI returns to neutral zone AND trade is in profit.

        This replaces the old BB_mid exit which triggered too quickly.
        RSI returning to ~50 means the extreme condition has genuinely reverted.
        Requiring profit ensures we don't exit a reversion that hasn't played out.
        """
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty:
            return False

        rsi = df.get("rsi", pd.Series(dtype=float))
        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return False

        curr_rsi = rsi.iloc[-1]

        if side == OrderSide.BUY:
            # Long: exit when RSI rises back above 50 AND profit exceeds min threshold
            # The threshold ensures we don't exit with crumbs that get eaten by fees/tax
            price_move = (current_price - entry_price) / entry_price
            in_profit = price_move > self._min_exit_profit_pct
            rsi_neutral = curr_rsi > 50
            return bool(in_profit and rsi_neutral)
        else:
            # Short: exit when RSI drops back below 50 AND meaningful profit
            price_move = (entry_price - current_price) / entry_price
            in_profit = price_move > self._min_exit_profit_pct
            rsi_neutral = curr_rsi < 50
            return bool(in_profit and rsi_neutral)
