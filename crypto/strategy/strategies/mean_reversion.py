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
    """Bollinger Band reversion with RSI divergence and volume confirmation."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("mean_reversion", config, market_data)
        sc = config.get("mean_reversion", {})
        self._rsi_oversold = sc.get("rsi_oversold", 25)
        self._rsi_overbought = sc.get("rsi_overbought", 75)
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 1.5)
        self._atr_target_mult = sc.get("atr_multiplier_target", 3.0)
        self._vol_confirm_mult = sc.get("volume_confirm_mult", 1.2)
        self.primary_timeframe = "15m"

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

        bb_upper = df.get("bb_upper", pd.Series(dtype=float))
        curr_bb_upper = bb_upper.iloc[-1] if not bb_upper.empty else 0

        # Volume confirmation: current candle volume must exceed average
        volume = df.get("volume", pd.Series(dtype=float))
        vol_sma = df.get("volume_sma", pd.Series(dtype=float))
        if not volume.empty and not vol_sma.empty:
            curr_vol = volume.iloc[-1]
            avg_vol = vol_sma.iloc[-1]
            if avg_vol > 0 and curr_vol < self._vol_confirm_mult * avg_vol:
                return None  # insufficient volume — no conviction

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
            stop = close - self._atr_stop_mult * curr_atr
            target = bb_mid + self._atr_target_mult * curr_atr * 0.5

            # Deeper oversold = higher confidence
            confidence = min(0.5 + (self._rsi_oversold - curr_rsi) / 50, 0.85)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=close,
                stop_loss=stop,
                target_price=target,
                metadata={"rsi": curr_rsi, "bb_lower": curr_bb_lower},
            )

        # SELL: price in upper 25% of BB + RSI deeply overbought
        if close >= upper_zone and curr_rsi > self._rsi_overbought:
            stop = close + self._atr_stop_mult * curr_atr
            target = bb_mid - self._atr_target_mult * curr_atr * 0.5

            confidence = min(0.5 + (curr_rsi - self._rsi_overbought) / 50, 0.85)

            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                entry_price=close,
                stop_loss=stop,
                target_price=target,
                metadata={"rsi": curr_rsi, "bb_upper": curr_bb_upper},
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
            # Long: exit when RSI rises back above 50 (mean reverted) AND in profit
            in_profit = current_price > entry_price
            rsi_neutral = curr_rsi > 50
            return bool(in_profit and rsi_neutral)
        else:
            # Short: exit when RSI drops back below 50 AND in profit
            in_profit = current_price < entry_price
            rsi_neutral = curr_rsi < 50
            return bool(in_profit and rsi_neutral)
