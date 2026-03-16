"""VWAP Reversion Strategy.

Enters when price deviates significantly from VWAP (Volume Weighted Average
Price) and expects reversion. VWAP is a strong intraday anchor — large
deviations tend to revert, especially in liquid large-cap stocks.

Key features:
- BUY when price is >deviation_pct below VWAP (with volume + trend confirmation)
- SELL when price is >deviation_pct above VWAP (with confirmation)
- Exit when price returns to VWAP
- Commission-aware: target must exceed min_target_pct to cover fees
- Higher-timeframe trend alignment required
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


class VWAPReversionStrategy(BaseStrategy):
    """Trade mean-reversion to VWAP with trend and volume confirmation."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("vwap_reversion", config, market_data)

        strat_config = config.get("vwap_reversion", {})
        self._deviation_pct = strat_config.get("deviation_pct", 1.5)
        self._exit_at_vwap = strat_config.get("exit_at_vwap", True)
        self._require_trend = strat_config.get("require_trend_alignment", True)
        self._volume_multiplier = strat_config.get("volume_multiplier", 1.2)
        self._min_target_pct = strat_config.get("min_target_pct", 0.5)
        self.primary_timeframe = Timeframe.M5

        # Time exit from base class
        self._max_hold_minutes = strat_config.get("max_hold_minutes", 60)

    def _get_vwap(self, symbol: str) -> Optional[float]:
        """Get current VWAP value."""
        vwap = self.market_data.get_indicator(symbol, self.primary_timeframe, "vwap")
        if vwap is None or vwap.empty or np.isnan(vwap.iloc[-1]):
            return None
        return float(vwap.iloc[-1])

    def _is_trend_aligned(self, symbol: str, side: OrderSide) -> bool:
        """Check higher-timeframe trend supports the trade direction."""
        ema_9 = self.market_data.get_indicator(symbol, Timeframe.M15, "ema_9")
        ema_21 = self.market_data.get_indicator(symbol, Timeframe.M15, "ema_21")

        if ema_9 is None or ema_21 is None or ema_9.empty or ema_21.empty:
            return True  # No data — allow

        ema_9_val = float(ema_9.iloc[-1])
        ema_21_val = float(ema_21.iloc[-1])

        if np.isnan(ema_9_val) or np.isnan(ema_21_val):
            return True

        if side == OrderSide.BUY:
            # For BUY (below VWAP), higher TF must not be in strong downtrend
            if ema_21_val > 0:
                gap_pct = (ema_9_val - ema_21_val) / ema_21_val * 100
                return gap_pct > -0.2  # Allow sideways and uptrend
        else:
            # For SELL (above VWAP), higher TF must not be in strong uptrend
            if ema_21_val > 0:
                gap_pct = (ema_9_val - ema_21_val) / ema_21_val * 100
                return gap_pct < 0.2  # Allow sideways and downtrend

        return True

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df is None or len(df) < 20:
            return None

        vwap_val = self._get_vwap(symbol)
        if vwap_val is None or vwap_val <= 0:
            return None

        current_price = float(df["close"].iloc[-1])
        deviation_pct = (current_price - vwap_val) / vwap_val * 100

        # Check RSI — avoid extreme conditions
        rsi = self.market_data.get_indicator(symbol, self.primary_timeframe, "rsi_14")
        rsi_val = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50

        # ATR for stop loss
        atr = self.market_data.get_indicator(symbol, self.primary_timeframe, "atr_14")
        if atr is None or atr.empty or np.isnan(atr.iloc[-1]):
            return None
        atr_value = float(atr.iloc[-1])

        # Volume confirmation
        if "volume" in df.columns and len(df) >= 20:
            current_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].iloc[-20:].mean())
            if avg_vol > 0 and current_vol < avg_vol * self._volume_multiplier:
                return None  # Below-average volume — skip

        # --- BUY: Price significantly below VWAP ---
        if deviation_pct <= -self._deviation_pct:
            if rsi_val < 15:
                return None  # Capitulation — skip

            if self._require_trend and not self._is_trend_aligned(symbol, OrderSide.BUY):
                return None

            target = vwap_val  # Revert to VWAP
            expected_profit_pct = (target - current_price) / current_price * 100
            if expected_profit_pct < self._min_target_pct:
                return None  # Target too small to cover fees

            stop_loss = current_price - 1.5 * atr_value
            confidence = min(abs(deviation_pct) / 3.0, 1.0)

            # RSI confirmation boost
            if rsi_val < 35:
                confidence = min(confidence + 0.1, 1.0)

            strength = SignalStrength.STRONG if confidence >= 0.75 else SignalStrength.MODERATE

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
                    "vwap": vwap_val,
                    "deviation_pct": deviation_pct,
                    "rsi": rsi_val,
                    "max_hold_minutes": self._max_hold_minutes,
                },
            )
            logger.info(
                "[%s] VWAP reversion BUY: %s price=%.2f vwap=%.2f dev=%.2f%% rsi=%.1f",
                self.strategy_id, symbol, current_price, vwap_val, deviation_pct, rsi_val,
            )
            return signal

        # --- SELL: Price significantly above VWAP ---
        if deviation_pct >= self._deviation_pct:
            if rsi_val > 85:
                return None  # Extreme overbought — could keep running

            if self._require_trend and not self._is_trend_aligned(symbol, OrderSide.SELL):
                return None

            target = vwap_val
            expected_profit_pct = (current_price - target) / current_price * 100
            if expected_profit_pct < self._min_target_pct:
                return None

            stop_loss = current_price + 1.5 * atr_value
            confidence = min(abs(deviation_pct) / 3.0, 1.0)

            if rsi_val > 65:
                confidence = min(confidence + 0.1, 1.0)

            strength = SignalStrength.STRONG if confidence >= 0.75 else SignalStrength.MODERATE

            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=OrderSide.SELL,
                strength=strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target,
                metadata={
                    "vwap": vwap_val,
                    "deviation_pct": deviation_pct,
                    "rsi": rsi_val,
                    "max_hold_minutes": self._max_hold_minutes,
                },
            )
            logger.info(
                "[%s] VWAP reversion SELL: %s price=%.2f vwap=%.2f dev=%.2f%% rsi=%.1f",
                self.strategy_id, symbol, current_price, vwap_val, deviation_pct, rsi_val,
            )
            return signal

        return None

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        if not self._exit_at_vwap:
            return False

        vwap_val = self._get_vwap(symbol)
        if vwap_val is None:
            return False

        # Determine position direction from the entry price relative to VWAP.
        # VWAP-reversion BUYs are entered when price < VWAP (negative deviation);
        # SELLs are entered when price > VWAP (positive deviation).
        # Using entry_price vs vwap_val is reliable here because VWAP barely moves
        # between entry and the first few exit checks.
        #
        # The previous implementation used current_price >= entry_price to detect a
        # "long in profit" but that condition is always true at the moment of entry
        # (current == entry), so SELL positions were immediately exited.
        if entry_price < vwap_val:
            # BUY position: exit when price has risen back to VWAP
            return current_price >= vwap_val
        else:
            # SELL position: exit when price has fallen back to VWAP
            return current_price <= vwap_val
