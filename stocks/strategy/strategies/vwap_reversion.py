"""VWAP Reversion Strategy (Enhanced).

Enters when price deviates significantly from VWAP (Volume Weighted Average
Price) and expects reversion. VWAP is a strong intraday anchor — large
deviations tend to revert, especially in liquid large-cap stocks.

Key features:
- BUY when price deviates below VWAP (default 1.2%, configurable)
- SELL when price deviates above VWAP (default 1.0%, lower for easier triggers)
- Exit when price returns to VWAP
- Commission-aware: target must exceed min_target_pct to cover fees
- Higher-timeframe trend alignment required
- RSI confirmation for both directions
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from stocks.core.enums import OrderSide, SignalStrength, Timeframe
from stocks.core.models import Position, Signal
from stocks.data.market_data_engine import MarketDataEngine
from stocks.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class VWAPReversionStrategy(BaseStrategy):
    """Trade mean-reversion to VWAP with trend and volume confirmation."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("vwap_reversion", config, market_data)

        strat_config = config.get("vwap_reversion", {})
        # Restored conservative defaults after 2026-03-19 losses
        self._deviation_pct_buy = strat_config.get("deviation_pct_buy", 1.5)  # Restored from 1.2%
        self._deviation_pct_sell = strat_config.get("deviation_pct_sell", 1.5)  # Restored from 1.0%
        self._exit_at_vwap = strat_config.get("exit_at_vwap", True)
        self._require_trend = strat_config.get("require_trend_alignment", True)
        self._volume_multiplier = strat_config.get("volume_multiplier", 1.2)  # Restored from 1.0
        self._min_target_pct = strat_config.get("min_target_pct", 0.5)  # Restored from 0.4%
        self.primary_timeframe = Timeframe.M5

        # RSI thresholds
        self._rsi_capitulation_floor = strat_config.get("rsi_capitulation_floor", 15)  # BUY floor
        self._rsi_overbought_ceiling = strat_config.get("rsi_overbought_ceiling", 85)  # SELL ceiling

        # Trailing stop with activation threshold (restored after tight stops caused losses)
        self._trail_activate_pct: float = strat_config.get("trail_activate_pct", 0.6)  # Restored from 0.4%
        self._trailing_stop_pct: float = strat_config.get("trailing_stop_pct", 0.8)  # CRITICAL FIX: was 0.3%

        # Time exit: safety net
        self._max_hold_minutes = strat_config.get("max_hold_minutes", 240)  # Restored from 180

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

        # Volume confirmation (relaxed - only skip on very low volume)
        if "volume" in df.columns and len(df) >= 20:
            current_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].iloc[-20:].mean())
            if avg_vol > 0 and current_vol < avg_vol * self._volume_multiplier:
                # Only log for debugging, don't block
                logger.debug(
                    "[%s] Low volume for %s: vol=%.0f < %.0f (avg)",
                    self.strategy_id, symbol, current_vol, avg_vol * self._volume_multiplier,
                )

        # --- BUY: Price below VWAP by deviation threshold ---
        if deviation_pct <= -self._deviation_pct_buy:
            # RSI capitulation filter: skip if RSI < 15 (free-fall)
            if rsi_val < self._rsi_capitulation_floor:
                logger.debug(
                    "[%s] RSI capitulation blocked BUY %s: RSI=%.1f < %d",
                    self.strategy_id, symbol, rsi_val, self._rsi_capitulation_floor,
                )
                return None

            # NEW: Momentum confirmation — avoid catching falling knives
            # Check if price is in free-fall (>2% drop in last 5 candles)
            if len(df) >= 5:
                closes = df["close"].iloc[-5:]
                momentum_5 = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
                if momentum_5 < -2.0:  # Price dropping >2% in 5 candles
                    logger.debug(
                        "[%s] Momentum filter blocked BUY %s: 5-candle momentum=%.2f%% (falling knife)",
                        self.strategy_id, symbol, momentum_5,
                    )
                    return None

            # Trend filter: higher TF must not be in strong downtrend
            if self._require_trend and not self._is_trend_aligned(symbol, OrderSide.BUY):
                logger.debug(
                    "[%s] Trend filter blocked BUY %s: not aligned",
                    self.strategy_id, symbol,
                )
                return None

            target = vwap_val  # Revert to VWAP
            expected_profit_pct = (target - current_price) / current_price * 100
            if expected_profit_pct < self._min_target_pct:
                logger.debug(
                    "[%s] Expected profit too small for %s: %.2f%% < %.2f%%",
                    self.strategy_id, symbol, expected_profit_pct, self._min_target_pct,
                )
                return None

            stop_loss = current_price - 1.5 * atr_value
            confidence = min(abs(deviation_pct) / 3.0, 1.0)

            # RSI confirmation boost (oversold = higher confidence)
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
                "[%s] VWAP reversion BUY: %s price=%.2f vwap=%.2f dev=%.2f%% rsi=%.1f conf=%.2f",
                self.strategy_id, symbol, current_price, vwap_val, deviation_pct, rsi_val, confidence,
            )
            return signal

        # --- SELL: Price above VWAP by deviation threshold (LOWER threshold) ---
        if deviation_pct >= self._deviation_pct_sell:
            # RSI overbought filter: skip if RSI > 85 (vertical pump)
            if rsi_val > self._rsi_overbought_ceiling:
                logger.debug(
                    "[%s] RSI overbought blocked SELL %s: RSI=%.1f > %d",
                    self.strategy_id, symbol, rsi_val, self._rsi_overbought_ceiling,
                )
                return None

            # Trend filter: higher TF must not be in strong uptrend
            if self._require_trend and not self._is_trend_aligned(symbol, OrderSide.SELL):
                logger.debug(
                    "[%s] Trend filter blocked SELL %s: not aligned",
                    self.strategy_id, symbol,
                )
                return None

            target = vwap_val
            expected_profit_pct = (current_price - target) / current_price * 100
            if expected_profit_pct < self._min_target_pct:
                logger.debug(
                    "[%s] Expected profit too small for %s: %.2f%% < %.2f%%",
                    self.strategy_id, symbol, expected_profit_pct, self._min_target_pct,
                )
                return None

            stop_loss = current_price + 1.5 * atr_value
            confidence = min(abs(deviation_pct) / 3.0, 1.0)

            # RSI confirmation boost (overbought = higher confidence)
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
                "[%s] VWAP reversion SELL: %s price=%.2f vwap=%.2f dev=%.2f%% rsi=%.1f conf=%.2f",
                self.strategy_id, symbol, current_price, vwap_val, deviation_pct, rsi_val, confidence,
            )
            return signal

        return None

    def should_exit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        """VWAP-touch exit for Phase 1 (small reversion).

        Returns True when price has returned to VWAP. The commission holdback in
        the scheduler prevents closing when gross < round-trip fees.
        Note: in Phase 2 (momentum mode), get_exit_signal bypasses this method.
        """
        if not self._exit_at_vwap:
            return False

        vwap_val = self._get_vwap(symbol)
        if vwap_val is None:
            return False

        if entry_price < vwap_val:
            # BUY position: exit when price has risen back to VWAP
            return current_price >= vwap_val
        else:
            # SELL position: exit when price has fallen back to VWAP
            return current_price <= vwap_val

    def get_exit_signal(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position: Optional[Position] = None,
    ) -> Optional[Signal]:
        """Three-phase exit for VWAP reversion.

        Phase 1 — Hard stop-loss (always fires, highest priority):
            If price has moved past the stop-loss level set at entry, exit
            immediately regardless of how long the position has been open.

        Phase 2 — Momentum vs. reversion:
            a) Momentum mode (unrealized profit ≥ trail_activate_pct):
               Skip the VWAP-touch exit. A trailing stop at `trailing_stop_pct`%
               from the peak now manages the position. The trade keeps running
               as long as momentum holds; exits only when price pulls back from
               its high (or low for a short).
            b) Reversion mode (profit < trail_activate_pct):
               Exit when price returns to VWAP (should_exit). The commission
               holdback in the scheduler then defers if gross < round-trip fees.

        Phase 3 — Time exit (safety net only):
            Extended to 240 min (configurable) so it only fires if neither
            stop-loss nor trailing stop nor VWAP-touch exited the position.
        """
        should_close = False
        exit_reason = "strategy"

        # --- Phase 1: hard stop-loss ---
        if position is not None and self.check_stop_loss(position):
            should_close = True
            exit_reason = "stop_loss"
            logger.info(
                "[vwap_reversion] Stop-loss hit: %s price=%.2f stop=%.2f",
                symbol, current_price, position.stop_loss,
            )

        # --- Phase 2: momentum vs. reversion ---
        if not should_close:
            in_momentum = (
                position is not None
                and self._trail_activate_pct > 0
                and position.unrealized_pnl_pct >= self._trail_activate_pct  # must be positive profit
            )

            if in_momentum:
                # Momentum mode: trailing stop manages the exit
                if self.check_trailing_stop_pct(position):
                    should_close = True
                    exit_reason = "trailing_stop"
                    logger.info(
                        "[vwap_reversion] Trailing stop: %s price=%.2f peak=%.2f trail=%.1f%%",
                        symbol, current_price,
                        position.highest_since_entry if position.side == OrderSide.BUY
                        else position.lowest_since_entry,
                        self._trailing_stop_pct,
                    )
                # (else: still in momentum, hold — do NOT exit at VWAP touch)
            else:
                # Reversion mode: exit at VWAP touch (holdback handles commission check)
                if self.should_exit(symbol, entry_price, current_price):
                    should_close = True
                    exit_reason = "strategy"

        # --- Phase 3: time exit (safety net) ---
        if not should_close and position is not None:
            if self.check_time_exit(position):
                should_close = True
                exit_reason = "time_exit"
                logger.info(
                    "[vwap_reversion] Time exit: %s held %.0f min",
                    symbol, position.hold_duration_minutes,
                )

        if not should_close:
            return None

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=OrderSide.SELL,
            strength=SignalStrength.MODERATE,
            confidence=0.7,
            entry_price=current_price,
            metadata={"exit_reason": exit_reason},
        )
