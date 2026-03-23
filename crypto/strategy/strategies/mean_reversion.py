"""Mean Reversion Strategy v2 — Wide Bollinger Band touch with extreme RSI.

Entry Requirements (ALL must be true):
  1. Price touches/pierces LOWER Bollinger Band (2.5σ wide - wider than standard 2σ)
  2. RSI < 25 (deeply oversold) OR RSI > 75 (deeply overbought for shorts)
  3. Volume confirmation (optional - helps filter false signals)
  4. EMA trend filter disabled for pure mean reversion (or optional)
  
Exit: RSI returns to neutral (45-55) AND minimum profit target hit

Why wider bands (2.5σ vs 2σ):
  - 2σ bands get touched too often in trending markets
  - 2.5σ = ~99% confidence interval = rarer, higher quality signals
  - Better risk:reward since entries are at more extreme levels
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
    """Wide Bollinger Band reversion with extreme RSI confirmation."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataEngine) -> None:
        super().__init__("mean_reversion", config, market_data)
        sc = config.get("mean_reversion", {})
        
        # RSI thresholds - more extreme for better signals
        self._rsi_oversold = sc.get("rsi_oversold", 25)
        self._rsi_overbought = sc.get("rsi_overbought", 75)
        
        # RSI extreme filter - skip capitulation zones
        self._rsi_capitulation_floor = sc.get("rsi_capitulation_floor", 10)
        self._rsi_capitulation_ceiling = sc.get("rsi_capitulation_ceiling", 90)
        
        # Stop/target settings - wider for mean reversion
        self._atr_stop_mult = sc.get("atr_multiplier_stop", 2.0)
        self._atr_target_mult = sc.get("atr_multiplier_target", 3.5)
        
        # Minimum exit profit - must cover fees + tax
        self._min_exit_profit_pct = sc.get("min_exit_profit_pct", 0.8) / 100
        
        # Trend filter (optional - can be disabled for pure mean reversion)
        self._trend_filter_enabled = sc.get("trend_filter_enabled", False)  # Disabled by default
        
        self.primary_timeframe = "15m"

        # Trailing stop + max hold from base
        self._trailing_stop_atr = sc.get("trailing_stop_atr_multiplier", 1.5)
        self._max_hold_minutes = sc.get("max_hold_minutes", 240)  # 4 hours max
        
        # State tracking for signal quality
        self._signal_history: dict[str, list] = {}

    def _is_deeply_oversold(self, df: pd.DataFrame) -> tuple[bool, float, str]:
        """Check if price is at extreme oversold condition.
        
        Returns: (is_valid, rsi_value, reason)
        """
        rsi = df.get("rsi", pd.Series(dtype=float))
        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return False, 0.0, "no_rsi"
            
        curr_rsi = rsi.iloc[-1]
        
        # Skip capitulation - RSI too low means free-fall
        if curr_rsi < self._rsi_capitulation_floor:
            return False, curr_rsi, "capitulation_floor"
        
        # Must be deeply oversold
        if curr_rsi >= self._rsi_oversold:
            return False, curr_rsi, "not_oversold"
        
        return True, curr_rsi, "valid"

    def _is_deeply_overbought(self, df: pd.DataFrame) -> tuple[bool, float, str]:
        """Check if price is at extreme overbought condition.
        
        Returns: (is_valid, rsi_value, reason)
        """
        rsi = df.get("rsi", pd.Series(dtype=float))
        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return False, 0.0, "no_rsi"
            
        curr_rsi = rsi.iloc[-1]
        
        # Skip vertical pump - RSI too high
        if curr_rsi > self._rsi_capitulation_ceiling:
            return False, curr_rsi, "capitulation_ceiling"
        
        # Must be deeply overbought
        if curr_rsi <= self._rsi_overbought:
            return False, curr_rsi, "not_overbought"
        
        return True, curr_rsi, "valid"

    def _check_price_at_band(self, df: pd.DataFrame, side: OrderSide) -> tuple[bool, str]:
        """Check if price is at the Bollinger Band extreme.
        
        Uses 2.5σ bands (wider than standard 2σ) for rarer, higher quality signals.
        
        Returns: (is_at_band, band_position)
        """
        close = df["close"].iloc[-1]
        bb_lower = df.get("bb_lower", pd.Series(dtype=float))
        bb_upper = df.get("bb_upper", pd.Series(dtype=float))
        bb_mid = df.get("bb_mid", pd.Series(dtype=float))
        
        if bb_lower.empty or bb_upper.empty or bb_mid.empty:
            return False, "no_data"
        
        curr_bb_lower = bb_lower.iloc[-1]
        curr_bb_upper = bb_upper.iloc[-1]
        curr_bb_mid = bb_mid.iloc[-1]
        
        if pd.isna(curr_bb_lower) or pd.isna(curr_bb_upper) or pd.isna(curr_bb_mid):
            return False, "nan_values"
        
        # Calculate band width for context
        bb_width = curr_bb_upper - curr_bb_lower
        
        if side == OrderSide.BUY:
            # For long: price should pierce or touch lower band
            # Allow small buffer (0.5%) for wick piercings
            band_threshold = curr_bb_lower * 1.005
            
            if close <= band_threshold:
                # Calculate how deep into the band
                depth_pct = (curr_bb_lower - close) / bb_width * 100 if bb_width > 0 else 0
                return True, f"lower_band (depth={depth_pct:.1f}%)"
            
            # Also check if price is in lower 10% of band width
            lower_zone = curr_bb_lower + bb_width * 0.1
            if close <= lower_zone:
                return True, "lower_zone_10pct"
                
        else:  # SELL
            # For short: price should pierce or touch upper band
            band_threshold = curr_bb_upper * 0.995
            
            if close >= band_threshold:
                depth_pct = (close - curr_bb_upper) / bb_width * 100 if bb_width > 0 else 0
                return True, f"upper_band (depth={depth_pct:.1f}%)"
            
            # Also check if price is in upper 10% of band width
            upper_zone = curr_bb_upper - bb_width * 0.1
            if close >= upper_zone:
                return True, "upper_zone_10pct"
        
        return False, "not_at_band"

    def _check_trend_filter(self, df: pd.DataFrame, side: OrderSide) -> tuple[bool, str]:
        """Check EMA trend alignment (optional filter).
        
        For mean reversion, we sometimes want to DISABLE this to catch
        reversals against the trend. But when enabled:
        - Don't buy dips in strong downtrends
        - Don't sell rips in strong uptrends
        """
        if not self._trend_filter_enabled:
            return True, "disabled"
        
        ema_9 = df.get("ema_9", pd.Series(dtype=float))
        ema_21 = df.get("ema_21", pd.Series(dtype=float))
        
        if ema_9.empty or ema_21.empty:
            return True, "no_ema_data"
        
        if pd.isna(ema_9.iloc[-1]) or pd.isna(ema_21.iloc[-1]):
            return True, "nan_ema"
        
        ema_bullish = ema_9.iloc[-1] > ema_21.iloc[-1]
        
        if side == OrderSide.BUY:
            # Don't buy dips in downtrend
            if not ema_bullish:
                return False, "ema_downtrend"
        else:
            # Don't sell rips in uptrend
            if ema_bullish:
                return False, "ema_uptrend"
        
        return True, "trend_aligned"

    def analyze(self, symbol: str) -> Optional[Signal]:
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty or len(df) < 30:  # Need enough data for BB calculation
            return None

        close = df["close"].iloc[-1]
        atr = df.get("atr", pd.Series(dtype=float))
        
        if atr.empty or pd.isna(atr.iloc[-1]) or atr.iloc[-1] == 0:
            return None
            
        curr_atr = atr.iloc[-1]
        
        # Get BB info for logging
        bb_lower = df.get("bb_lower", pd.Series(dtype=float))
        bb_upper = df.get("bb_upper", pd.Series(dtype=float))
        bb_mid = df.get("bb_mid", pd.Series(dtype=float))
        
        curr_bb_lower = bb_lower.iloc[-1] if not bb_lower.empty else 0
        curr_bb_upper = bb_upper.iloc[-1] if not bb_upper.empty else 0
        curr_bb_mid = bb_mid.iloc[-1] if not bb_mid.empty else 0

        # === CHECK FOR LONG ENTRY (Oversold condition) ===
        oversold_valid, curr_rsi, os_reason = self._is_deeply_oversold(df)
        
        if oversold_valid:
            at_band, band_info = self._check_price_at_band(df, OrderSide.BUY)
            
            if at_band:
                trend_ok, trend_reason = self._check_trend_filter(df, OrderSide.BUY)
                
                if trend_ok:
                    # All conditions met for long entry
                    stop = close - self._atr_stop_mult * curr_atr
                    target = curr_bb_mid  # Target is mean (BB midline)
                    
                    # Confidence based on RSI depth
                    confidence = min(0.5 + (self._rsi_oversold - curr_rsi) / 40, 0.85)
                    
                    logger.info(
                        "[mean_reversion] %s SIGNAL: BUY | RSI=%.1f | %s | BB_low=%.4f | mid=%.4f",
                        symbol, curr_rsi, band_info, curr_bb_lower, curr_bb_mid,
                    )
                    
                    return Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side=OrderSide.BUY,
                        strength=SignalStrength.MODERATE,
                        confidence=confidence,
                        entry_price=close,
                        stop_loss=stop,
                        target_price=target,
                        metadata={
                            "rsi": curr_rsi,
                            "bb_lower": curr_bb_lower,
                            "bb_mid": curr_bb_mid,
                            "band_info": band_info,
                        },
                    )
                else:
                    logger.debug(
                        "[mean_reversion] %s SKIP BUY: %s (RSI=%.1f)",
                        symbol, trend_reason, curr_rsi,
                    )

        # === CHECK FOR SHORT ENTRY (Overbought condition) ===
        overbought_valid, curr_rsi, ob_reason = self._is_deeply_overbought(df)
        
        if overbought_valid:
            at_band, band_info = self._check_price_at_band(df, OrderSide.SELL)
            
            if at_band:
                trend_ok, trend_reason = self._check_trend_filter(df, OrderSide.SELL)
                
                if trend_ok:
                    # All conditions met for short entry
                    stop = close + self._atr_stop_mult * curr_atr
                    target = curr_bb_mid  # Target is mean (BB midline)
                    
                    # Confidence based on RSI height
                    confidence = min(0.5 + (curr_rsi - self._rsi_overbought) / 40, 0.85)
                    
                    logger.info(
                        "[mean_reversion] %s SIGNAL: SELL | RSI=%.1f | %s | BB_high=%.4f | mid=%.4f",
                        symbol, curr_rsi, band_info, curr_bb_upper, curr_bb_mid,
                    )
                    
                    return Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        strength=SignalStrength.MODERATE,
                        confidence=confidence,
                        entry_price=close,
                        stop_loss=stop,
                        target_price=target,
                        metadata={
                            "rsi": curr_rsi,
                            "bb_upper": curr_bb_upper,
                            "bb_mid": curr_bb_mid,
                            "band_info": band_info,
                        },
                    )
                else:
                    logger.debug(
                        "[mean_reversion] %s SKIP SELL: %s (RSI=%.1f)",
                        symbol, trend_reason, curr_rsi,
                    )

        return None

    def should_exit(
        self, symbol: str, entry_price: float, current_price: float,
        side: OrderSide = OrderSide.BUY,
    ) -> bool:
        """Exit when RSI returns to neutral AND minimum profit achieved.
        
        Mean reversion works by:
        1. Enter at extremes (RSI < 25 or > 75)
        2. Exit when mean is approached (RSI ~ 50)
        3. Must have meaningful profit to cover fees/tax
        """
        df = self.market_data.get_dataframe(symbol, self.primary_timeframe)
        if df.empty:
            return False

        rsi = df.get("rsi", pd.Series(dtype=float))
        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return False

        curr_rsi = rsi.iloc[-1]
        
        # Calculate price move percentage
        if side == OrderSide.BUY:
            price_move_pct = (current_price - entry_price) / entry_price
            in_profit = price_move_pct >= self._min_exit_profit_pct
            rsi_neutral = curr_rsi >= 45  # RSI returned to neutral from oversold
        else:
            price_move_pct = (entry_price - current_price) / entry_price
            in_profit = price_move_pct >= self._min_exit_profit_pct
            rsi_neutral = curr_rsi <= 55  # RSI returned to neutral from overbought
        
        # Both conditions must be true
        if in_profit and rsi_neutral:
            logger.debug(
                "[mean_reversion] %s EXIT: profit=%.2f%% RSI=%.1f (neutral)",
                symbol, price_move_pct * 100, curr_rsi,
            )
            return True
        
        # Emergency exit: RSI went way against us (capitulation)
        if side == OrderSide.BUY and curr_rsi < self._rsi_capitulation_floor:
            logger.debug(
                "[mean_reversion] %s EMERGENCY EXIT: RSI capitulation %.1f",
                symbol, curr_rsi,
            )
            return True
            
        if side == OrderSide.SELL and curr_rsi > self._rsi_capitulation_ceiling:
            logger.debug(
                "[mean_reversion] %s EMERGENCY EXIT: RSI capitulation %.1f",
                symbol, curr_rsi,
            )
            return True
        
        return False
