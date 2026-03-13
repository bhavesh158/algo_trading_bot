"""AI-Assisted Analysis Layer.

Multi-factor signal confidence adjustment using:
- Technical indicators (trend, volatility, RSI, volume, VWAP, S/R, candle patterns)
- LLM-powered market analysis (optional)
- News sentiment (optional)

Per PRD §8: AI should assist decision making, not be the sole decision maker.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from stocks.core.enums import SignalStrength, Timeframe
from stocks.core.event_bus import EventBus
from stocks.core.models import Signal
from stocks.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class AIAnalysis:
    """Multi-factor signal filter and confidence adjuster with LLM + news support."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._market_data: Optional[MarketDataEngine] = None
        self._llm_client: Any = None
        self._news_sentiment: Any = None

        ai_config = config.get("ai_analysis", {})
        self._enabled = ai_config.get("enabled", True)
        self._min_boost = ai_config.get("min_confidence_boost", 0.1)
        self._max_boost = ai_config.get("max_confidence_boost", 0.3)
        self._lookback = ai_config.get("lookback_candles", 50)

        # Factor weights
        weights = ai_config.get("factor_weights", {})
        self._weight_technical = weights.get("technical", 0.6)
        self._weight_llm = weights.get("llm", 0.25)
        self._weight_news = weights.get("news", 0.15)

        # Initialize LLM + news
        self._init_ai_modules(config)

        logger.info("AIAnalysis initialized (enabled=%s, llm=%s, news=%s)",
                    self._enabled,
                    self._llm_client.is_enabled if self._llm_client else False,
                    self._news_sentiment.is_enabled if self._news_sentiment else False)

    def _init_ai_modules(self, config: dict[str, Any]) -> None:
        """Initialize LLM client and news sentiment (graceful if imports fail)."""
        try:
            from common.llm_client import LLMClient
            self._llm_client = LLMClient(config)
        except Exception:
            logger.debug("LLMClient not available — LLM analysis disabled")
            self._llm_client = None

        try:
            from common.news_sentiment import NewsSentiment
            self._news_sentiment = NewsSentiment(config, llm_client=self._llm_client)
        except Exception:
            logger.debug("NewsSentiment not available — news analysis disabled")
            self._news_sentiment = None

    def set_market_data(self, market_data: MarketDataEngine) -> None:
        self._market_data = market_data

    def evaluate_signal(self, signal: Signal) -> Signal:
        """Evaluate and adjust a signal's confidence using weighted multi-factor analysis.

        Returns the signal with potentially modified confidence and strength.
        """
        if not self._enabled or self._market_data is None:
            return signal

        # --- Technical factors ---
        tech_adjustments: list[float] = []
        tech_adjustments.append(self._check_trend_alignment(signal))
        tech_adjustments.append(self._check_volatility_regime(signal))
        tech_adjustments.append(self._check_rsi_confirmation(signal))
        tech_adjustments.append(self._check_volume_confirmation(signal))
        tech_adjustments.append(self._check_vwap_position(signal))
        tech_adjustments.append(self._check_support_resistance(signal))
        tech_adjustments.append(self._check_candle_patterns(signal))

        tech_score = sum(tech_adjustments) / len(tech_adjustments) if tech_adjustments else 0.0

        # --- LLM factor ---
        llm_score = 0.0
        if self._llm_client and self._llm_client.is_enabled:
            llm_score = self._get_llm_adjustment(signal)

        # --- News factor ---
        news_score = 0.0
        if self._news_sentiment and self._news_sentiment.is_enabled:
            news_score = self._get_news_adjustment(signal)

        # Weighted combination (redistribute disabled weights to technical)
        llm_active = self._llm_client and self._llm_client.is_enabled
        news_active = self._news_sentiment and self._news_sentiment.is_enabled

        if llm_active and news_active:
            adjustment = (self._weight_technical * tech_score +
                         self._weight_llm * llm_score +
                         self._weight_news * news_score)
        elif llm_active:
            w_tech = self._weight_technical + self._weight_news
            adjustment = w_tech * tech_score + self._weight_llm * llm_score
        elif news_active:
            w_tech = self._weight_technical + self._weight_llm
            adjustment = w_tech * tech_score + self._weight_news * news_score
        else:
            adjustment = tech_score

        adjustment = max(-self._max_boost, min(self._max_boost, adjustment))
        new_confidence = max(0.0, min(1.0, signal.confidence + adjustment))

        if new_confidence != signal.confidence:
            logger.debug(
                "AI adjusted %s %s: %.2f -> %.2f (tech=%.3f llm=%.3f news=%.3f)",
                signal.strategy_id, signal.symbol, signal.confidence,
                new_confidence, tech_score, llm_score, news_score,
            )

        signal.confidence = new_confidence

        # Update strength based on new confidence
        if new_confidence >= 0.8:
            signal.strength = SignalStrength.STRONG
        elif new_confidence >= 0.5:
            signal.strength = SignalStrength.MODERATE
        else:
            signal.strength = SignalStrength.WEAK

        return signal

    # --- LLM + News factors ---

    def _get_llm_adjustment(self, signal: Signal) -> float:
        """Get confidence adjustment from LLM analysis."""
        try:
            df = self._market_data.get_dataframe(signal.symbol, Timeframe.M5)
            if df is None or df.empty:
                return 0.0

            indicators: dict[str, Any] = {
                "price": signal.entry_price,
                "side": signal.side.name,
                "strategy": signal.strategy_id,
            }
            for col in ["rsi_14", "atr_14", "ema_9", "ema_21", "vwap"]:
                if col in df.columns and not np.isnan(df[col].iloc[-1]):
                    indicators[col] = float(df[col].iloc[-1])

            result = self._llm_client.analyze_market_context(
                symbol=signal.symbol,
                indicators=indicators,
                regime="unknown",
                asset_type="stock",
            )
            return result.get("confidence_adjustment", 0.0)
        except Exception:
            logger.debug("LLM adjustment failed for %s", signal.symbol)
            return 0.0

    def _get_news_adjustment(self, signal: Signal) -> float:
        """Get confidence adjustment from news sentiment."""
        try:
            sentiment = self._news_sentiment.get_sentiment(signal.symbol, asset_type="stock")
            from stocks.core.enums import OrderSide
            if signal.side == OrderSide.BUY:
                return sentiment * self._min_boost
            else:
                return -sentiment * self._min_boost
        except Exception:
            logger.debug("News adjustment failed for %s", signal.symbol)
            return 0.0

    def _check_trend_alignment(self, signal: Signal) -> float:
        """Boost if signal aligns with higher-timeframe trend."""
        ema_9 = self._market_data.get_indicator(signal.symbol, Timeframe.M15, "ema_9")
        ema_21 = self._market_data.get_indicator(signal.symbol, Timeframe.M15, "ema_21")

        if ema_9 is None or ema_21 is None or ema_9.empty or ema_21.empty:
            return 0.0

        ema_9_val = float(ema_9.iloc[-1])
        ema_21_val = float(ema_21.iloc[-1])

        from stocks.core.enums import OrderSide
        if signal.side == OrderSide.BUY and ema_9_val > ema_21_val:
            return self._min_boost  # Trend supports buy
        elif signal.side == OrderSide.SELL and ema_9_val < ema_21_val:
            return self._min_boost  # Trend supports sell
        elif signal.side == OrderSide.BUY and ema_9_val < ema_21_val:
            return -self._min_boost  # Counter-trend buy
        elif signal.side == OrderSide.SELL and ema_9_val > ema_21_val:
            return -self._min_boost  # Counter-trend sell

        return 0.0

    def _check_volatility_regime(self, signal: Signal) -> float:
        """Penalize signals during extreme volatility."""
        atr = self._market_data.get_indicator(signal.symbol, Timeframe.M5, "atr_14")
        if atr is None or len(atr) < self._lookback:
            return 0.0

        current_atr = float(atr.iloc[-1])
        mean_atr = float(atr.iloc[-self._lookback:].mean())

        if mean_atr == 0:
            return 0.0

        ratio = current_atr / mean_atr

        # High volatility (>2x normal) — penalize
        if ratio > 2.0:
            return -self._max_boost
        # Very low volatility (<0.5x) — slight penalty (choppy)
        elif ratio < 0.5:
            return -self._min_boost
        # Moderate volatility — neutral to positive
        elif 0.8 <= ratio <= 1.5:
            return self._min_boost * 0.5

        return 0.0

    def _check_rsi_confirmation(self, signal: Signal) -> float:
        """Check if RSI supports the signal direction."""
        rsi = self._market_data.get_indicator(signal.symbol, Timeframe.M5, "rsi_14")
        if rsi is None or rsi.empty or np.isnan(rsi.iloc[-1]):
            return 0.0

        rsi_val = float(rsi.iloc[-1])

        from stocks.core.enums import OrderSide
        if signal.side == OrderSide.BUY:
            if rsi_val < 30:
                return self._min_boost  # Oversold — supports buy
            elif rsi_val > 70:
                return -self._min_boost  # Overbought — penalize buy
        elif signal.side == OrderSide.SELL:
            if rsi_val > 70:
                return self._min_boost  # Overbought — supports sell
            elif rsi_val < 30:
                return -self._min_boost  # Oversold — penalize sell

        return 0.0

    def _check_volume_confirmation(self, signal: Signal) -> float:
        """Boost if current volume is above average."""
        df = self._market_data.get_dataframe(signal.symbol, Timeframe.M5)
        if df is None or "volume" not in df.columns or len(df) < 20:
            return 0.0

        current_vol = float(df["volume"].iloc[-1])
        avg_vol = float(df["volume"].iloc[-20:].mean())

        if avg_vol == 0:
            return 0.0

        ratio = current_vol / avg_vol

        if ratio > 1.5:
            return self._min_boost  # Strong volume
        elif ratio < 0.5:
            return -self._min_boost  # Weak volume

        return 0.0

    def _check_vwap_position(self, signal: Signal) -> float:
        """Boost signals aligned with VWAP (buy below, sell above)."""
        vwap = self._market_data.get_indicator(signal.symbol, Timeframe.M5, "vwap")
        if vwap is None or vwap.empty or np.isnan(vwap.iloc[-1]):
            return 0.0

        vwap_val = float(vwap.iloc[-1])
        price = signal.entry_price

        if price <= 0 or vwap_val <= 0:
            return 0.0

        from stocks.core.enums import OrderSide
        if signal.side == OrderSide.BUY and price < vwap_val:
            return self._min_boost  # Buying below VWAP — good
        elif signal.side == OrderSide.BUY and price > vwap_val * 1.01:
            return -self._min_boost * 0.5  # Buying well above VWAP — slight penalty
        elif signal.side == OrderSide.SELL and price > vwap_val:
            return self._min_boost  # Selling above VWAP — good
        elif signal.side == OrderSide.SELL and price < vwap_val * 0.99:
            return -self._min_boost * 0.5  # Selling well below VWAP — slight penalty

        return 0.0

    def _check_support_resistance(self, signal: Signal) -> float:
        """Boost if entry price is near a support (for BUY) or resistance (for SELL)."""
        df = self._market_data.get_dataframe(signal.symbol, Timeframe.M5)
        if df is None or len(df) < self._lookback:
            return 0.0

        price = signal.entry_price
        recent = df.iloc[-self._lookback:]

        # Find recent support (lowest lows) and resistance (highest highs)
        support = float(recent["low"].rolling(5).min().iloc[-1])
        resistance = float(recent["high"].rolling(5).max().iloc[-1])

        if np.isnan(support) or np.isnan(resistance):
            return 0.0

        price_range = resistance - support
        if price_range <= 0:
            return 0.0

        from stocks.core.enums import OrderSide
        if signal.side == OrderSide.BUY:
            # Boost if near support (bottom 20% of range)
            if price <= support + price_range * 0.2:
                return self._min_boost
            # Penalize if near resistance
            if price >= resistance - price_range * 0.1:
                return -self._min_boost * 0.5
        elif signal.side == OrderSide.SELL:
            # Boost if near resistance
            if price >= resistance - price_range * 0.2:
                return self._min_boost
            # Penalize if near support
            if price <= support + price_range * 0.1:
                return -self._min_boost * 0.5

        return 0.0

    def _check_candle_patterns(self, signal: Signal) -> float:
        """Detect simple reversal candle patterns."""
        df = self._market_data.get_dataframe(signal.symbol, Timeframe.M5)
        if df is None or len(df) < 3:
            return 0.0

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        o = float(latest["open"])
        h = float(latest["high"])
        l = float(latest["low"])
        c = float(latest["close"])
        body = abs(c - o)
        candle_range = h - l

        if candle_range == 0:
            return 0.0

        from stocks.core.enums import OrderSide

        # Hammer (bullish reversal): small body at top, long lower wick
        if signal.side == OrderSide.BUY:
            lower_wick = min(o, c) - l
            if body > 0 and lower_wick > body * 2 and (h - max(o, c)) < body * 0.5:
                return self._min_boost  # Hammer pattern

            # Bullish engulfing: current candle body engulfs previous bearish body
            prev_o = float(prev["open"])
            prev_c = float(prev["close"])
            if prev_c < prev_o and c > o:  # Previous bearish, current bullish
                if c > prev_o and o < prev_c:  # Current engulfs previous
                    return self._min_boost

        # Shooting star / bearish engulfing (for SELL signals)
        elif signal.side == OrderSide.SELL:
            upper_wick = h - max(o, c)
            if body > 0 and upper_wick > body * 2 and (min(o, c) - l) < body * 0.5:
                return self._min_boost  # Shooting star

            prev_o = float(prev["open"])
            prev_c = float(prev["close"])
            if prev_c > prev_o and c < o:  # Previous bullish, current bearish
                if o > prev_c and c < prev_o:  # Current engulfs previous
                    return self._min_boost

        return 0.0
