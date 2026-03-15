"""Stock Selection Engine — dynamically builds the daily trading watchlist.

Selection criteria (from PRD §10):
- Trading volume
- Liquidity
- Volatility
- Relevance to major market indices
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from stocks.core.enums import Timeframe
from stocks.core.event_bus import EventBus
from stocks.data.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)

# Nifty 50 constituent symbols (Yahoo format: SYMBOL.NS)
# Full Nifty 50 + select Nifty Next 50 high-volume names.
NIFTY50_SYMBOLS = [
    # --- Nifty 50 ---
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
    "TRENT.NS", "HCLTECH.NS", "M&M.NS", "NTPC.NS", "POWERGRID.NS",
    "TATASTEEL.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS", "ADANIPORTS.NS",
    "ONGC.NS", "COALINDIA.NS", "BPCL.NS", "GRASIM.NS", "CIPLA.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", "TECHM.NS",
    "HEROMOTOCO.NS", "BRITANNIA.NS", "HINDALCO.NS", "SHRIRAMFIN.NS", "BEL.NS",
    "TATACONSUM.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ADANIENT.NS", "TMCV.NS", "TMPV.NS",
    # --- Nifty Next 50 (high volume picks) ---
    "HAL.NS", "BANKBARODA.NS", "IOC.NS", "PNB.NS", "IRFC.NS",
    "ETERNAL.NS", "JIOFIN.NS", "DLF.NS", "ABB.NS", "VEDL.NS",
    "TATAPOWER.NS", "CANBK.NS", "RECLTD.NS", "PFC.NS", "NHPC.NS",
    # --- Oil & Gas / Energy (macro tailwind: geopolitical tensions) ---
    "GAIL.NS", "OIL.NS", "PETRONET.NS", "HINDPETRO.NS",
    "ADANIGREEN.NS", "ADANIENSOL.NS", "MGL.NS", "IGL.NS", "GSPL.NS",
]


class StockSelector:
    """Builds and maintains a daily watchlist of tradeable symbols."""

    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        market_data: MarketDataEngine,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.market_data = market_data

        sel_config = config.get("selection", {})
        self._min_volume = sel_config.get("min_volume", 500_000)
        self._min_price = sel_config.get("min_price", 50.0)
        self._max_price = sel_config.get("max_price", 10_000.0)
        self._max_watchlist_size = sel_config.get("max_watchlist_size", 20)

        # Priority symbols (sector momentum / macro tailwinds)
        self._priority_symbols: set[str] = set(
            sel_config.get("priority_symbols", [])
        )
        self._sector_momentum_boost = sel_config.get("sector_momentum_boost", 25)

        self._watchlist: list[str] = []
        self._candidate_pool: list[str] = list(NIFTY50_SYMBOLS)
        self._macro_analyst: Any = None

        logger.info(
            "StockSelector initialized (pool_size=%d, priority=%d)",
            len(self._candidate_pool), len(self._priority_symbols),
        )

    def set_macro_analyst(self, macro_analyst: Any) -> None:
        """Wire in the MacroAnalyst for AI-driven priority symbol updates."""
        self._macro_analyst = macro_analyst

    def build_watchlist(self) -> list[str]:
        """Score and rank candidates to build today's watchlist.

        Scoring considers:
        - Volume (higher = better)
        - Volatility (moderate preferred)
        - Price within range
        - Sector momentum boost for priority symbols
        """
        scored: list[tuple[str, float]] = []

        # --- MacroAnalyst: apply AI-driven symbol overrides ---
        # pairs_to_add -> add to priority list (sector tailwinds)
        # avoid_pairs  -> temporarily remove from candidate pool
        effective_pool = list(self._candidate_pool)
        effective_priority = set(self._priority_symbols)
        if self._macro_analyst is not None:
            macro_ctx = self._macro_analyst.get_context()
            if macro_ctx and macro_ctx.is_valid:
                if macro_ctx.pairs_to_add:
                    for sym in macro_ctx.pairs_to_add:
                        if sym in self._candidate_pool:
                            effective_priority.add(sym)
                    logger.info(
                        "MacroAnalyst: promoting %d symbol(s) to priority: %s",
                        len(macro_ctx.pairs_to_add),
                        ", ".join(macro_ctx.pairs_to_add),
                    )
                if macro_ctx.avoid_pairs:
                    effective_pool = [s for s in effective_pool if s not in macro_ctx.avoid_pairs]
                    logger.info(
                        "MacroAnalyst: excluding %d symbol(s) from pool: %s",
                        len(macro_ctx.avoid_pairs),
                        ", ".join(macro_ctx.avoid_pairs),
                    )

        for symbol in effective_pool:
            score = self._score_symbol(symbol)
            if score > 0:
                # Sector momentum: boost priority symbols (e.g. oil/gas during war)
                if symbol in effective_priority:
                    score += self._sector_momentum_boost
                scored.append((symbol, score))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        self._watchlist = [s for s, _ in scored[: self._max_watchlist_size]]

        # Ensure priority symbols are included if they pass basic filters
        # (even if they didn't score high enough for top N)
        watchlist_set = set(self._watchlist)
        for symbol in effective_priority:
            if symbol not in watchlist_set:
                score = self._score_symbol(symbol)
                if score > 0:  # Passes basic filters
                    self._watchlist.append(symbol)
                    watchlist_set.add(symbol)

        logger.info(
            "Watchlist built: %d symbols selected from %d candidates (%d priority)",
            len(self._watchlist), len(self._candidate_pool),
            len(self._priority_symbols & watchlist_set),
        )
        for sym in self._watchlist:
            tag = " [PRIORITY]" if sym in self._priority_symbols else ""
            logger.debug("  %s%s", sym, tag)

        return self._watchlist

    def _score_symbol(self, symbol: str) -> float:
        """Score a symbol based on selection criteria. Returns 0 if disqualified."""
        df = self.market_data.get_dataframe(symbol, Timeframe.D1)
        if df is None or len(df) < 10:
            return 0.0

        latest = df.iloc[-1]
        price = float(latest["close"])
        volume = float(latest.get("volume", 0))

        # Price filter
        if price < self._min_price or price > self._max_price:
            return 0.0

        # Volume filter
        avg_volume = float(df["volume"].tail(10).mean()) if "volume" in df.columns else 0
        if avg_volume < self._min_volume:
            return 0.0

        # Volume score (normalized)
        volume_score = min(avg_volume / 5_000_000, 1.0) * 40

        # Volatility score — prefer moderate volatility (10-day ATR as % of price)
        atr = self.market_data.get_indicator(symbol, Timeframe.D1, "atr_14")
        volatility_score = 0.0
        if atr is not None and not atr.empty:
            atr_pct = float(atr.iloc[-1]) / price * 100 if price > 0 else 0
            # Peak score at ~2% ATR, decays for very low or very high
            if 0.5 < atr_pct < 5.0:
                volatility_score = 30 * (1 - abs(atr_pct - 2.0) / 3.0)
                volatility_score = max(volatility_score, 0)

        # Trend score — stocks with clear direction are preferable
        adx = self.market_data.get_indicator(symbol, Timeframe.D1, "adx_14")
        trend_score = 0.0
        if adx is not None and not adx.empty:
            adx_val = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
            trend_score = min(adx_val / 40, 1.0) * 30

        return volume_score + volatility_score + trend_score

    @property
    def watchlist(self) -> list[str]:
        return list(self._watchlist)

    def set_candidate_pool(self, symbols: list[str]) -> None:
        """Override the default candidate pool."""
        self._candidate_pool = list(symbols)
        logger.info("Candidate pool updated: %d symbols", len(symbols))
