"""Trading Pair Selection System (PRD §7, §17).

Dynamically determines which crypto pairs are eligible for trading based on
volume, spread, liquidity, and volatility criteria.
"""

from __future__ import annotations

import logging
from typing import Any

from crypto.core.event_bus import EventBus
from crypto.data.providers.ccxt_provider import CcxtProvider

logger = logging.getLogger(__name__)


class PairSelector:
    """Filters and ranks trading pairs based on liquidity and quality metrics."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus, provider: CcxtProvider) -> None:
        self.config = config
        self.event_bus = event_bus
        self.provider = provider

        sel = config.get("selection", {})
        self._candidates = sel.get("candidate_pairs", [])
        self._min_volume = sel.get("min_24h_volume_usdt", 10_000_000)
        self._max_spread = sel.get("max_spread_pct", 0.15)
        self._max_pairs = sel.get("max_active_pairs", 10)

        liq = config.get("liquidity_protection", {})
        self._min_depth = liq.get("min_orderbook_depth_usdt", 50_000)

        self.active_pairs: list[str] = []
        logger.info("PairSelector initialized (%d candidates)", len(self._candidates))

    def build_watchlist(self) -> list[str]:
        """Evaluate all candidates and return the top eligible pairs."""
        scored: list[tuple[str, float]] = []

        for pair in self._candidates:
            ticker = self.provider.fetch_ticker(pair)
            if not ticker or ticker.get("last", 0) <= 0:
                continue

            volume = ticker.get("quote_volume", 0)
            if volume < self._min_volume:
                logger.debug("%s rejected: volume %.0f < %.0f", pair, volume, self._min_volume)
                continue

            ob = self.provider.fetch_orderbook(pair, limit=20)
            spread = ob.get("spread_pct", 999)
            if spread > self._max_spread:
                logger.debug("%s rejected: spread %.3f%% > %.3f%%", pair, spread, self._max_spread)
                continue

            depth = min(ob.get("bids_depth", 0), ob.get("asks_depth", 0))
            if depth < self._min_depth:
                logger.debug("%s rejected: depth %.0f < %.0f", pair, depth, self._min_depth)
                continue

            # Score: higher volume + tighter spread = better
            score = volume / 1e6 - spread * 100
            scored.append((pair, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        self.active_pairs = [p for p, _ in scored[:self._max_pairs]]

        logger.info(
            "Watchlist built: %d/%d pairs passed filters — %s",
            len(self.active_pairs), len(self._candidates),
            ", ".join(self.active_pairs) if self.active_pairs else "(none)",
        )
        return self.active_pairs
