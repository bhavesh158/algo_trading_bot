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

_SENTINEL = float("inf")


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

        macro_cfg = config.get("macro_analyst", {})
        self._max_ai_pairs: int = int(macro_cfg.get("max_ai_pairs", 3))

        self.active_pairs: list[str] = []
        self._protected_pairs: set[str] = set()  # pairs with open positions — never filtered out
        self._macro_analyst: Any = None
        logger.info("PairSelector initialized (%d candidates)", len(self._candidates))

    def set_macro_analyst(self, macro_analyst: Any) -> None:
        """Wire in the MacroAnalyst for AI-driven pair injection."""
        self._macro_analyst = macro_analyst

    def set_protected_pairs(self, symbols: set[str]) -> None:
        """Set pairs that must always be in the watchlist (have open positions)."""
        new_set = set(symbols)
        if new_set != self._protected_pairs:
            self._protected_pairs = new_set
            if new_set:
                logger.info("Protected pairs updated: %s", ", ".join(sorted(new_set)))
            else:
                logger.info("Protected pairs cleared — no open positions")

    def build_watchlist(self) -> list[str]:
        """Evaluate all candidates and return the top eligible pairs."""
        scored: list[tuple[str, float]] = []

        for pair in self._candidates:
            # Skip filtering for protected pairs — always include them
            if pair in self._protected_pairs:
                scored.append((pair, float("inf")))
                logger.info("%s included (protected — open position)", pair)
                continue

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

        # --- MacroAnalyst pair injection ---
        # AI-suggested pairs get a boosted score so they survive the top-N cut.
        # They still need to pass a minimum volume check for safety.
        if self._macro_analyst is not None:
            macro_ctx = self._macro_analyst.get_context()
            if macro_ctx and macro_ctx.is_valid and macro_ctx.pairs_to_add:
                already_scored = {p for p, _ in scored}
                candidate_set = set(self._candidates)
                injected: list[str] = []
                for ai_pair in macro_ctx.pairs_to_add[: self._max_ai_pairs]:
                    if ai_pair in already_scored:
                        continue  # already included via normal scoring
                    if ai_pair not in candidate_set:
                        logger.debug("MacroAnalyst suggested unknown pair %s — skipped", ai_pair)
                        continue
                    # Quick volume check (skip spread/depth for AI-injected pairs)
                    ticker = self.provider.fetch_ticker(ai_pair)
                    if not ticker or ticker.get("last", 0) <= 0:
                        continue
                    volume = ticker.get("quote_volume", 0)
                    if volume < self._min_volume:
                        logger.info(
                            "MacroAnalyst pair %s skipped: volume %.0f < %.0f",
                            ai_pair, volume, self._min_volume,
                        )
                        continue
                    # Give it a very high score so it appears near the top
                    scored.append((ai_pair, _SENTINEL - 1))
                    injected.append(ai_pair)

                if injected:
                    logger.info(
                        "MacroAnalyst injected %d pair(s) into watchlist: %s",
                        len(injected), ", ".join(injected),
                    )
                    scored.sort(key=lambda x: x[1], reverse=True)

        self.active_pairs = [p for p, _ in scored[:self._max_pairs]]

        # Ensure protected pairs are always present even if max_pairs limit was hit
        for pp in self._protected_pairs:
            if pp not in self.active_pairs:
                self.active_pairs.append(pp)

        logger.info(
            "Watchlist built: %d/%d pairs passed filters — %s",
            len(self.active_pairs), len(self._candidates),
            ", ".join(self.active_pairs) if self.active_pairs else "(none)",
        )
        return self.active_pairs
