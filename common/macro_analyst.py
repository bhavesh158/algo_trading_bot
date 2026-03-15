"""Macro-AI Market Analyst.

Runs periodically to scan crypto and macro news, classify market mood, and
produce a MacroContext that gates strategy entries and expands the watchlist.

Workflow (every N minutes, default 30):
  1. Fetch headlines from multiple free sources (CryptoPanic, CoinTelegraph,
     Google News crypto/macro RSS feeds)
  2. Fetch Fear & Greed Index from Alternative.me (free, no key)
  3. Send to LLM via LLMClient.call_raw() with a macro-analysis prompt
     (falls back to keyword scoring if LLM is disabled / unavailable)
  4. Produce MacroContext: market_mood, active_themes, pair_recommendations,
     pairs_to_add, avoid_pairs

Graceful degradation:
  - If LLM is disabled  → keyword fallback (still extracts mood + themes)
  - If news fetch fails  → use whatever headlines did arrive
  - If everything fails  → neutral MacroContext (never blocks trading)

Enabling full LLM analysis:
  Set `ai_analysis.llm_enabled: true` in config and supply LLM_API_KEY env var.
  Enable the section:  `macro_analyst.enabled: true` (default).
  Optionally supply NEWS_API_KEY for authenticated CryptoPanic access.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists for the LLM-free fallback
# ---------------------------------------------------------------------------
_RISK_OFF_KEYWORDS: frozenset[str] = frozenset({
    "war", "conflict", "attack", "invasion", "sanctions", "missile",
    "hack", "exploit", "breach", "rug pull", "fraud", "ponzi",
    "crash", "crisis", "collapse", "panic", "sell-off", "dump",
    "ban", "crackdown", "lawsuit", "sec charges", "arrest",
    "recession", "inflation spike", "rate hike", "default",
    "bankruptcy", "liquidity crisis", "depegged", "depeg",
})
_RISK_ON_KEYWORDS: frozenset[str] = frozenset({
    "etf", "etf approval", "etf inflow", "institutional",
    "adoption", "partnership", "integration", "launch", "mainnet",
    "upgrade", "halving", "bull", "rally", "breakout",
    "all-time high", "record", "inflow", "accumulation",
    "approval", "approved", "listing", "staking reward",
})

# RSS feeds fetched for macro context (order: crypto-specific first, macro second)
_RSS_FEEDS: list[tuple[str, int]] = [
    ("https://cointelegraph.com/rss", 12),
    (
        "https://news.google.com/rss/search"
        "?q=bitcoin+crypto+market&hl=en&gl=US&ceid=US:en",
        10,
    ),
    (
        "https://news.google.com/rss/search"
        "?q=global+economy+geopolitical+market&hl=en&gl=US&ceid=US:en",
        8,
    ),
]

_MACRO_SYSTEM_PROMPT = (
    "You are a macro trading analyst for a crypto algorithmic trading bot. "
    "Analyze the provided news headlines and market sentiment data, then respond "
    "ONLY with valid JSON matching the requested schema. Be concise and precise."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairRecommendation:
    """Directional bias for a specific trading pair."""
    symbol: str
    bias: str          # "BUY" | "SELL" | "NEUTRAL"
    confidence: float  # 0.0 – 1.0
    reason: str


@dataclass
class MacroContext:
    """Snapshot of macro market conditions produced by MacroAnalyst."""
    market_mood: str          # "risk_on" | "risk_off" | "neutral"
    mood_confidence: float    # 0.0 – 1.0
    active_themes: list[str]  # e.g. ["geopolitical_risk", "institutional_adoption"]
    pair_recommendations: list[PairRecommendation]
    pairs_to_add: list[str]   # extra pairs to inject into watchlist this cycle
    avoid_pairs: list[str]    # hard-block these pairs this cycle
    reasoning: str
    fear_greed_score: int     # 0 (extreme fear) – 100 (extreme greed)
    fear_greed_label: str
    fetched_at: float         # Unix timestamp
    expires_at: float         # Unix timestamp

    # -- helpers -----------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """True if context hasn't expired yet."""
        return time.time() < self.expires_at

    @property
    def blocks_buys(self) -> bool:
        """Strong risk-off with high confidence: block all new BUY entries."""
        return self.market_mood == "risk_off" and self.mood_confidence >= 0.70

    def get_pair_bias(self, symbol: str) -> str:
        """Return directional bias for a pair ('NEUTRAL' if not found)."""
        for rec in self.pair_recommendations:
            if rec.symbol == symbol:
                return rec.bias
        return "NEUTRAL"

    def should_avoid(self, symbol: str) -> bool:
        return symbol in self.avoid_pairs


# ---------------------------------------------------------------------------
# MacroAnalyst
# ---------------------------------------------------------------------------

class MacroAnalyst:
    """Periodic macro market analyst.

    Call ``refresh()`` at startup and every ``refresh_minutes`` thereafter.
    Call ``get_context()`` to retrieve the cached MacroContext (returns None
    if no valid context exists yet).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        cfg = config.get("macro_analyst", {})

        self._enabled: bool = cfg.get("enabled", True)
        self._refresh_minutes: int = int(cfg.get("refresh_minutes", 30))
        self._max_ai_pairs: int = int(cfg.get("max_ai_pairs", 3))
        self._apply_mood_filter: bool = cfg.get("apply_mood_filter", True)
        self._apply_pair_bias: bool = cfg.get("apply_pair_bias", True)

        # All known candidate pairs (for safety gating — AI can only suggest these)
        self._candidate_pairs: list[str] = config.get("selection", {}).get("candidate_pairs", [])

        # Optional CryptoPanic API key (public endpoint still works without it)
        import os
        self._news_api_key: str = os.environ.get("NEWS_API_KEY", "")

        # LLM client — loaded lazily
        self._llm_client: Any = None
        self._init_llm(config)

        self._context: MacroContext | None = None
        self._last_refresh: float = 0.0

        if self._enabled:
            logger.info(
                "MacroAnalyst initialized (refresh=%dm, llm=%s, candidates=%d, "
                "mood_filter=%s, pair_bias=%s)",
                self._refresh_minutes,
                self._llm_client.is_enabled if self._llm_client else False,
                len(self._candidate_pairs),
                self._apply_mood_filter,
                self._apply_pair_bias,
            )
        else:
            logger.info("MacroAnalyst disabled — no macro gating will be applied")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def get_context(self) -> MacroContext | None:
        """Return the cached context if it is still valid, else None."""
        if self._context and self._context.is_valid:
            return self._context
        return None

    def should_refresh(self) -> bool:
        """Return True when it is time to fetch fresh macro data."""
        if not self._enabled:
            return False
        elapsed = time.time() - self._last_refresh
        return elapsed >= self._refresh_minutes * 60

    def refresh(self) -> MacroContext | None:
        """Fetch news + F&G, analyze, update internal cache, and return context."""
        if not self._enabled:
            return None

        try:
            logger.info("MacroAnalyst: refreshing macro context…")
            headlines = self._fetch_all_headlines()
            fear_greed_score, fear_greed_label = self._fetch_fear_greed()

            if self._llm_client and self._llm_client.is_enabled:
                context = self._analyze_with_llm(headlines, fear_greed_score, fear_greed_label)
            else:
                context = self._analyze_with_keywords(headlines, fear_greed_score, fear_greed_label)

            self._context = context
            self._last_refresh = time.time()

            # Summary log
            themes_str = ", ".join(context.active_themes) or "none"
            to_add_str = ", ".join(context.pairs_to_add) or "none"
            avoid_str = ", ".join(context.avoid_pairs) or "none"
            bias_summary = ", ".join(
                f"{r.symbol}={r.bias}" for r in context.pair_recommendations
            ) or "none"

            logger.info(
                "MacroAnalyst: mood=%s(%.2f) | F&G=%d(%s) | themes=[%s] | "
                "inject=[%s] | avoid=[%s] | biases=[%s]",
                context.market_mood, context.mood_confidence,
                context.fear_greed_score, context.fear_greed_label,
                themes_str, to_add_str, avoid_str, bias_summary,
            )
            if context.reasoning:
                logger.info("MacroAnalyst reasoning: %s", context.reasoning)

            return context

        except Exception:
            logger.exception("MacroAnalyst refresh failed — using neutral context")
            self._last_refresh = time.time()  # don't retry immediately on error
            ctx = self._neutral_context()
            self._context = ctx
            return ctx

    # ------------------------------------------------------------------
    # News fetching
    # ------------------------------------------------------------------

    def _fetch_all_headlines(self) -> list[str]:
        """Gather headlines from all sources; return deduplicated list."""
        headlines: list[str] = []

        headlines.extend(self._fetch_cryptopanic())
        for url, limit in _RSS_FEEDS:
            headlines.extend(self._fetch_rss(url, limit))

        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for h in headlines:
            h_clean = h.strip()
            if h_clean and h_clean not in seen:
                seen.add(h_clean)
                unique.append(h_clean)

        logger.debug("MacroAnalyst: fetched %d unique headlines", len(unique))
        return unique[:40]  # cap to keep LLM prompt size reasonable

    def _fetch_cryptopanic(self) -> list[str]:
        headlines: list[str] = []
        base = "https://cryptopanic.com/api/free/v1/posts/"
        url = (
            f"{base}?auth_token={self._news_api_key}&kind=news&public=true"
            if self._news_api_key
            else f"{base}?kind=news&public=true"
        )
        try:
            req = Request(url, headers={"User-Agent": "AlgoTradingBot/1.0"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                for item in data.get("results", [])[:20]:
                    title = item.get("title", "").strip()
                    if title:
                        headlines.append(title)
        except Exception as exc:
            logger.debug("CryptoPanic fetch failed: %s", exc)
        return headlines

    def _fetch_rss(self, url: str, limit: int = 10) -> list[str]:
        headlines: list[str] = []
        try:
            req = Request(url, headers={"User-Agent": "AlgoTradingBot/1.0"})
            with urlopen(req, timeout=10) as resp:
                tree = ET.parse(resp)
                for item in tree.findall(".//item")[:limit]:
                    title_el = item.find("title")
                    if title_el is not None and title_el.text:
                        headlines.append(title_el.text.strip())
        except Exception as exc:
            logger.debug("RSS fetch failed (%s…): %s", url[:60], exc)
        return headlines

    def _fetch_fear_greed(self) -> tuple[int, str]:
        """Alternative.me Fear & Greed Index (free, no API key needed)."""
        try:
            req = Request(
                "https://api.alternative.me/fng/?limit=1",
                headers={"User-Agent": "AlgoTradingBot/1.0"},
            )
            with urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
                entry = data.get("data", [{}])[0]
                score = int(entry.get("value", 50))
                label = entry.get("value_classification", "Neutral")
                logger.debug("Fear & Greed: %d (%s)", score, label)
                return score, label
        except Exception as exc:
            logger.debug("Fear & Greed fetch failed: %s", exc)
            return 50, "Neutral"

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _init_llm(self, config: dict[str, Any]) -> None:
        try:
            from common.llm_client import LLMClient
            self._llm_client = LLMClient(config)
        except Exception as exc:
            logger.warning("MacroAnalyst: could not init LLMClient: %s", exc)
            self._llm_client = None

    def _analyze_with_llm(
        self,
        headlines: list[str],
        fear_greed_score: int,
        fear_greed_label: str,
    ) -> MacroContext:
        """Ask the LLM to classify macro context from headlines."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        headlines_text = "\n".join(f"- {h}" for h in headlines[:30])
        candidate_symbols = [p.split("/")[0] for p in self._candidate_pairs[:25]]

        user_prompt = f"""Current time: {now_str}
Fear & Greed Index: {fear_greed_score}/100 ({fear_greed_label})
Available trading pairs (symbol only, all quoted USDT): {", ".join(candidate_symbols)}

Recent news headlines:
{headlines_text}

Respond with JSON matching this exact schema:
{{
  "market_mood": "risk_on|risk_off|neutral",
  "mood_confidence": <float 0.0-1.0>,
  "active_themes": ["<theme>", ...],
  "pair_recommendations": [
    {{"symbol": "<SYMBOL/USDT>", "bias": "BUY|SELL|NEUTRAL", "confidence": <float>, "reason": "<brief>"}}
  ],
  "pairs_to_add": ["<SYMBOL/USDT>", ...],
  "avoid_pairs": ["<SYMBOL/USDT>", ...],
  "reasoning": "<1-2 sentence macro summary>"
}}

Rules:
- pairs_to_add: max {self._max_ai_pairs}; must use SYMBOL/USDT format from available pairs above
- avoid_pairs: pairs to completely block this cycle (e.g. pair of a hacked protocol)
- mood = risk_off if: geopolitical escalation, major hack/exploit, regulatory crackdown, panic sell-off
- mood = risk_on  if: institutional inflows, ETF approvals, major partnerships, positive regulatory news
- mood = neutral  if: mixed signals or low-significance headlines
- Only recommend directional biases / new pairs when there is a clear macro catalyst"""

        raw = self._llm_client.call_raw(
            system_prompt=_MACRO_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cache_key="macro_context",
            max_tokens=900,
        )

        if not raw:
            logger.info("MacroAnalyst: LLM returned empty — falling back to keywords")
            return self._analyze_with_keywords(headlines, fear_greed_score, fear_greed_label)

        try:
            return self._parse_llm_response(raw, fear_greed_score, fear_greed_label)
        except Exception:
            logger.exception("MacroAnalyst: failed to parse LLM response — falling back")
            return self._analyze_with_keywords(headlines, fear_greed_score, fear_greed_label)

    def _parse_llm_response(
        self,
        raw: dict[str, Any],
        fear_greed_score: int,
        fear_greed_label: str,
    ) -> MacroContext:
        now = time.time()

        # market_mood
        mood = str(raw.get("market_mood", "neutral")).lower()
        if mood not in ("risk_on", "risk_off", "neutral"):
            mood = "neutral"

        # mood_confidence
        mood_conf = float(raw.get("mood_confidence", 0.5))
        mood_conf = max(0.0, min(1.0, mood_conf))

        # active_themes
        themes: list[str] = [
            str(t) for t in raw.get("active_themes", [])[:6]
        ]

        # pair_recommendations — safety: only known candidates accepted
        candidate_set = set(self._candidate_pairs)
        recs: list[PairRecommendation] = []
        for item in raw.get("pair_recommendations", [])[:12]:
            sym = str(item.get("symbol", "")).upper()
            bias = str(item.get("bias", "NEUTRAL")).upper()
            conf = float(item.get("confidence", 0.5))
            reason = str(item.get("reason", ""))[:200]
            # Normalise "BTC" → "BTC/USDT" if caller omitted the quote
            if "/" not in sym:
                sym = sym + "/USDT"
            if sym in candidate_set and bias in ("BUY", "SELL", "NEUTRAL"):
                recs.append(PairRecommendation(sym, bias, conf, reason))

        # pairs_to_add
        pairs_to_add: list[str] = []
        for p in raw.get("pairs_to_add", []):
            sym = str(p).upper()
            if "/" not in sym:
                sym = sym + "/USDT"
            if sym in candidate_set and sym not in pairs_to_add:
                pairs_to_add.append(sym)
                if len(pairs_to_add) >= self._max_ai_pairs:
                    break

        # avoid_pairs
        avoid_pairs: list[str] = []
        for p in raw.get("avoid_pairs", []):
            sym = str(p).upper()
            if "/" not in sym:
                sym = sym + "/USDT"
            if sym in candidate_set:
                avoid_pairs.append(sym)
        avoid_pairs = avoid_pairs[:10]

        reasoning = str(raw.get("reasoning", ""))[:500]

        return MacroContext(
            market_mood=mood,
            mood_confidence=mood_conf,
            active_themes=themes,
            pair_recommendations=recs,
            pairs_to_add=pairs_to_add,
            avoid_pairs=avoid_pairs,
            reasoning=reasoning,
            fear_greed_score=fear_greed_score,
            fear_greed_label=fear_greed_label,
            fetched_at=now,
            expires_at=now + self._refresh_minutes * 60,
        )

    # ------------------------------------------------------------------
    # Keyword-based fallback
    # ------------------------------------------------------------------

    def _analyze_with_keywords(
        self,
        headlines: list[str],
        fear_greed_score: int,
        fear_greed_label: str,
    ) -> MacroContext:
        """Deterministic keyword-scoring fallback (no LLM required)."""
        all_text = " ".join(headlines).lower()

        risk_off_score = sum(1 for kw in _RISK_OFF_KEYWORDS if kw in all_text)
        risk_on_score = sum(1 for kw in _RISK_ON_KEYWORDS if kw in all_text)

        # Fear & Greed index adds weight
        if fear_greed_score <= 20:
            risk_off_score += 4
        elif fear_greed_score <= 35:
            risk_off_score += 2
        elif fear_greed_score <= 45:
            risk_off_score += 1
        elif fear_greed_score >= 80:
            risk_on_score += 4
        elif fear_greed_score >= 65:
            risk_on_score += 2
        elif fear_greed_score >= 55:
            risk_on_score += 1

        total = risk_off_score + risk_on_score
        if total == 0:
            mood, conf = "neutral", 0.3
        elif risk_off_score > risk_on_score:
            mood = "risk_off"
            conf = min(0.45 + risk_off_score / max(total, 1), 0.85)
        else:
            mood = "risk_on"
            conf = min(0.45 + risk_on_score / max(total, 1), 0.85)

        # Extract macro themes from keywords
        themes: list[str] = []
        if any(kw in all_text for kw in ("war", "conflict", "invasion", "missile", "sanctions")):
            themes.append("geopolitical_risk")
        if any(kw in all_text for kw in ("rate hike", "inflation", "cpi", "fed", "rate")):
            themes.append("monetary_policy")
        if any(kw in all_text for kw in ("hack", "exploit", "breach", "rug pull", "fraud")):
            themes.append("security_incident")
        if any(kw in all_text for kw in ("etf", "institutional", "adoption", "approval")):
            themes.append("institutional_adoption")
        if any(kw in all_text for kw in ("ban", "crackdown", "lawsuit", "sec charges")):
            themes.append("regulatory_risk")
        if any(kw in all_text for kw in ("halving", "upgrade", "mainnet", "launch")):
            themes.append("protocol_catalyst")

        reasoning = (
            f"Keyword analysis: {risk_off_score} risk-off signal(s), "
            f"{risk_on_score} risk-on signal(s). "
            f"Fear & Greed: {fear_greed_score} ({fear_greed_label})."
        )

        now = time.time()
        return MacroContext(
            market_mood=mood,
            mood_confidence=conf,
            active_themes=themes,
            pair_recommendations=[],  # no per-pair biases without LLM
            pairs_to_add=[],          # no pair injection without LLM
            avoid_pairs=[],
            reasoning=reasoning,
            fear_greed_score=fear_greed_score,
            fear_greed_label=fear_greed_label,
            fetched_at=now,
            expires_at=now + self._refresh_minutes * 60,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _neutral_context(self) -> MacroContext:
        """Completely neutral context used as an error fallback."""
        now = time.time()
        return MacroContext(
            market_mood="neutral",
            mood_confidence=0.0,
            active_themes=[],
            pair_recommendations=[],
            pairs_to_add=[],
            avoid_pairs=[],
            reasoning="",
            fear_greed_score=50,
            fear_greed_label="Neutral",
            fetched_at=now,
            expires_at=now + self._refresh_minutes * 60,
        )
