"""Shared News Sentiment Engine.

Fetches news headlines for crypto and stock symbols, uses LLM to score
sentiment, and caches results. Used by both AI analysis modules.

Configuration via environment variables:
    NEWS_API_KEY    — CryptoPanic API key (for crypto news)
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


class NewsSentiment:
    """Fetches and scores news sentiment for trading symbols."""

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any = None) -> None:
        cfg = config or {}
        ai_cfg = cfg.get("ai_analysis", {})

        self._enabled = ai_cfg.get("news_enabled", False)
        self._refresh_minutes = ai_cfg.get("news_refresh_minutes", 30)
        self._llm_client = llm_client
        self._api_key = os.environ.get("NEWS_API_KEY", "")

        # Cache: {symbol: (timestamp, sentiment_score)}
        self._cache: dict[str, tuple[float, float]] = {}
        # Headlines cache: {symbol: (timestamp, [headlines])}
        self._headlines_cache: dict[str, tuple[float, list[str]]] = {}

        if self._enabled:
            logger.info("NewsSentiment initialized (refresh=%dm)", self._refresh_minutes)
        else:
            logger.info("NewsSentiment disabled (set ai_analysis.news_enabled=true to activate)")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def get_sentiment(self, symbol: str, asset_type: str = "crypto") -> float:
        """Get sentiment score for a symbol.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT") or stock symbol (e.g. "RELIANCE")
            asset_type: "crypto" or "stock"

        Returns:
            Sentiment score from -1.0 (very bearish) to +1.0 (very bullish).
            Returns 0.0 (neutral) if disabled or on error.
        """
        if not self._enabled:
            return 0.0

        # Check cache
        cached = self._cache.get(symbol)
        if cached:
            ts, score = cached
            if time.time() - ts < self._refresh_minutes * 60:
                return score

        # Fetch and score headlines
        try:
            # Extract base symbol for news search
            base_symbol = self._extract_base_symbol(symbol, asset_type)
            headlines = self._fetch_headlines(base_symbol, asset_type)

            if not headlines:
                self._cache[symbol] = (time.time(), 0.0)
                return 0.0

            score = self._score_headlines(base_symbol, headlines, asset_type)
            self._cache[symbol] = (time.time(), score)
            logger.debug("News sentiment for %s: %.2f (%d headlines)", symbol, score, len(headlines))
            return score

        except Exception:
            logger.exception("News sentiment failed for %s — returning neutral", symbol)
            if cached:
                return cached[1]
            return 0.0

    def _extract_base_symbol(self, symbol: str, asset_type: str) -> str:
        """Extract the base currency/stock name from a trading pair."""
        if asset_type == "crypto":
            # "BTC/USDT" -> "BTC", "ETH/USDT" -> "ETH"
            return symbol.split("/")[0] if "/" in symbol else symbol
        else:
            # Stock symbols are already base: "RELIANCE", "HDFCBANK"
            return symbol

    def _fetch_headlines(self, base_symbol: str, asset_type: str) -> list[str]:
        """Fetch recent news headlines for a symbol."""
        # Check headlines cache first
        cache_key = f"{asset_type}:{base_symbol}"
        cached = self._headlines_cache.get(cache_key)
        if cached:
            ts, headlines = cached
            if time.time() - ts < self._refresh_minutes * 60:
                return headlines

        headlines: list[str] = []

        if asset_type == "crypto":
            headlines = self._fetch_crypto_headlines(base_symbol)
        else:
            headlines = self._fetch_stock_headlines(base_symbol)

        self._headlines_cache[cache_key] = (time.time(), headlines)
        return headlines

    def _fetch_crypto_headlines(self, symbol: str) -> list[str]:
        """Fetch crypto news from CryptoPanic API."""
        headlines: list[str] = []

        # CryptoPanic free API
        url = f"https://cryptopanic.com/api/free/v1/posts/?auth_token={self._api_key}&currencies={symbol}&kind=news&public=true"
        if not self._api_key:
            # Fallback: try without auth (limited)
            url = f"https://cryptopanic.com/api/free/v1/posts/?currencies={symbol}&kind=news&public=true"

        try:
            req = Request(url, headers={"User-Agent": "AlgoTradingBot/1.0"})
            with urlopen(req, timeout=10) as resp:
                import json
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                for item in results[:15]:  # Limit to 15 most recent
                    title = item.get("title", "")
                    if title:
                        headlines.append(title)
        except (URLError, Exception) as e:
            logger.debug("CryptoPanic fetch failed for %s: %s", symbol, e)

        # Fallback: CoinGecko trending/news if CryptoPanic fails
        if not headlines:
            headlines = self._fetch_coingecko_news(symbol)

        return headlines

    def _fetch_coingecko_news(self, symbol: str) -> list[str]:
        """Fallback: fetch from CoinGecko-related RSS."""
        headlines: list[str] = []
        # Generic crypto news RSS
        rss_urls = [
            "https://cointelegraph.com/rss",
        ]
        for rss_url in rss_urls:
            try:
                req = Request(rss_url, headers={"User-Agent": "AlgoTradingBot/1.0"})
                with urlopen(req, timeout=10) as resp:
                    tree = ET.parse(resp)
                    for item in tree.findall(".//item")[:20]:
                        title_el = item.find("title")
                        if title_el is not None and title_el.text:
                            # Only include headlines mentioning the symbol
                            if symbol.upper() in title_el.text.upper():
                                headlines.append(title_el.text)
            except Exception:
                continue

        return headlines[:10]

    def _fetch_stock_headlines(self, symbol: str) -> list[str]:
        """Fetch Indian stock news from RSS feeds."""
        headlines: list[str] = []
        rss_urls = [
            f"https://news.google.com/rss/search?q={symbol}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en",
            f"https://news.google.com/rss/search?q={symbol}+share+price&hl=en-IN&gl=IN&ceid=IN:en",
        ]

        for rss_url in rss_urls:
            try:
                req = Request(rss_url, headers={"User-Agent": "AlgoTradingBot/1.0"})
                with urlopen(req, timeout=10) as resp:
                    tree = ET.parse(resp)
                    for item in tree.findall(".//item")[:10]:
                        title_el = item.find("title")
                        if title_el is not None and title_el.text:
                            headlines.append(title_el.text)
            except Exception:
                continue

        return headlines[:15]

    def _score_headlines(self, symbol: str, headlines: list[str], asset_type: str) -> float:
        """Score headlines using LLM or keyword-based fallback."""
        if self._llm_client and self._llm_client.is_enabled:
            return self._score_with_llm(symbol, headlines, asset_type)
        return self._score_with_keywords(headlines)

    def _score_with_llm(self, symbol: str, headlines: list[str], asset_type: str) -> float:
        """Use LLM to score headline sentiment."""
        headlines_text = "\n".join(f"- {h}" for h in headlines[:10])
        indicators = {
            "task": "news_sentiment_scoring",
            "headlines": headlines_text,
            "instruction": f"Score overall sentiment for {symbol} based on these headlines",
        }
        result = self._llm_client.analyze_market_context(
            symbol=symbol,
            indicators=indicators,
            regime="news_analysis",
            asset_type=asset_type,
        )
        return result.get("sentiment", 0.0)

    def _score_with_keywords(self, headlines: list[str]) -> float:
        """Simple keyword-based sentiment as fallback when LLM is unavailable."""
        if not headlines:
            return 0.0

        positive_keywords = {
            "surge", "rally", "bullish", "gain", "rise", "jump", "soar",
            "breakout", "high", "upgrade", "buy", "growth", "profit",
            "recovery", "boost", "strong", "outperform", "beat",
        }
        negative_keywords = {
            "crash", "bearish", "drop", "fall", "plunge", "dump", "low",
            "sell", "loss", "decline", "risk", "warning", "fear", "hack",
            "fraud", "ban", "regulate", "weak", "downgrade", "miss",
        }

        total_score = 0.0
        for headline in headlines:
            words = set(headline.lower().split())
            pos_count = len(words & positive_keywords)
            neg_count = len(words & negative_keywords)
            if pos_count + neg_count > 0:
                total_score += (pos_count - neg_count) / (pos_count + neg_count)

        if not headlines:
            return 0.0

        avg_score = total_score / len(headlines)
        return max(-1.0, min(1.0, avg_score))

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._cache.clear()
        self._headlines_cache.clear()
