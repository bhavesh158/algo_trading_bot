"""Shared LLM Client for AI-powered market analysis.

Supports Gemini and OpenAI APIs. Used by both stocks and crypto AI analysis
modules to get intelligent confidence adjustments and market context.

Configuration via environment variables:
    LLM_API_KEY     — API key for the chosen provider
    LLM_PROVIDER    — "gemini" (default) or "openai"
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Neutral response returned when LLM is disabled or fails
_NEUTRAL_RESPONSE: dict[str, Any] = {
    "sentiment": 0.0,
    "confidence_adjustment": 0.0,
    "reasoning": "",
    "key_levels": [],
}

_SYSTEM_PROMPT = """You are a quantitative trading analyst. Given market data for a financial instrument,
provide a brief analysis in JSON format with these fields:
- "sentiment": float from -1.0 (very bearish) to +1.0 (very bullish)
- "confidence_adjustment": float from -0.3 to +0.3 (how much to adjust signal confidence)
- "reasoning": string, 1-2 sentences explaining your assessment
- "key_levels": list of floats, important price levels (support/resistance)

Be concise. Focus on actionable insights. Respond ONLY with valid JSON."""


class LLMClient:
    """Rate-limited, cached LLM client for market analysis."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        ai_cfg = cfg.get("ai_analysis", {})

        self._enabled = ai_cfg.get("llm_enabled", False)
        self._provider = ai_cfg.get("llm_provider", os.environ.get("LLM_PROVIDER", "gemini"))
        self._cache_minutes = ai_cfg.get("llm_cache_minutes", 15)
        self._api_key = os.environ.get("LLM_API_KEY", "")
        # Model name — override via LLM_MODEL env var or ai_analysis.llm_model config
        self._gemini_model = ai_cfg.get(
            "llm_model", os.environ.get("LLM_MODEL", "gemini-2.0-flash-001")
        )

        # TTL cache: {cache_key: (timestamp, response)}
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

        if self._enabled and not self._api_key:
            logger.warning("LLM enabled but LLM_API_KEY not set — LLM analysis will be skipped")
            self._enabled = False

        if self._enabled:
            logger.info("LLMClient initialized (provider=%s, cache=%dm)", self._provider, self._cache_minutes)
        else:
            logger.info("LLMClient disabled (set ai_analysis.llm_enabled=true and LLM_API_KEY to activate)")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def analyze_market_context(
        self,
        symbol: str,
        indicators: dict[str, Any],
        regime: str = "unknown",
        asset_type: str = "crypto",
    ) -> dict[str, Any]:
        """Analyze market context using LLM.

        Args:
            symbol: Trading pair or stock symbol (e.g. "BTC/USDT" or "RELIANCE")
            indicators: Dict of indicator values (rsi, ema_9, ema_21, adx, atr, bb_upper, bb_lower, etc.)
            regime: Current market regime string
            asset_type: "crypto" or "stock"

        Returns:
            Dict with sentiment, confidence_adjustment, reasoning, key_levels.
            Returns neutral response on any error.
        """
        if not self._enabled:
            return _NEUTRAL_RESPONSE.copy()

        # Check cache
        cache_key = f"{asset_type}:{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            ts, response = cached
            if time.time() - ts < self._cache_minutes * 60:
                return response.copy()

        # Build prompt
        prompt = self._build_prompt(symbol, indicators, regime, asset_type)

        try:
            if self._provider == "gemini":
                response = self._call_gemini(prompt)
            elif self._provider == "openai":
                response = self._call_openai(prompt)
            else:
                logger.warning("Unknown LLM provider: %s", self._provider)
                return _NEUTRAL_RESPONSE.copy()

            # Validate and clamp response
            response = self._validate_response(response)

            # Cache successful response
            self._cache[cache_key] = (time.time(), response)
            logger.debug("LLM analysis for %s: sentiment=%.2f adj=%.2f", symbol, response["sentiment"], response["confidence_adjustment"])
            return response.copy()

        except Exception:
            logger.exception("LLM analysis failed for %s — returning neutral", symbol)
            # Return stale cache if available
            if cached:
                return cached[1].copy()
            return _NEUTRAL_RESPONSE.copy()

    def _build_prompt(
        self, symbol: str, indicators: dict[str, Any], regime: str, asset_type: str,
    ) -> str:
        """Build a structured prompt for the LLM."""
        lines = [
            f"Asset: {symbol} ({asset_type})",
            f"Market Regime: {regime}",
            "Current Indicators:",
        ]
        for key, value in indicators.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        lines.append("\nProvide your analysis as JSON.")
        return "\n".join(lines)

    def call_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        cache_key: Optional[str] = None,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        """Call LLM with fully custom system/user prompts, returning raw JSON.

        Unlike analyze_market_context(), no schema validation/clamping is applied.
        The caller is responsible for parsing the response.
        Returns {} when LLM is disabled or the call fails.
        """
        if not self._enabled:
            return {}

        if cache_key:
            cached = self._cache.get(cache_key)
            if cached:
                ts, response = cached
                if time.time() - ts < self._cache_minutes * 60:
                    return response.copy()

        try:
            if self._provider == "gemini":
                response = self._call_gemini_raw(system_prompt, user_prompt, max_tokens)
            elif self._provider == "openai":
                response = self._call_openai_raw(system_prompt, user_prompt, max_tokens)
            else:
                logger.warning("Unknown LLM provider: %s", self._provider)
                return {}

            if cache_key and response:
                self._cache[cache_key] = (time.time(), response)

            return response
        except Exception:
            logger.exception("LLM call_raw failed (cache_key=%s)", cache_key)
            return {}

    def _parse_response_text(self, response: object) -> str:
        """Extract text from a Gemini response, handling thinking-model part structure.

        Thinking models (gemini-2.5-*) may leave response.text as None when the
        thinking budget occupies dedicated parts.  Fall back to iterating parts in
        reverse so the last real-text part is returned.
        """
        text = getattr(response, "text", None)
        if text:
            return text
        try:
            for candidate in (getattr(response, "candidates", None) or []):
                parts = getattr(getattr(candidate, "content", None), "parts", None) or []
                for part in reversed(parts):
                    t = getattr(part, "text", None)
                    if t:
                        return t
        except Exception:
            pass
        return ""

    @staticmethod
    def _clean_json(text: str) -> str:
        """Remove JS-style syntax that thinking models sometimes emit inside JSON."""
        text = re.sub(r"//[^\n]*", "", text)               # // line comments
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)  # /* block comments */
        text = re.sub(r",(\s*[}\]])", r"\1", text)         # trailing commas
        return text.strip()

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from text, tolerating markdown fences, JS comments, trailing commas."""
        text = text.strip()
        if not text:
            raise ValueError("Empty response text from LLM")
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text.strip()).strip()
        # Direct parse after cleaning
        try:
            return json.loads(self._clean_json(text))
        except json.JSONDecodeError:
            # Extract first complete {...} block (handles thinking preamble leaking in)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(self._clean_json(match.group()))
            raise

    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        """Call Gemini API (google-genai) with the fixed per-symbol system prompt."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.warning("google-genai not installed — pip install google-genai")
            return _NEUTRAL_RESPONSE.copy()

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._gemini_model,
            contents=prompt,
            config=self._gemini_json_config(
                types,
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=300,
            ),
        )
        return self._extract_json(self._parse_response_text(response))

    def _call_gemini_raw(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> dict[str, Any]:
        """Call Gemini with fully custom prompts."""
        try:
            from google import genai
            from google.genai import types
            from google.genai.errors import ServerError
        except ImportError:
            logger.warning("google-genai not installed — pip install google-genai")
            return {}

        client = genai.Client(api_key=self._api_key)
        try:
            response = client.models.generate_content(
                model=self._gemini_model,
                contents=user_prompt,
                config=self._gemini_json_config(
                    types,
                    system_instruction=system_prompt,
                    temperature=0.2,
                    max_output_tokens=max_tokens,
                ),
            )
        except ServerError as exc:
            if getattr(exc, "status_code", 0) == 503:
                logger.warning(
                    "Gemini API temporarily unavailable (503 — high demand): %s", exc
                )
                return {}
            raise
        return self._extract_json(self._parse_response_text(response))

    @staticmethod
    def _gemini_json_config(types: object, **kwargs: Any) -> object:
        """Build a GenerateContentConfig requesting JSON output with thinking disabled.

        Disabling thinking (budget=0) prevents gemini-2.5-* from injecting reasoning
        text or JS-style comments into the structured JSON output.
        """
        thinking_cfg = getattr(types, "ThinkingConfig", None)
        if thinking_cfg is not None:
            kwargs["thinking_config"] = thinking_cfg(thinking_budget=0)
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            **kwargs,
        )

    def _call_openai(self, prompt: str) -> dict[str, Any]:
        """Call OpenAI API with the fixed per-symbol system prompt."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed — pip install openai")
            return _NEUTRAL_RESPONSE.copy()

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def _call_openai_raw(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> dict[str, Any]:
        """Call OpenAI with fully custom prompts."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed — pip install openai")
            return {}

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def _validate_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Validate and clamp LLM response values."""
        result = _NEUTRAL_RESPONSE.copy()

        sentiment = response.get("sentiment", 0.0)
        if isinstance(sentiment, (int, float)):
            result["sentiment"] = max(-1.0, min(1.0, float(sentiment)))

        adj = response.get("confidence_adjustment", 0.0)
        if isinstance(adj, (int, float)):
            result["confidence_adjustment"] = max(-0.3, min(0.3, float(adj)))

        reasoning = response.get("reasoning", "")
        if isinstance(reasoning, str):
            result["reasoning"] = reasoning[:500]  # Truncate long reasoning

        key_levels = response.get("key_levels", [])
        if isinstance(key_levels, list):
            result["key_levels"] = [float(l) for l in key_levels[:10] if isinstance(l, (int, float))]

        return result

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()

    def invalidate_symbol(self, symbol: str) -> None:
        """Remove cached response for a specific symbol."""
        for key in list(self._cache.keys()):
            if symbol in key:
                del self._cache[key]
