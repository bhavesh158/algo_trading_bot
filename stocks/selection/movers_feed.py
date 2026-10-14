"""NSE movers feed — dynamic pre-market candidate source (no API key).

Pulls NSE's own free live-analysis endpoints (top gainers/losers by
index + most-active by volume/value) and maps them to Yahoo-style
`.NS` symbols for the watchlist builder.

Fail-open by design: any fetch/parse error returns empty lists and the
selector falls back to the static pool. Called once per pre-market, so
no intraday latency or rate-limit concern.

Note: at 08:45 pre-market these endpoints still show the previous
session (continuation/gap setups) — still the best free proxy for
"where money is moving" without scraping.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE = "https://www.nseindia.com"
_TIMEOUT = 20


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def _to_ns(symbol: str) -> str | None:
    sym = (symbol or "").strip().upper()
    if not sym or sym.startswith("NIFTY") or " " in sym:
        return None
    return f"{sym}.NS"


def fetch_movers(
    max_per_source: int = 20,
    min_ltp: float = 50.0,
    max_ltp: float = 10000.0,
) -> dict[str, list[str]]:
    """Return {'gainers': [...], 'losers': [...], 'active': [...]} (.NS symbols).

    Never raises — returns empty lists on any failure.
    """
    out: dict[str, list[str]] = {"gainers": [], "losers": [], "active": []}
    try:
        s = _session()
        s.get(_BASE, timeout=_TIMEOUT)  # cookie handshake

        def _rows(url: str) -> list[dict[str, Any]]:
            r = s.get(url, timeout=_TIMEOUT)
            r.raise_for_status()
            j = r.json()
            if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
                return j["data"]
            rows: list[dict[str, Any]] = []
            for v in j.values():
                if isinstance(v, dict) and isinstance(v.get("data"), list):
                    rows.extend(v["data"])
            return rows

        def _pick(rows: list[dict[str, Any]], price_keys: tuple[str, ...]) -> list[str]:
            picked: list[str] = []
            for r in rows:
                if r.get("series", "EQ") != "EQ":
                    continue
                ns = _to_ns(str(r.get("symbol", "")))
                if ns is None or ns in picked:
                    continue
                px = 0.0
                for k in price_keys:
                    try:
                        px = float(r.get(k, 0) or 0)
                    except (TypeError, ValueError):
                        px = 0.0
                    if px > 0:
                        break
                if not (min_ltp <= px <= max_ltp):
                    continue
                picked.append(ns)
                if len(picked) >= max_per_source:
                    break
            return picked

        g = _rows(f"{_BASE}/api/live-analysis-variations?index=gainers")
        out["gainers"] = _pick(g, ("ltp", "high_price", "open_price"))
        # No losers bucket on this endpoint ("Missing index or key") — derive
        # losers from most-active breadth (rows carry pChange).
        out["losers"] = []
        act: list[dict[str, Any]] = []
        for idx in ("volume", "value"):
            try:
                act.extend(
                    _rows(f"{_BASE}/api/live-analysis-most-active-securities?index={idx}")
                )
            except Exception:
                logger.debug("movers: most-active %s unavailable", idx, exc_info=True)
        try:
            neg = [r for r in act if float(r.get("pChange", 0) or 0) < 0]
            neg.sort(key=lambda r: float(r.get("pChange", 0) or 0))
            for r in neg:  # most-active rows lack ltp; use range midpoint
                try:
                    hi = float(r.get("dayHigh", 0) or 0)
                    lo_p = float(r.get("dayLow", 0) or 0)
                    if hi > 0 and lo_p > 0:
                        r["ltp"] = (hi + lo_p) / 2
                except (TypeError, ValueError):
                    pass
            out["losers"] = _pick(neg, ("ltp",))
        except (TypeError, ValueError):
            logger.debug("movers: loser derivation failed", exc_info=True)
        # Most-active rows lack ltp; estimate from day range midpoint.
        ranked: list[str] = []
        for r in act:
            if r.get("series", "EQ") != "EQ" and "series" in r:
                continue
            ns = _to_ns(str(r.get("symbol", "")))
            if ns is None or ns in ranked:
                continue
            try:
                hi = float(r.get("dayHigh", 0) or 0)
                lo_p = float(r.get("dayLow", 0) or 0)
                px = (hi + lo_p) / 2 if hi > 0 and lo_p > 0 else 0.0
            except (TypeError, ValueError):
                px = 0.0
            if px > 0 and not (min_ltp <= px <= max_ltp):
                continue
            ranked.append(ns)
            if len(ranked) >= max_per_source:
                break
        out["active"] = ranked

        total = sum(len(v) for v in out.values())
        logger.info(
            "Movers feed: gainers=%d losers=%d active=%d",
            len(out["gainers"]), len(out["losers"]), len(out["active"]),
        )
        return out
    except Exception:
        logger.warning("Movers feed failed — falling back to static pool", exc_info=True)
        return out
