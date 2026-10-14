"""Historical data acquisition + disk cache for the backtest harness.

Uses yfinance directly (same source as the live YahooProvider).
Cache lives in stocks/backtest/data/ (CSV per symbol/timeframe).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"

# Symbols known-dead on Yahoo (verified 2026-09-03, 0 rows returned)
DEAD_SYMBOLS = {"GSPL.NS", "LTIM.NS", "TATAMOTORS.NS"}

# Blocked at live execution; backtest mirrors tradability
BLACKLISTED = {"VEDL.NS", "ETERNAL.NS", "JIOFIN.NS"}


def tradable_universe(pool: list[str]) -> list[str]:
    """Candidate pool minus dead and blacklisted symbols."""
    out = [s for s in pool if s not in DEAD_SYMBOLS and s not in BLACKLISTED]
    skipped = [s for s in pool if s not in out]
    logger.info("Universe: %d/%d symbols (skipped: %s)", len(out), len(pool), skipped)
    return out


def _cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace(".", "_")
    return DATA_DIR / f"{safe}_{interval}.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase cols, IST-naive DatetimeIndex, sorted, OHLCV only."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    # yfinance multi-symbol frames nest under ticker; caller slices first
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        except Exception:
            try:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
            except Exception:
                pass
    df = df.sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].dropna(subset=["close"])


def fetch_symbol(symbol: str, interval: str, period: str, force: bool = False) -> pd.DataFrame:
    """Fetch one symbol/interval, using cache unless force=True."""
    import yfinance as yf

    path = _cache_path(symbol, interval)
    if path.exists() and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            return df
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        logger.warning("No data for %s (%s)", symbol, interval)
        return pd.DataFrame()
    df = _normalize(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def fetch_all(symbols: list[str], interval: str = "15m", period: str = "60d",
              pause: float = 0.3, force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch (cached) intraday history for a symbol list."""
    out: dict[str, pd.DataFrame] = {}
    for i, s in enumerate(symbols):
        df = fetch_symbol(s, interval, period, force=force)
        if not df.empty:
            out[s] = df
        if (i + 1) % 20 == 0:
            logger.info("Fetched %d/%d (%s)", i + 1, len(symbols), interval)
        time.sleep(pause)
    logger.info("Have %s data for %d/%d symbols", interval, len(out), len(symbols))
    return out


def fetch_daily_index(symbol: str = "^NSEI", period: str = "1y") -> pd.DataFrame:
    """Daily bars for the market-regime approximation."""
    return fetch_symbol(symbol, "1d", period)
