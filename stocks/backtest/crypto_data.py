"""CCXT public-kline fetcher for crypto validation (no API keys needed)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"


def _cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return DATA_DIR / f"CRYPTO_{safe}_{interval}.csv"


def fetch_klines(symbol: str, interval: str = "15m", days: int = 365,
                 force: bool = False) -> pd.DataFrame:
    """Paginated public OHLCV; cached. Returns UTC-naive indexed OHLCV."""
    import ccxt

    path = _cache_path(symbol, interval)
    if path.exists() and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            return df
    ex = ccxt.binance({"enableRateLimit": True})
    ms = {"15m": 15 * 60 * 1000, "1h": 3600 * 1000}[interval]
    since = ex.milliseconds() - days * 24 * 3600 * 1000
    all_rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, interval, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + ms
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.warning("Cached %s %s: %d candles", symbol, interval, len(df))
    return df
