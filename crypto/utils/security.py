"""Secure credential management for exchange API keys.

Credentials are loaded exclusively from environment variables / .env file.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def validate_live_trading_prerequisites(exchange: str = "binance") -> list[str]:
    """Check that all prerequisites for live trading are met.

    Returns a list of missing prerequisites (empty = all good).
    """
    issues: list[str] = []

    if not os.environ.get("EXCHANGE_API_KEY"):
        issues.append("EXCHANGE_API_KEY environment variable not set")
    if not os.environ.get("EXCHANGE_API_SECRET"):
        issues.append("EXCHANGE_API_SECRET environment variable not set")

    # Some exchanges (e.g. Kraken, KuCoin) also need a password/passphrase
    if exchange in ("kraken", "kucoin"):
        if not os.environ.get("EXCHANGE_PASSWORD"):
            issues.append(f"EXCHANGE_PASSWORD environment variable not set (required for {exchange})")

    return issues
