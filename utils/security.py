"""Secure credential management for broker API keys and secrets.

Credentials are loaded exclusively from environment variables.
Never hard-code secrets in config files or source code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrokerCredentials:
    """Immutable broker credentials."""
    api_key: str
    api_secret: str
    access_token: Optional[str] = None


def load_broker_credentials(prefix: str = "BROKER") -> BrokerCredentials:
    """Load broker credentials from environment variables.

    Expected env vars:
        {prefix}_API_KEY
        {prefix}_API_SECRET
        {prefix}_ACCESS_TOKEN (optional)

    Raises:
        EnvironmentError: If required credentials are missing.
    """
    api_key = os.environ.get(f"{prefix}_API_KEY")
    api_secret = os.environ.get(f"{prefix}_API_SECRET")
    access_token = os.environ.get(f"{prefix}_ACCESS_TOKEN")

    if not api_key or not api_secret:
        raise EnvironmentError(
            f"Missing required broker credentials. "
            f"Set {prefix}_API_KEY and {prefix}_API_SECRET environment variables."
        )

    logger.info("Broker credentials loaded (prefix=%s)", prefix)
    return BrokerCredentials(
        api_key=api_key,
        api_secret=api_secret,
        access_token=access_token,
    )


def validate_live_trading_prerequisites() -> list[str]:
    """Check that all prerequisites for live trading are met.

    Returns a list of missing prerequisites (empty = all good).
    """
    issues: list[str] = []

    if not os.environ.get("BROKER_API_KEY"):
        issues.append("BROKER_API_KEY environment variable not set")
    if not os.environ.get("BROKER_API_SECRET"):
        issues.append("BROKER_API_SECRET environment variable not set")

    return issues
