"""Configuration management — loads YAML config with env-var overrides.

Environment variables are loaded from a `.env` file in the project root
(if present) using python-dotenv, then overridden by actual env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    # Load .env from project root (three levels up from stocks/config/settings.py)
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _env_path = _PROJECT_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)  # real env vars take precedence
except ImportError:
    pass  # python-dotenv is optional; env vars still work normally

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(user_config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from default YAML, optionally merged with user overrides.

    Priority (highest to lowest):
    1. Environment variables (ALGO_TRADING__section__key format)
    2. User config file
    3. Default config
    """
    # Load default config
    with open(_DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Merge user config if provided
    if user_config_path and Path(user_config_path).exists():
        with open(user_config_path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    # Apply environment variable overrides
    # Format: ALGO_TRADING__section__key=value
    prefix = "ALGO_TRADING__"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("__")
        _set_nested(config, parts, _parse_value(env_value))

    return config


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _parse_value(value: str) -> Any:
    """Attempt to parse an env-var string into the appropriate Python type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested config value."""
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current
