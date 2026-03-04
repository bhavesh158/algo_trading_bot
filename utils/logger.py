"""Centralized logging configuration for the trading system."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    """Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Optional directory for log files. If None, logs to console only.
    """
    global _initialized
    if _initialized:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(log_path / f"trading_{date_str}.log")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    _initialized = True
    logging.getLogger(__name__).info("Logging initialized (level=%s)", level)
