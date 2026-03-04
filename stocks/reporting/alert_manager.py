"""Alert / Monitoring System (PRD §22).

Monitors operational health and generates alerts for:
- Connection failures
- Broker errors
- Abnormal losses
- System crashes

Alerts are published to configured channels (console, file, webhook).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from stocks.core.enums import AlertSeverity
from stocks.core.event_bus import EventBus
from stocks.core.events import AlertEvent
from stocks.core.models import Alert

logger = logging.getLogger(__name__)


class AlertManager:
    """Collects and dispatches system alerts."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus

        alert_config = config.get("alerts", {})
        self._enabled = alert_config.get("enabled", True)
        self._channels = alert_config.get("channels", ["console"])
        self._alert_on_trade = alert_config.get("alert_on_trade", True)
        self._alert_on_risk = alert_config.get("alert_on_risk_breach", True)
        self._alert_on_error = alert_config.get("alert_on_error", True)

        self._alerts: list[Alert] = []
        self._alert_log_path = Path(
            config.get("system", {}).get("data_dir", "./data_store")
        ) / "alerts.jsonl"

        # Subscribe to AlertEvents on the event bus
        self.event_bus.subscribe(AlertEvent, self._handle_alert_event)

        logger.info("AlertManager initialized (channels=%s)", self._channels)

    def _handle_alert_event(self, event: AlertEvent) -> None:
        """Handle incoming alert events from the event bus."""
        if event.alert and self._enabled:
            self.dispatch(event.alert)

    def dispatch(self, alert: Alert) -> None:
        """Route an alert to all configured channels."""
        self._alerts.append(alert)

        for channel in self._channels:
            if channel == "console":
                self._console_alert(alert)
            elif channel == "file":
                self._file_alert(alert)
            elif channel == "webhook":
                self._webhook_alert(alert)

    def _console_alert(self, alert: Alert) -> None:
        """Print alert to console via logging."""
        severity_map = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
        }
        log_fn = severity_map.get(alert.severity, logger.info)
        log_fn(
            "🔔 ALERT [%s] %s: %s",
            alert.severity.name, alert.source, alert.message,
        )

    def _file_alert(self, alert: Alert) -> None:
        """Append alert to JSONL file."""
        try:
            self._alert_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._alert_log_path, "a") as f:
                entry = {
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity.name,
                    "source": alert.source,
                    "message": alert.message,
                }
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.exception("Failed to write alert to file")

    def _webhook_alert(self, alert: Alert) -> None:
        """Send alert via webhook (placeholder for future implementation)."""
        # TODO: Implement webhook integration (Slack, Telegram, etc.)
        logger.debug("Webhook alert not yet implemented: %s", alert.message)

    def create_alert(
        self, severity: AlertSeverity, source: str, message: str,
    ) -> Alert:
        """Create and dispatch a new alert."""
        alert = Alert(severity=severity, source=source, message=message)
        self.dispatch(alert)
        return alert

    @property
    def recent_alerts(self) -> list[Alert]:
        """Get the last 50 alerts."""
        return self._alerts[-50:]

    @property
    def critical_alerts(self) -> list[Alert]:
        """Get all unacknowledged critical alerts."""
        return [
            a for a in self._alerts
            if a.severity == AlertSeverity.CRITICAL and not a.acknowledged
        ]
