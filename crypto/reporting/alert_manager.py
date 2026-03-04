"""Monitoring and Alert System (PRD §18).

Monitors system health and generates alerts for critical events.
"""

from __future__ import annotations

import logging
from typing import Any

from crypto.core.enums import AlertSeverity
from crypto.core.event_bus import EventBus
from crypto.core.events import AlertEvent, OrderEvent, RiskEvent, ExchangeConnectionEvent

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages system alerts and notifications."""

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus

        alerts_cfg = config.get("alerts", {})
        self._enabled = alerts_cfg.get("enabled", True)
        self._alert_on_trade = alerts_cfg.get("alert_on_trade", True)
        self._alert_on_risk = alerts_cfg.get("alert_on_risk_breach", True)
        self._alert_on_error = alerts_cfg.get("alert_on_error", True)
        self._alert_on_connection = alerts_cfg.get("alert_on_connection_loss", True)

        if self._enabled:
            event_bus.subscribe(AlertEvent, self._handle_alert)
            if self._alert_on_trade:
                event_bus.subscribe(OrderEvent, self._handle_order)
            if self._alert_on_risk:
                event_bus.subscribe(RiskEvent, self._handle_risk)
            event_bus.subscribe(ExchangeConnectionEvent, self._handle_connection)

        logger.info("AlertManager initialized (enabled=%s)", self._enabled)

    def _handle_alert(self, event: AlertEvent) -> None:
        if not event.alert:
            return
        alert = event.alert
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical("🚨 ALERT [%s]: %s", alert.source, alert.message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning("⚠️  ALERT [%s]: %s", alert.source, alert.message)
        else:
            logger.info("ℹ️  ALERT [%s]: %s", alert.source, alert.message)

    def _handle_order(self, event: OrderEvent) -> None:
        if not event.order:
            return
        order = event.order
        logger.info(
            "📋 TRADE: %s %s qty=%.6f status=%s",
            order.side.name, order.symbol, order.quantity, order.status.name,
        )

    def _handle_risk(self, event: RiskEvent) -> None:
        logger.warning("⛔ RISK: %s — action=%s", event.message, event.action)

    def _handle_connection(self, event: ExchangeConnectionEvent) -> None:
        if not event.connected and self._alert_on_connection:
            logger.critical("🔌 EXCHANGE DISCONNECTED: %s — %s", event.exchange, event.message)
        elif event.connected:
            logger.info("🔌 Exchange connected: %s", event.exchange)
