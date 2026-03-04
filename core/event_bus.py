"""Central event bus for inter-module communication.

Modules publish and subscribe to typed events. This enables loose coupling
between components — no module needs to import another directly.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Type

from core.events import Event

logger = logging.getLogger(__name__)

# Type alias for event handler functions
EventHandler = Callable[[Any], None]


class EventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[Type[Event], list[EventHandler]] = defaultdict(list)
        self._event_count: int = 0

    def subscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(
            "Subscribed %s to %s", handler.__qualname__, event_type.__name__
        )

    def unsubscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        """Remove a handler for a specific event type."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)
            logger.debug(
                "Unsubscribed %s from %s", handler.__qualname__, event_type.__name__
            )

    def publish(self, event: Event) -> None:
        """Publish an event to all registered handlers.

        Handlers are called synchronously in registration order.
        Exceptions in handlers are caught and logged to prevent cascading failures.
        """
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        self._event_count += 1

        if not handlers:
            logger.debug("No handlers for %s", event_type.__name__)
            return

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Error in handler %s for event %s",
                    handler.__qualname__,
                    event_type.__name__,
                )

    @property
    def event_count(self) -> int:
        """Total number of events published."""
        return self._event_count

    @property
    def subscriber_count(self) -> int:
        """Total number of active subscriptions."""
        return sum(len(handlers) for handlers in self._subscribers.values())

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._subscribers.clear()
        logger.info("Event bus cleared — all subscriptions removed")
