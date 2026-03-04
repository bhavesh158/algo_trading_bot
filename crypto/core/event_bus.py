"""Central event bus for inter-module communication.

Modules publish and subscribe to typed events. This enables loose coupling
between components — no module needs to import another directly.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Type

from crypto.core.events import Event

logger = logging.getLogger(__name__)

EventHandler = Callable[[Any], None]


class EventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[Type[Event], list[EventHandler]] = defaultdict(list)
        self._event_count: int = 0

    def subscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        self._event_count += 1

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Error in handler %s for event %s",
                    handler.__qualname__, event_type.__name__,
                )

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def subscriber_count(self) -> int:
        return sum(len(h) for h in self._subscribers.values())

    def clear(self) -> None:
        self._subscribers.clear()
