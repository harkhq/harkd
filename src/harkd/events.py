"""In-process async event bus for broadcasting events to WebSocket clients."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field

__all__ = ["Event", "EventBus", "get_event_bus"]

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 256


@dataclass
class Event:
    """A broadcast event."""

    type: str
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": self.type, "ts": self.ts, **self.data}


class EventBus:
    """Pub/sub event bus. Each subscriber gets an asyncio.Queue."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(q)

    def emit(self, event: Event) -> None:
        """Broadcast event to all subscribers. Non-blocking (put_nowait).

        Safe to call from non-async threads under CPython's GIL.
        Drops events for slow consumers whose queues are full.
        """
        for q in self._subscribers[:]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("Dropping event %s for slow consumer", event.type)


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the module-level singleton EventBus."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
