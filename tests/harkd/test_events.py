"""Tests for the in-process event bus."""

import asyncio

import pytest

from harkd.events import _QUEUE_MAXSIZE, Event, EventBus, get_event_bus


class TestEvent:
    def test_to_dict(self):
        e = Event(type="recording_started", data={"recording_id": "abc"}, ts=1000.0)
        d = e.to_dict()
        assert d["type"] == "recording_started"
        assert d["recording_id"] == "abc"
        assert d["ts"] == 1000.0

    def test_to_dict_defaults(self):
        e = Event(type="invalidate")
        d = e.to_dict()
        assert d["type"] == "invalidate"
        assert "ts" in d
        assert isinstance(d["ts"], float)


class TestEventBus:
    def test_subscribe_unsubscribe(self):
        bus = EventBus()
        q = bus.subscribe()
        assert q in bus._subscribers
        bus.unsubscribe(q)
        assert q not in bus._subscribers

    def test_unsubscribe_missing(self):
        bus = EventBus()
        q: asyncio.Queue[Event] = asyncio.Queue()
        bus.unsubscribe(q)  # should not raise

    def test_emit_to_multiple_subscribers(self):
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()

        event = Event(type="test", data={"x": 1})
        bus.emit(event)

        assert q1.qsize() == 1
        assert q2.qsize() == 1
        assert q1.get_nowait() is event
        assert q2.get_nowait() is event

    def test_emit_no_subscribers(self):
        bus = EventBus()
        bus.emit(Event(type="test"))  # should not raise

    def test_queue_full_drops_event(self):
        bus = EventBus()
        q = bus.subscribe()

        # Fill the queue
        for i in range(_QUEUE_MAXSIZE):
            bus.emit(Event(type="fill", data={"i": i}))

        assert q.qsize() == _QUEUE_MAXSIZE

        # Next emit should be silently dropped
        bus.emit(Event(type="overflow"))
        assert q.qsize() == _QUEUE_MAXSIZE

    @pytest.mark.asyncio
    async def test_async_get(self):
        bus = EventBus()
        q = bus.subscribe()

        event = Event(type="async_test", data={"val": 42})
        bus.emit(event)

        result = await asyncio.wait_for(q.get(), timeout=1.0)
        assert result.type == "async_test"
        assert result.data["val"] == 42

    def test_emit_during_unsubscribe(self):
        """Unsubscribing one queue mid-emit doesn't crash; remaining queue gets event."""
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()

        # Unsubscribe q1, then emit — q2 should still receive the event
        bus.unsubscribe(q1)
        event = Event(type="after_unsub")
        bus.emit(event)

        assert q1.qsize() == 0
        assert q2.qsize() == 1
        assert q2.get_nowait() is event

    def test_emit_snapshot_iteration(self):
        """Emit iterates a snapshot so unsubscribe during emit is safe."""
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()

        # Simulate concurrent unsubscribe by removing q1 while iterating
        # The snapshot-based emit should still deliver to q2
        original_put = q1.put_nowait

        def unsubscribe_and_put(item):
            bus.unsubscribe(q1)
            original_put(item)

        q1.put_nowait = unsubscribe_and_put

        event = Event(type="concurrent_test")
        bus.emit(event)

        # q1 got the event (it was in the snapshot), q2 also got it
        assert q1.qsize() == 1
        assert q2.qsize() == 1


class TestGetEventBus:
    def test_singleton_consistency(self):
        """get_event_bus() returns the same instance across calls."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
