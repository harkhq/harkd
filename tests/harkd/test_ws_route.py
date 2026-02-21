"""Integration tests for the WebSocket endpoint."""

from starlette.testclient import TestClient

from harkd.events import Event, get_event_bus


def _make_app():
    """Create a minimal FastAPI app with just the WS route."""
    from fastapi import FastAPI

    from harkd.api.routes.ws import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


class TestWebSocketEndpoint:
    def test_receive_event(self):
        """Connect to /api/v1/ws, emit an event, verify JSON received."""
        app = _make_app()
        bus = get_event_bus()

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws") as ws:
                event = Event(
                    type="invalidate",
                    data={"entity": "recordings"},
                    ts=1000.0,
                )
                bus.emit(event)

                data = ws.receive_json()
                assert data["type"] == "invalidate"
                assert data["entity"] == "recordings"
                assert data["ts"] == 1000.0

    def test_disconnect_cleans_up_subscriber(self):
        """After WS disconnect, bus should have no subscribers from that connection."""
        app = _make_app()
        bus = get_event_bus()

        initial_count = len(bus._subscribers)

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws") as _ws:
                assert len(bus._subscribers) == initial_count + 1

            # After disconnect
            assert len(bus._subscribers) == initial_count

    def test_multiple_events(self):
        """Multiple events are received in order."""
        app = _make_app()
        bus = get_event_bus()

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws") as ws:
                for i in range(3):
                    bus.emit(Event(type="test", data={"i": i}, ts=float(i)))

                for i in range(3):
                    data = ws.receive_json()
                    assert data["type"] == "test"
                    assert data["i"] == i
