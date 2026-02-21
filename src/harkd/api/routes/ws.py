"""WebSocket endpoint for real-time event streaming."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from harkd.events import get_event_bus

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Accept a WebSocket connection and stream events."""
    await ws.accept()
    bus = get_event_bus()
    queue = bus.subscribe()
    logger.info("WebSocket client connected")
    try:
        while True:
            event = await queue.get()
            await ws.send_text(json.dumps(event.to_dict()))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.debug("WebSocket send error", exc_info=True)
    finally:
        bus.unsubscribe(queue)
