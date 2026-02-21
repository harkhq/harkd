"""API route handlers."""

from fastapi import FastAPI

from harkd.api.routes import chat, recordings, settings, voice_profiles, ws

__all__ = ["register_routes"]


def register_routes(app: FastAPI) -> None:
    """Register all API routes to the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    # Register all routers under /api/v1 prefix
    app.include_router(recordings.router, prefix="/api/v1")
    app.include_router(settings.router, prefix="/api/v1")
    app.include_router(voice_profiles.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(ws.router, prefix="/api/v1")
