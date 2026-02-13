"""API models for health check."""

from pydantic import BaseModel

__all__ = [
    "HealthResponse",
]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    uptime_seconds: float
