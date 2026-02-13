"""API models for error responses."""

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "ErrorDetail",
    "ErrorResponse",
]


class ErrorDetail(BaseModel):
    """Error detail information."""

    code: str = Field(..., description="Error code (e.g., RECORDING_NOT_FOUND)")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error context")


class ErrorResponse(BaseModel):
    """Standard error response wrapper."""

    error: ErrorDetail
