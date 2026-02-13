"""Tests for error and health API models."""

from harkd.api.models.error import ErrorDetail, ErrorResponse
from harkd.api.models.health import HealthResponse


class TestErrorDetail:
    """Test ErrorDetail model."""

    def test_error_detail_minimal(self):
        """Test creating error detail with minimal fields."""
        detail = ErrorDetail(code="TEST_ERROR", message="Test error message")
        assert detail.code == "TEST_ERROR"
        assert detail.message == "Test error message"
        assert detail.details is None

    def test_error_detail_with_details(self):
        """Test creating error detail with additional context."""
        detail = ErrorDetail(
            code="RECORDING_NOT_FOUND",
            message="Recording not found",
            details={"recording_id": "rec-123"},
        )
        assert detail.code == "RECORDING_NOT_FOUND"
        assert detail.message == "Recording not found"
        assert detail.details is not None
        assert detail.details["recording_id"] == "rec-123"

    def test_error_detail_complex_details(self):
        """Test error detail with complex details object."""
        detail = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Multiple validation errors",
            details={
                "field_errors": [
                    {"field": "title", "error": "Too long"},
                    {"field": "duration", "error": "Must be positive"},
                ],
                "count": 2,
            },
        )
        assert detail.details is not None
        assert len(detail.details["field_errors"]) == 2
        assert detail.details["count"] == 2


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_error_response(self):
        """Test creating error response."""
        detail = ErrorDetail(code="NOT_FOUND", message="Resource not found")
        response = ErrorResponse(error=detail)
        assert response.error.code == "NOT_FOUND"
        assert response.error.message == "Resource not found"

    def test_error_response_with_details(self):
        """Test error response with additional context."""
        detail = ErrorDetail(
            code="RECORDING_IN_PROGRESS",
            message="A recording is already in progress",
            details={"active_recording_id": "rec-456"},
        )
        response = ErrorResponse(error=detail)
        assert response.error.details is not None
        assert response.error.details["active_recording_id"] == "rec-456"


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_health_response(self):
        """Test creating health response."""
        response = HealthResponse(version="0.1.0", uptime_seconds=3600.5)
        assert response.status == "ok"
        assert response.version == "0.1.0"
        assert response.uptime_seconds == 3600.5

    def test_health_response_zero_uptime(self):
        """Test health response with zero uptime."""
        response = HealthResponse(version="0.1.0", uptime_seconds=0)
        assert response.uptime_seconds == 0

    def test_health_response_custom_status(self):
        """Test health response with custom status."""
        response = HealthResponse(status="degraded", version="0.1.0", uptime_seconds=100)
        assert response.status == "degraded"
