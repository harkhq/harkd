"""FastAPI application creation and configuration."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from harkd import __version__
from harkd.api.models.error import ErrorDetail, ErrorResponse
from harkd.api.models.health import HealthResponse
from harkd.api.routes import register_routes
from harkd.config import HarkdSettings, get_settings
from harkd.exceptions import HarkdError

__all__ = ["create_app"]

# Track startup time
_startup_time: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Args:
        app: FastAPI application instance
    """
    import asyncio
    import logging

    logger = logging.getLogger(__name__)
    global _startup_time
    _startup_time = time.time()

    yield

    # Cleanup on shutdown - cancel all background processing tasks
    logger.info("Shutting down - cancelling background tasks...")
    from harkd.api.deps import _recording_services

    for service in _recording_services.values():
        if hasattr(service, "_processing_tasks"):
            for recording_id, task in list(service._processing_tasks.items()):
                if not task.done():
                    logger.info(f"Cancelling background task for recording {recording_id}")
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except asyncio.CancelledError:
                        logger.info(f"Task for recording {recording_id} cancelled successfully")
                    except TimeoutError:
                        logger.warning(f"Task for recording {recording_id} did not cancel in time")
                    except Exception as e:
                        logger.error(
                            f"Exception during task cancellation for {recording_id}: {e}",
                            exc_info=True,
                        )

    logger.info("Shutdown complete")


def create_app(settings: HarkdSettings | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        settings: Optional settings override (defaults to get_settings())

    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="harkd",
        description="Voice recording and transcription daemon",
        version=__version__,
        lifespan=lifespan,
    )

    # Configure CORS
    if settings.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors.origins,
            allow_credentials=settings.cors.allow_credentials,
            allow_methods=settings.cors.allow_methods,
            allow_headers=settings.cors.allow_headers,
        )

    # Add exception handlers
    _add_exception_handlers(app)

    # Add health endpoint
    _add_health_endpoint(app)

    # Register API routes
    register_routes(app)

    return app


def _add_exception_handlers(app: FastAPI) -> None:
    """Add custom exception handlers to the app.

    Args:
        app: FastAPI application
    """

    @app.exception_handler(HarkdError)
    async def harkd_exception_handler(request: Request, exc: HarkdError):
        """Convert HarkdError to JSON response."""
        status_code = {
            "RECORDING_IN_PROGRESS": 409,
            "NO_ACTIVE_RECORDING": 409,
            "RECORDING_NOT_FOUND": 404,
            "VOICE_PROFILE_NOT_FOUND": 404,
            "STORAGE_ERROR": 500,
            "INVALID_STATE": 409,
            "NO_MICROPHONE": 422,
            "NO_LOOPBACK_DEVICE": 422,
        }.get(exc.code, 500)

        error_response = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                details=exc.details if exc.details else None,
            )
        )

        return JSONResponse(status_code=status_code, content=error_response.model_dump())


def _add_health_endpoint(app: FastAPI) -> None:
    """Add health check endpoint to the app.

    Args:
        app: FastAPI application
    """

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        """Health check endpoint.

        Returns:
            Health status with version and uptime
        """
        global _startup_time
        if _startup_time is None:
            _startup_time = time.time()
        uptime = time.time() - _startup_time
        return HealthResponse(status="ok", version=__version__, uptime_seconds=uptime)
