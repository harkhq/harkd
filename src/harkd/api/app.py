"""FastAPI application creation and configuration."""

import logging
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


def _configure_logging(level: str) -> None:
    """Configure the harkd logger hierarchy.

    Uvicorn only configures its own loggers. This sets up a handler
    for the ``harkd`` namespace so application logs reach the console.
    """
    harkd_logger = logging.getLogger("harkd")
    if not harkd_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
        harkd_logger.addHandler(handler)
    harkd_logger.setLevel(level.upper())

    # Silence noisy websockets debug output (keepalive pings, connection lifecycle)
    for _ws_logger in ("websockets", "websockets.client", "websockets.server"):
        logging.getLogger(_ws_logger).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Args:
        app: FastAPI application instance
    """
    logger = logging.getLogger(__name__)
    global _startup_time
    _startup_time = time.time()

    # Startup: eagerly create and start processing workers, run recovery
    from harkd.api.deps import get_processing_worker_eager

    settings = get_settings()
    storage_path = settings.storage.base_path
    worker = get_processing_worker_eager(storage_path, settings)
    worker.start()
    recovered = await worker.recover()
    if recovered:
        logger.info(f"Recovered {recovered} stuck/retryable recordings")

    # Recover orphaned infrastructure from previous runs
    if hasattr(worker._backend, "recover_orphans"):
        try:
            await worker._backend.recover_orphans()
        except Exception:
            logger.warning("Failed to recover orphaned infrastructure", exc_info=True)

    yield

    # Shutdown: gracefully stop workers
    logger.info("Shutting down processing workers...")
    from harkd.api.deps import _processing_workers

    for w in _processing_workers.values():
        await w.shutdown(timeout=10.0)

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

    _configure_logging(settings.logging.level)

    app = FastAPI(
        title="harkd",
        description=(
            "Meeting minutes, summaries, transcription & diarization,"
            " and task extraction daemon with REST API"
        ),
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

    # Serve UI static files (no-op if dist not present)
    _mount_ui(app)

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
            "RETRY_NOT_ALLOWED": 409,
            "CHAT_THREAD_NOT_FOUND": 404,
            "LLM_NOT_CONFIGURED": 503,
        }.get(exc.code, 500)

        error_response = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                details=exc.details if exc.details else None,
            )
        )

        return JSONResponse(status_code=status_code, content=error_response.model_dump())


def _mount_ui(app: FastAPI) -> None:
    """Mount the vendored UI dist if present.

    Serves ``/assets/*`` as static files and falls back to ``index.html``
    for all other non-API paths (SPA client-side routing).  If no dist
    directory is found the function is a no-op so the API still works
    standalone during development.
    """
    from pathlib import Path

    dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
    if not dist.is_dir() or not (dist / "index.html").exists():
        return

    logging.getLogger(__name__).info("Serving UI from %s", dist)

    from starlette.responses import FileResponse
    from starlette.staticfiles import StaticFiles

    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="ui-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str) -> FileResponse:
        return FileResponse(str(dist / "index.html"))


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
