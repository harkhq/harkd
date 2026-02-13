"""REST API routes for recordings."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from harkd.api.deps import get_recording_service
from harkd.api.models.recording import (
    RecordingCreate,
    RecordingListResponse,
    RecordingResponse,
    RecordingStatus,
    RecordingUpdate,
)
from harkd.exceptions import (
    InvalidStateError,
    NoActiveRecordingError,
    RecordingInProgressError,
    RecordingNotFoundError,
)
from harkd.services.recording_service import RecordingService

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recordings", tags=["recordings"])

ServiceDep = Annotated[RecordingService, Depends(get_recording_service)]


@router.post(
    "",
    response_model=RecordingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start new recording",
    description=(
        "Start a new recording session. Returns immediately"
        " with recording ID and status 'recording'."
    ),
)
async def start_recording(
    request: RecordingCreate,
    service: ServiceDep,
) -> RecordingResponse:
    """Start a new recording.

    Args:
        request: Recording creation request
        service: Recording service (injected)

    Returns:
        Initial recording response with status "recording"

    Raises:
        HTTPException 409: If a recording is already in progress
        HTTPException 422: If request validation fails
    """
    logger.info(f"POST /recordings - Starting new recording: {request}")

    try:
        result = await service.start_recording(request)
        logger.info(f"Recording {result.id} started successfully")
        return result
    except RecordingInProgressError as e:
        logger.warning(f"Cannot start recording: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                }
            },
        ) from e


@router.get(
    "",
    response_model=RecordingListResponse,
    summary="List recordings",
    description="Get a paginated list of recordings with optional filtering.",
)
async def list_recordings(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of results")] = 50,
    offset: Annotated[int, Query(ge=0, description="Result offset for pagination")] = 0,
    status_filter: Annotated[
        RecordingStatus | None,
        Query(alias="status", description="Filter by status"),
    ] = None,
    search: Annotated[str | None, Query(description="Search in title/transcript")] = None,
) -> RecordingListResponse:
    """List recordings with filtering and pagination.

    Args:
        service: Recording service (injected)
        limit: Maximum number of results (1-100)
        offset: Result offset for pagination
        status_filter: Filter by recording status
        search: Search query for title/transcript

    Returns:
        Paginated list of recordings
    """
    logger.debug(
        f"GET /recordings - limit={limit}, offset={offset}, status={status_filter}, search={search}"
    )

    result = await service.list_recordings(
        limit=limit,
        offset=offset,
        status=status_filter,
        search=search,
    )

    logger.debug(f"Returning {len(result.recordings)} recordings (total={result.total})")
    return result


@router.get(
    "/active",
    response_model=RecordingResponse,
    summary="Get active recording",
    description="Get the currently active recording with real-time duration.",
)
async def get_active_recording(
    service: ServiceDep,
) -> RecordingResponse:
    """Get the currently active recording.

    Returns:
        Recording response with real-time duration

    Raises:
        HTTPException 409: If no recording is currently active
    """
    logger.debug("GET /recordings/active")

    try:
        result = await service.get_active_recording()
        return result
    except NoActiveRecordingError as e:
        logger.warning("No active recording")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                }
            },
        ) from e


@router.post(
    "/active/stop",
    response_model=RecordingResponse,
    summary="Stop active recording",
    description=(
        "Stop the currently active recording and begin processing. No recording ID required."
    ),
)
async def stop_active_recording(
    service: ServiceDep,
) -> RecordingResponse:
    """Stop the currently active recording and begin processing.

    Returns:
        Recording response with status "processing"

    Raises:
        HTTPException 409: If no recording is currently active
    """
    logger.info("POST /recordings/active/stop")

    try:
        result = await service.stop_active_recording()
        logger.info(f"Active recording {result.id} stopped, processing started")
        return result
    except NoActiveRecordingError as e:
        logger.warning("No active recording to stop")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                }
            },
        ) from e


@router.get(
    "/{recording_id}",
    response_model=RecordingResponse,
    summary="Get recording",
    description="Get recording by ID. For active recordings, duration is updated in real-time.",
)
async def get_recording(
    recording_id: str,
    service: ServiceDep,
) -> RecordingResponse:
    """Get recording by ID.

    Args:
        recording_id: Recording ID
        service: Recording service (injected)

    Returns:
        Recording response with current status

    Raises:
        HTTPException 404: If recording not found
    """
    logger.debug(f"GET /recordings/{recording_id}")

    try:
        result = await service.get_recording(recording_id)
        return result
    except RecordingNotFoundError as e:
        logger.warning(f"Recording not found: {recording_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                }
            },
        ) from e


@router.patch(
    "/{recording_id}",
    response_model=RecordingResponse,
    summary="Update recording",
    description="Update recording metadata (title, speakers). Only for completed recordings.",
)
async def update_recording(
    recording_id: str,
    update: RecordingUpdate,
    service: ServiceDep,
) -> RecordingResponse:
    """Update recording metadata.

    Args:
        recording_id: Recording ID
        update: Update request
        service: Recording service (injected)

    Returns:
        Updated recording response

    Raises:
        HTTPException 404: If recording not found
        HTTPException 409: If recording not complete
    """
    logger.info(f"PATCH /recordings/{recording_id} - {update}")

    try:
        result = await service.update_recording(recording_id, update)
        logger.info(f"Recording {recording_id} updated successfully")
        return result
    except RecordingNotFoundError as e:
        logger.warning(f"Recording not found: {recording_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                }
            },
        ) from e
    except InvalidStateError as e:
        logger.warning(f"Invalid state for update: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                }
            },
        ) from e


@router.delete(
    "/{recording_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete recording",
    description="Delete/cancel recording. Works for any status (recording, processing, complete).",
)
async def delete_recording(
    recording_id: str,
    service: ServiceDep,
) -> None:
    """Delete or cancel a recording.

    - If status = "recording": stop and discard
    - If status = "processing": cancel processing
    - If status = "complete": delete from storage

    Args:
        recording_id: Recording ID
        service: Recording service (injected)

    Raises:
        HTTPException 404: If recording not found
    """
    logger.info(f"DELETE /recordings/{recording_id}")

    try:
        await service.delete_recording(recording_id)
        logger.info(f"Recording {recording_id} deleted successfully")
    except RecordingNotFoundError as e:
        logger.warning(f"Recording not found: {recording_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                }
            },
        ) from e
