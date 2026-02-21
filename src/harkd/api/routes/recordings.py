"""REST API routes for recordings."""

import asyncio
import io
import logging
import struct
import wave
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from harkd.api.deps import get_recording_service
from harkd.api.models.recording import (
    ActiveRecordingUpdate,
    RecordingCreate,
    RecordingListResponse,
    RecordingResponse,
    RecordingStatus,
    RecordingUpdate,
    RetryOverrides,
)
from harkd.exceptions import (
    InvalidStateError,
    NoActiveRecordingError,
    NoLoopbackDeviceError,
    NoMicrophoneError,
    RecordingInProgressError,
    RecordingNotFoundError,
    RetryNotAllowedError,
    VoiceProfileNotFoundError,
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
    sort: Annotated[Literal["asc", "desc"], Query(description="Sort by date")] = "desc",
    created_after: Annotated[
        datetime | None, Query(description="Filter: created at or after (ISO 8601)")
    ] = None,
    created_before: Annotated[
        datetime | None, Query(description="Filter: created at or before (ISO 8601)")
    ] = None,
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
        f"GET /recordings - limit={limit}, offset={offset}, status={status_filter}, "
        f"search={search}, sort={sort}, created_after={created_after}, "
        f"created_before={created_before}"
    )

    result = await service.list_recordings(
        limit=limit,
        offset=offset,
        status=status_filter,
        search=search,
        sort=sort,
        created_after=created_after,
        created_before=created_before,
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


@router.patch(
    "/active",
    response_model=RecordingResponse,
    summary="Update active recording",
    description="Update the active recording (title, mic/speaker toggles).",
)
async def update_active_recording(
    request: ActiveRecordingUpdate,
    service: ServiceDep,
) -> RecordingResponse:
    """Update the active recording (title, mic/speaker toggles).

    Returns:
        Updated recording response

    Raises:
        HTTPException 409: If no recording is active or device unavailable
    """
    logger.info(f"PATCH /recordings/active - {request}")

    try:
        return await service.update_active_recording(request)
    except NoActiveRecordingError as e:
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
    except (NoMicrophoneError, NoLoopbackDeviceError) as e:
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


@router.post(
    "/{recording_id}/retry",
    response_model=RecordingResponse,
    summary="Retry or reprocess recording",
    description="Re-enqueue a failed or completed recording for processing.",
)
async def retry_recording(
    recording_id: str,
    service: ServiceDep,
    overrides: RetryOverrides | None = None,
) -> RecordingResponse:
    """Retry a failed or reprocess a completed recording.

    Args:
        recording_id: Recording ID
        service: Recording service (injected)
        overrides: Optional diarization overrides for re-processing

    Returns:
        Updated recording response with status "processing"

    Raises:
        HTTPException 404: If recording not found
        HTTPException 409: If recording is not in error/complete state or already queued
    """
    logger.info(f"POST /recordings/{recording_id}/retry (overrides={overrides})")

    try:
        overrides_dict = overrides.model_dump(exclude_none=True) if overrides else None
        result = await service.retry_recording(recording_id, overrides=overrides_dict)
        logger.info(f"Recording {recording_id} retry enqueued")
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
    except RetryNotAllowedError as e:
        logger.warning(f"Retry not allowed: {e}")
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
    description=(
        "Update recording metadata (title, speakers). Title can be updated in any state."
        " Speaker names can only be updated for completed recordings."
    ),
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
    except VoiceProfileNotFoundError as e:
        logger.warning(f"Voice profile not found during update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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


@router.get(
    "/{recording_id}/audio/clip",
    response_class=Response,
    summary="Get audio clip",
    description="Extract a time-range clip from a recording's audio file.",
)
async def get_audio_clip(
    recording_id: str,
    service: ServiceDep,
    start: float = Query(..., ge=0, description="Start time in seconds"),
    end: float = Query(..., gt=0, description="End time in seconds"),
) -> Response:
    """Extract an audio clip from a recording.

    Args:
        recording_id: Recording ID
        service: Recording service (injected)
        start: Start time in seconds
        end: End time in seconds

    Returns:
        WAV audio response

    Raises:
        HTTPException 404: If recording or audio file not found
        HTTPException 422: If start >= end
    """
    if end <= start:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "INVALID_RANGE",
                    "message": "end must be greater than start",
                }
            },
        )

    # Verify recording exists
    storage_recording = await service.storage.get(recording_id)
    if storage_recording is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "RECORDING_NOT_FOUND",
                    "message": f"Recording not found: {recording_id}",
                }
            },
        )

    audio_path = service.storage.base_path / "recordings" / recording_id / "audio.wav"
    if not await asyncio.to_thread(audio_path.exists):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "AUDIO_NOT_FOUND",
                    "message": "Audio file not found for this recording",
                }
            },
        )

    def _extract_clip() -> bytes:
        with wave.open(str(audio_path), "rb") as wf:
            framerate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()

            start_frame = int(start * framerate)
            end_frame = int(end * framerate)

            # Clamp to file bounds
            start_frame = min(start_frame, n_frames)
            end_frame = min(end_frame, n_frames)

            if start_frame >= end_frame:
                end_frame = start_frame

            wf.setpos(start_frame)
            frames = wf.readframes(end_frame - start_frame)

        # Downmix stereo to mono (recordings are L=mic, R=speaker)
        if n_channels == 2 and sampwidth == 2:
            n_samples = len(frames) // (n_channels * sampwidth)
            stereo = struct.unpack(f"<{n_samples * 2}h", frames)
            mono = struct.pack(
                f"<{n_samples}h",
                *(
                    max(-32768, min(32767, (stereo[i] + stereo[i + 1]) // 2))
                    for i in range(0, len(stereo), 2)
                ),
            )
            frames = mono
            n_channels = 1

        # Upsample to 48kHz for reliable browser playback.
        # Browser audio pipelines run at 44.1/48kHz; feeding 16kHz
        # can cause audible speed/pitch glitches in some browsers.
        target_rate = 48000
        if n_channels == 1 and sampwidth == 2 and framerate != target_rate and frames:
            import numpy as np

            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            n_out = int(len(samples) * target_rate / framerate)
            x_new = np.linspace(0, len(samples) - 1, n_out)
            upsampled = np.interp(x_new, np.arange(len(samples)), samples)
            frames = np.clip(upsampled, -32768, 32767).astype(np.int16).tobytes()
            framerate = target_rate

        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(n_channels)
            out.setsampwidth(sampwidth)
            out.setframerate(framerate)
            out.writeframes(frames)

        return buf.getvalue()

    try:
        content = await asyncio.to_thread(_extract_clip)
        return Response(
            content=content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-store",
            },
        )
    except wave.Error as e:
        logger.error(f"Failed to read audio file for {recording_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "AUDIO_READ_ERROR",
                    "message": "Failed to read audio file",
                }
            },
        ) from e
