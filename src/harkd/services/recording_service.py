"""Recording service for managing recording lifecycle and processing."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from harkd.api.models.recording import (
    ActiveRecordingUpdate,
    ProcessingStage,
    RecordingCreate,
    RecordingListResponse,
    RecordingResponse,
    RecordingStatus,
    RecordingUpdate,
    SegmentModel,
    WordModel,
)
from harkd.audio.recorder import AudioRecorder
from harkd.config import HarkdSettings
from harkd.events import Event, get_event_bus
from harkd.exceptions import (
    InvalidStateError,
    NoActiveRecordingError,
    NoLoopbackDeviceError,
    NoMicrophoneError,
    RecordingInProgressError,
    RecordingNotFoundError,
)
from harkd.state.recording_state import RecordingState, get_recording_state
from harkd.storage.filesystem.recordings import FilesystemRecordingStorage
from harkd.storage.models import StorageRecording

if TYPE_CHECKING:
    from harkd.services.processing_worker import ProcessingWorker
    from harkd.services.voice_profile_service import VoiceProfileService

__all__ = ["RecordingService"]

logger = logging.getLogger(__name__)

# Minimum interval between audio level writes to disk (seconds)
_AUDIO_LEVEL_WRITE_INTERVAL = 1.0


class RecordingService:
    """Service for managing recordings.

    Orchestrates the complete recording workflow:
    1. Start recording (audio capture)
    2. Stop recording and begin processing
    3. Background processing (transcribe/diarize)
    4. Store final result
    """

    def __init__(
        self,
        storage: FilesystemRecordingStorage,
        config: HarkdSettings,
        recording_state: RecordingState | None = None,
        worker: ProcessingWorker | None = None,
        voice_profile_service: VoiceProfileService | None = None,
    ):
        """Initialize recording service.

        Args:
            storage: Recording storage backend
            config: Daemon settings (provides recording defaults)
            recording_state: Recording state manager (defaults to global singleton)
            worker: Processing worker for background transcription
            voice_profile_service: Voice profile service for creating profiles from speakers
        """
        self.storage = storage
        self.config = config
        self.recording_state = recording_state or get_recording_state()
        self.worker = worker
        self.voice_profile_service = voice_profile_service
        self._recorder: AudioRecorder | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._mic_level: float = 0.0
        self._speaker_level: float = 0.0
        self._last_audio_level_write: float = 0.0

    async def start_recording(self, create_request: RecordingCreate) -> RecordingResponse:
        """Start a new recording.

        Args:
            create_request: Recording creation request

        Returns:
            Initial recording response with status "recording"

        Raises:
            RecordingInProgressError: If a recording is already active
        """
        # Check if already recording
        if self.recording_state.is_recording:
            active_id = self.recording_state.active_recording_id
            if active_id is None:
                raise RuntimeError("is_recording is True but active_recording_id is None")
            logger.warning(f"Cannot start recording: already recording {active_id}")
            raise RecordingInProgressError(active_id)

        # Generate recording ID (date-based + short UUID)
        now = datetime.now(UTC)
        short_uuid = str(uuid4())[:8]
        recording_id = f"{now.strftime('%Y-%m-%d')}-{short_uuid}"

        # Merge daemon defaults with per-recording overrides
        final_settings = self.config.recording.model_dump()
        if create_request.settings is not None:
            overrides = create_request.settings.model_dump(exclude_none=True)
            final_settings.update(overrides)

        # Determine mic/speaker enabled (default both True)
        mic_enabled = final_settings.pop("mic_enabled", True)
        speaker_enabled = final_settings.pop("speaker_enabled", True)
        mic_gain = final_settings.get("mic_gain", 1.0)

        logger.info(
            f"Starting recording {recording_id}: "
            f"mic={mic_enabled}, speaker={speaker_enabled}, "
            f"model={final_settings['model']}"
        )

        # Store mic/speaker state in settings for the response
        final_settings["mic_enabled"] = mic_enabled
        final_settings["speaker_enabled"] = speaker_enabled

        # Create initial recording metadata
        storage_recording = StorageRecording(
            id=recording_id,
            status="recording",
            created_at=now,
            title=create_request.title or "Untitled Recording",
            duration=0.0,
            mic_enabled=mic_enabled,
            speaker_enabled=speaker_enabled,
            mic_level=0.0,
            speaker_level=0.0,
            settings=cast(Any, final_settings),
        )

        # Save initial metadata (this creates the directory structure)
        await self.storage.create(storage_recording)

        # Now get the audio path from the created directory
        audio_path = self.storage.base_path / "recordings" / recording_id / "audio.wav"

        # Capture the event loop for cross-thread async calls
        self._event_loop = asyncio.get_running_loop()

        # Create audio recorder
        recorder = None
        try:
            recorder = AudioRecorder(
                output_path=audio_path,
                mic_enabled=mic_enabled,
                speaker_enabled=speaker_enabled,
                sample_rate=16000,  # Whisper uses 16kHz
                level_callback=lambda mic_lvl, spk_lvl: self._update_audio_levels(
                    recording_id, mic_lvl, spk_lvl
                ),
                mic_gain=mic_gain,
            )

            # Start recording
            recorder.start()

            # Store recorder reference so stop_recording() can stop it
            self._recorder = recorder

            # Update state - if this throws, recorder.stop() is called in finally
            await self.recording_state.start(recording_id)

        except Exception as e:
            # Stop recorder if it was started
            if recorder is not None:
                with contextlib.suppress(Exception):
                    recorder.stop()
            self._recorder = None
            logger.error(f"Failed to start recording {recording_id}: {e}", exc_info=True)
            # Clean up metadata
            await self.storage.delete(recording_id)
            raise

        logger.info(f"Recording {recording_id} started successfully")

        bus = get_event_bus()
        bus.emit(
            Event(
                "recording_started",
                {"recording_id": recording_id, "title": storage_recording.title},
            )
        )
        bus.emit(Event("invalidate", {"entity": "recordings"}))

        # Pre-warm transcription backend (e.g. start provisioning GPU infra)
        if self.worker is not None:
            asyncio.create_task(self.worker.pre_warm_backend())

        return self._storage_to_response(storage_recording)

    async def get_active_recording(self) -> RecordingResponse:
        """Get the currently active recording.

        Returns:
            Recording response with real-time duration for active recording

        Raises:
            NoActiveRecordingError: If no recording is currently active
        """
        if not self.recording_state.is_recording:
            raise NoActiveRecordingError()

        active_id = self.recording_state.active_recording_id
        if not active_id:
            raise NoActiveRecordingError()

        # Get recording from storage
        storage_recording = await self.storage.get(active_id)
        if storage_recording is None:
            logger.error(f"Active recording {active_id} not found in storage")
            raise RecordingNotFoundError(active_id)

        # Update duration with real-time value
        storage_recording.duration = self.recording_state.duration
        # Use in-memory levels instead of storage
        storage_recording.mic_level = self._mic_level
        storage_recording.speaker_level = self._speaker_level

        # Update enabled state from recorder
        if self._recorder:
            storage_recording.mic_enabled = not self._recorder._mic_muted
            storage_recording.speaker_enabled = not self._recorder._speaker_muted

        return self._storage_to_response(storage_recording)

    async def stop_active_recording(self) -> RecordingResponse:
        """Stop the currently active recording and begin processing.

        Returns:
            Recording response with status "processing"

        Raises:
            NoActiveRecordingError: If no recording is currently active
        """
        if not self.recording_state.is_recording:
            raise NoActiveRecordingError()

        active_id = self.recording_state.active_recording_id
        if not active_id:
            raise NoActiveRecordingError()

        # Delegate to existing stop_recording method
        return await self.stop_recording(active_id)

    async def stop_recording(self, recording_id: str) -> RecordingResponse:
        """Stop active recording and begin processing.

        Args:
            recording_id: ID of recording to stop

        Returns:
            Recording response with status "processing"

        Raises:
            RecordingNotFoundError: If recording doesn't exist
            InvalidStateError: If recording is not in "recording" state
        """
        logger.info(f"Stopping recording {recording_id}")

        # Verify recording exists and is in recording state
        storage_recording = await self.storage.get(recording_id)
        if storage_recording is None:
            raise RecordingNotFoundError(recording_id)

        if storage_recording.status != RecordingStatus.RECORDING.value:
            raise InvalidStateError(
                message=f"Cannot stop recording in state: {storage_recording.status}",
                current_state=storage_recording.status,
                expected_state=RecordingStatus.RECORDING.value,
            )

        # Stop the audio recorder
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None

        # Update recording state
        try:
            active_id, start_time = await self.recording_state.stop()
            if active_id != recording_id:
                logger.warning(f"State mismatch: expected {recording_id}, got {active_id}")
        except NoActiveRecordingError as e:
            logger.error(f"Failed to stop recording state: {e}")
            raise InvalidStateError(
                message="No active recording in state",
                current_state="none",
                expected_state="recording",
            ) from e

        # Calculate final duration
        duration = (datetime.now(UTC) - start_time).total_seconds()

        # Update recording status
        storage_recording.status = "processing"
        storage_recording.duration = duration
        storage_recording.processing_stage = "preprocessing"
        storage_recording.processing_progress = 0.0
        storage_recording.processing_started_at = datetime.now(UTC)
        storage_recording.mic_level = None  # No longer recording
        storage_recording.speaker_level = None

        await self.storage.update(storage_recording)

        bus = get_event_bus()
        bus.emit(Event("recording_stopped", {"recording_id": recording_id}))
        bus.emit(Event("invalidate", {"entity": "recordings"}))

        # Enqueue for background processing
        if self.worker is not None:
            self.worker.enqueue(recording_id)
        else:
            logger.warning(
                f"[{recording_id}] No processing worker available, "
                "recording will stay in processing state"
            )

        logger.info(
            f"Recording {recording_id} stopped, duration={duration:.1f}s, processing enqueued"
        )

        return self._storage_to_response(storage_recording)

    async def cancel_recording(self, recording_id: str) -> None:
        """Cancel/delete a recording.

        - If status = "recording": stop and discard
        - If status = "processing": cancel processing
        - If status = "complete": delete from storage

        Args:
            recording_id: ID of recording to cancel

        Raises:
            RecordingNotFoundError: If recording doesn't exist
        """
        logger.info(f"Canceling recording {recording_id}")

        storage_recording = await self.storage.get(recording_id)
        if storage_recording is None:
            raise RecordingNotFoundError(recording_id)

        status = storage_recording.status

        if status == RecordingStatus.RECORDING.value:
            # Stop the audio recorder
            if self._recorder is not None:
                self._recorder.stop()
                self._recorder = None
            # Clear recording state
            try:
                await self.recording_state.cancel()
            except NoActiveRecordingError:
                logger.warning("Recording state already cleared")

        elif status == RecordingStatus.PROCESSING.value:
            # Recording is queued/processing in the worker — just let deletion proceed.
            # The worker will notice the recording is gone when it tries to read storage.
            logger.debug(f"Recording {recording_id} is processing, will be deleted")

        # Delete recording
        await self.storage.delete(recording_id)
        get_event_bus().emit(Event("invalidate", {"entity": "recordings"}))
        logger.info(f"Recording {recording_id} cancelled and deleted")

    async def get_recording(self, recording_id: str) -> RecordingResponse:
        """Get recording by ID.

        Args:
            recording_id: ID of recording to retrieve

        Returns:
            Recording response

        Raises:
            RecordingNotFoundError: If recording doesn't exist
        """
        storage_recording = await self.storage.get(recording_id)
        if storage_recording is None:
            raise RecordingNotFoundError(recording_id)

        # Update duration for active recordings
        if storage_recording.status == RecordingStatus.RECORDING.value:
            storage_recording.duration = self.recording_state.duration

        return self._storage_to_response(storage_recording)

    async def list_recordings(
        self,
        limit: int = 50,
        offset: int = 0,
        status: RecordingStatus | None = None,
        search: str | None = None,
        sort: str = "desc",
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> RecordingListResponse:
        """List recordings with filtering.

        Args:
            limit: Maximum number of results (1-100)
            offset: Result offset for pagination
            status: Filter by status
            search: Search in title/transcript

        Returns:
            Paginated list of recordings
        """
        logger.debug(
            f"Listing recordings: limit={limit}, offset={offset}, status={status}, search={search}"
        )

        # Load all recordings with status filter at storage level
        status_value = status.value if status else None
        all_recordings = await self.storage.list(limit=0, offset=0, status=status_value)

        # Apply search filter
        if search:
            search_lower = search.lower()
            all_recordings = [
                r
                for r in all_recordings
                if search_lower in r.title.lower()
                or (r.transcript and search_lower in r.transcript.lower())
            ]

        # Apply date range filters
        if created_after:
            all_recordings = [r for r in all_recordings if r.created_at >= created_after]
        if created_before:
            all_recordings = [r for r in all_recordings if r.created_at <= created_before]

        # Sort by created_at
        all_recordings.sort(key=lambda r: r.created_at, reverse=(sort == "desc"))

        total = len(all_recordings)

        # Apply pagination after filtering
        paginated = all_recordings[offset : offset + limit]

        # Convert to list items
        from harkd.api.models.recording import RecordingListItem

        items = [
            RecordingListItem(
                id=r.id,
                title=r.title,
                created_at=r.created_at,
                duration=r.duration,
                status=RecordingStatus(r.status),
                speakers=list(r.speakers),
                tags=list(r.tags),
                language=r.language,
            )
            for r in paginated
        ]

        return RecordingListResponse(
            total=total,
            limit=limit,
            offset=offset,
            recordings=items,
        )

    async def update_recording(
        self, recording_id: str, update: RecordingUpdate
    ) -> RecordingResponse:
        """Update recording metadata.

        Args:
            recording_id: ID of recording to update
            update: Update request

        Returns:
            Updated recording response

        Raises:
            RecordingNotFoundError: If recording doesn't exist
            InvalidStateError: If recording is not complete
        """
        logger.info(f"Updating recording {recording_id}: {update}")

        storage_recording = await self.storage.get(recording_id)
        if storage_recording is None:
            raise RecordingNotFoundError(recording_id)

        # Apply updates
        update_dict = update.model_dump(exclude_unset=True)

        # Non-complete recordings: only title updates allowed
        if storage_recording.status != RecordingStatus.COMPLETE.value:
            disallowed = set(update_dict.keys()) - {
                "title",
                "tags",
                "tasks",
                "decisions",
                "create_voice_profiles",
                "speaker_profile_ids",
            }
            if disallowed:
                raise InvalidStateError(
                    message=(
                        f"Can only update {', '.join(sorted(disallowed))} on completed recordings"
                    ),
                    current_state=storage_recording.status,
                    expected_state=RecordingStatus.COMPLETE.value,
                )

        if "title" in update_dict:
            storage_recording.title = update_dict["title"]

        if "tags" in update_dict:
            storage_recording.tags = update_dict["tags"]

        if "tasks" in update_dict:
            storage_recording.tasks = update_dict["tasks"]

        if "decisions" in update_dict:
            storage_recording.decisions = update_dict["decisions"]

        if "speakers" in update_dict:
            # Update speaker names in segments
            speaker_mapping = update_dict["speakers"]
            for segment in storage_recording.segments:
                if segment.get("speaker") in speaker_mapping:
                    segment["speaker"] = speaker_mapping[segment["speaker"]]
                # Update words too
                for word in segment.get("words", []):
                    if word.get("speaker") in speaker_mapping:
                        word["speaker"] = speaker_mapping[word["speaker"]]

            # Update speakers list (deduplicate for merged speakers)
            storage_recording.speakers = list(dict.fromkeys(speaker_mapping.values()))

            # Link speakers to voice profiles
            speaker_profiles: dict[str, str] = {}

            # 1. Process explicit profile links first
            explicit_profile_ids = update_dict.get("speaker_profile_ids") or {}
            if explicit_profile_ids and self.voice_profile_service:
                for label, profile_id in explicit_profile_ids.items():
                    # Validate profile exists (raises VoiceProfileNotFoundError)
                    await self.voice_profile_service.get_profile(profile_id)
                    embedding = (storage_recording.speaker_embeddings or {}).get(label)
                    if embedding is None:
                        continue
                    await self.voice_profile_service.add_embedding(
                        profile_id=profile_id,
                        recording_id=recording_id,
                        speaker_label=label,
                        vector=embedding,
                        audio_duration=storage_recording.duration,
                    )
                    speaker_profiles[label] = profile_id

            # 2. Create voice profiles for remaining (unlinked) speakers
            if (
                update_dict.get("create_voice_profiles")
                and self.voice_profile_service
                and storage_recording.speaker_embeddings
            ):
                for original_label, name in speaker_mapping.items():
                    if original_label in speaker_profiles:
                        continue
                    embedding = storage_recording.speaker_embeddings.get(original_label)
                    if embedding is None:
                        continue
                    profile = await self.voice_profile_service.find_or_create_by_name(name)
                    await self.voice_profile_service.add_embedding(
                        profile_id=profile.id,
                        recording_id=recording_id,
                        speaker_label=original_label,
                        vector=embedding,
                        audio_duration=storage_recording.duration,
                    )
                    speaker_profiles[original_label] = profile.id

            if speaker_profiles:
                storage_recording.speaker_profiles = speaker_profiles

        await self.storage.update(storage_recording)
        get_event_bus().emit(Event("invalidate", {"entity": "recordings"}))

        logger.info(f"Recording {recording_id} updated successfully")
        return self._storage_to_response(storage_recording)

    async def toggle_inputs(self, mic_enabled: bool | None, speaker_enabled: bool | None) -> None:
        """Toggle mic/speaker inputs on the active recording.

        Args:
            mic_enabled: Set mic state (None = no change)
            speaker_enabled: Set speaker state (None = no change)

        Raises:
            NoActiveRecordingError: If no recording is active
            NoMicrophoneError: If trying to enable an unavailable mic
            NoLoopbackDeviceError: If trying to enable an unavailable speaker
        """
        if self._recorder is None:
            raise NoActiveRecordingError()
        if mic_enabled is not None:
            if mic_enabled and not self._recorder._mic_available:
                raise NoMicrophoneError()
            self._recorder.set_mic_enabled(mic_enabled)
        if speaker_enabled is not None:
            if speaker_enabled and not self._recorder._speaker_available:
                raise NoLoopbackDeviceError()
            self._recorder.set_speaker_enabled(speaker_enabled)

    async def update_active_recording(self, update: ActiveRecordingUpdate) -> RecordingResponse:
        """Update the active recording (title, input toggles).

        Args:
            update: Update request with optional title, mic_enabled, speaker_enabled

        Returns:
            Updated recording response

        Raises:
            NoActiveRecordingError: If no recording is active
        """
        if not self.recording_state.is_recording:
            raise NoActiveRecordingError()

        active_id = self.recording_state.active_recording_id
        if not active_id:
            raise NoActiveRecordingError()

        # Toggle inputs if requested
        if update.mic_enabled is not None or update.speaker_enabled is not None:
            await self.toggle_inputs(update.mic_enabled, update.speaker_enabled)

        # Update title in storage if requested
        if update.title is not None:
            storage_recording = await self.storage.get(active_id)
            if storage_recording is None:
                raise RecordingNotFoundError(active_id)
            storage_recording.title = update.title
            await self.storage.update(storage_recording)

        # Update toggle state in storage
        if update.mic_enabled is not None or update.speaker_enabled is not None:
            storage_recording = await self.storage.get(active_id)
            if storage_recording is not None and self._recorder is not None:
                storage_recording.mic_enabled = not self._recorder._mic_muted
                storage_recording.speaker_enabled = not self._recorder._speaker_muted
                await self.storage.update(storage_recording)

        return await self.get_active_recording()

    async def retry_recording(
        self, recording_id: str, overrides: dict | None = None
    ) -> RecordingResponse:
        """Retry a failed recording.

        Args:
            recording_id: ID of recording to retry
            overrides: Optional diarization overrides to merge into settings

        Returns:
            Updated recording response with status "processing"

        Raises:
            RecordingNotFoundError: If recording doesn't exist
            RetryNotAllowedError: If recording is not in error state
            RuntimeError: If no processing worker is available
        """
        if self.worker is None:
            raise RuntimeError("No processing worker available")
        if overrides:
            storage_recording = await self.storage.get(recording_id)
            if storage_recording is None:
                raise RecordingNotFoundError(recording_id)
            storage_recording.settings.update(overrides)
            await self.storage.update(storage_recording)
        await self.worker.retry(recording_id)
        return await self.get_recording(recording_id)

    async def delete_recording(self, recording_id: str) -> None:
        """Delete a recording.

        Alias for cancel_recording.

        Args:
            recording_id: ID of recording to delete

        Raises:
            RecordingNotFoundError: If recording doesn't exist
        """
        await self.cancel_recording(recording_id)

    def _update_audio_levels(
        self, recording_id: str, mic_level: float, speaker_level: float
    ) -> None:
        """Update audio levels for active recording (called from recorder callback).

        Keeps audio levels in memory and throttles disk writes to avoid excessive I/O.

        Args:
            recording_id: Recording ID
            mic_level: Mic audio level (0-1)
            speaker_level: Speaker audio level (0-1)
        """
        self._mic_level = mic_level
        self._speaker_level = speaker_level

        get_event_bus().emit(
            Event(
                "audio_levels",
                {
                    "recording_id": recording_id,
                    "mic_level": mic_level,
                    "speaker_level": speaker_level,
                },
            )
        )

        # Throttle disk writes to at most once per second
        now = time.monotonic()
        if now - self._last_audio_level_write < _AUDIO_LEVEL_WRITE_INTERVAL:
            return
        self._last_audio_level_write = now

        # Schedule async update from different thread
        if self._event_loop and not self._event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._async_update_audio_levels(recording_id, mic_level, speaker_level),
                self._event_loop,
            )

    async def _async_update_audio_levels(
        self, recording_id: str, mic_level: float, speaker_level: float
    ) -> None:
        """Async update of audio levels (throttled).

        Args:
            recording_id: Recording ID
            mic_level: Mic audio level (0-1)
            speaker_level: Speaker audio level (0-1)
        """
        try:
            storage_recording = await self.storage.get(recording_id)
            if storage_recording and storage_recording.status == RecordingStatus.RECORDING.value:
                storage_recording.mic_level = mic_level
                storage_recording.speaker_level = speaker_level
                await self.storage.update(storage_recording)
        except Exception as e:
            logger.debug(f"Failed to update audio levels for {recording_id}: {e}")

    def _storage_to_response(self, storage: StorageRecording) -> RecordingResponse:
        """Convert storage model to API response.

        Args:
            storage: Storage recording model

        Returns:
            API recording response
        """
        # Convert segments
        segments = None
        if storage.segments:
            segments = [
                SegmentModel(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    speaker=seg.get("speaker"),
                    words=[
                        WordModel(
                            start=w["start"],
                            end=w["end"],
                            word=w["word"],
                            speaker=w.get("speaker"),
                        )
                        for w in seg.get("words", [])
                    ],
                )
                for seg in storage.segments
            ]

        return RecordingResponse(
            id=storage.id,
            status=RecordingStatus(storage.status),
            created_at=storage.created_at,
            title=storage.title,
            duration=storage.duration,
            mic_enabled=storage.mic_enabled,
            speaker_enabled=storage.speaker_enabled,
            mic_level=storage.mic_level,
            speaker_level=storage.speaker_level,
            processing_stage=(
                ProcessingStage(storage.processing_stage) if storage.processing_stage else None
            ),
            processing_progress=storage.processing_progress,
            processing_started_at=storage.processing_started_at,
            retry_count=storage.retry_count,
            last_error=storage.last_error,
            model=storage.model,
            language=storage.language,
            language_confidence=storage.language_confidence,
            diarized=storage.diarized,
            speakers=storage.speakers,
            speaker_embeddings=storage.speaker_embeddings,
            speaker_profiles=storage.speaker_profiles,
            segments=segments,
            transcript=storage.transcript,
            tags=list(storage.tags),
            executive_summary=list(storage.executive_summary),
            meeting_notes=cast(Any, storage.meeting_notes),
            tasks=cast(Any, storage.tasks),
            decisions=list(storage.decisions),
            settings=storage.settings,
        )
