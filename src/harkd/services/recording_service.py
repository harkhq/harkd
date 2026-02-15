"""Recording service for managing recording lifecycle and processing."""

import asyncio
import contextlib
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from harkd.api.models.recording import (
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
from harkd.exceptions import (
    InvalidStateError,
    NoActiveRecordingError,
    RecordingInProgressError,
    RecordingNotFoundError,
)
from harkd.services.title_generator import generate_title
from harkd.state.recording_state import RecordingState, get_recording_state
from harkd.storage.filesystem.recordings import FilesystemRecordingStorage
from harkd.storage.models import StorageRecording

__all__ = ["RecordingService"]

logger = logging.getLogger(__name__)

# Subprocess timeout for transcription (10 minutes)
_SUBPROCESS_TIMEOUT = 600

# Minimum interval between audio level writes to disk (seconds)
_AUDIO_LEVEL_WRITE_INTERVAL = 1.0

# JSON delimiter used to find structured output in subprocess stdout
_JSON_DELIMITER = "---HARKD_JSON_RESULT---"


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
    ):
        """Initialize recording service.

        Args:
            storage: Recording storage backend
            config: Daemon settings (provides recording defaults)
            recording_state: Recording state manager (defaults to global singleton)
        """
        self.storage = storage
        self.config = config
        self.recording_state = recording_state or get_recording_state()
        self._processing_tasks: dict[str, asyncio.Task] = {}
        self._recorder: AudioRecorder | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._audio_level: float = 0.0
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

        logger.info(
            f"Starting recording {recording_id}: "
            f"source={final_settings['input_source']}, "
            f"model={final_settings['model']}"
        )

        # Create initial recording metadata
        storage_recording = StorageRecording(
            id=recording_id,
            status="recording",
            created_at=now,
            title=create_request.title or "Untitled Recording",
            duration=0.0,
            audio_level=0.0,
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
                input_source=final_settings["input_source"],
                sample_rate=16000,  # Whisper uses 16kHz
                level_callback=lambda level: self._update_audio_level(recording_id, level),
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
        # Use in-memory audio level instead of storage
        storage_recording.audio_level = self._audio_level

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
        storage_recording.audio_level = None  # No longer recording

        await self.storage.update(storage_recording)

        # Start background processing
        task = asyncio.create_task(self._process_recording(recording_id))

        # Add done callback to log any unhandled exceptions
        def _log_task_exception(task: asyncio.Task) -> None:
            try:
                task.result()
            except asyncio.CancelledError:
                pass  # Task was cancelled, this is expected
            except Exception as e:
                logger.error(
                    f"Unhandled exception in background processing for {recording_id}: {e}",
                    exc_info=True,
                )

        task.add_done_callback(_log_task_exception)
        self._processing_tasks[recording_id] = task

        logger.info(
            f"Recording {recording_id} stopped, duration={duration:.1f}s, "
            "background processing started"
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
            # Cancel processing task
            task = self._processing_tasks.get(recording_id)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Processing task for {recording_id} cancelled")

        # Delete recording
        await self.storage.delete(recording_id)
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
            disallowed = set(update_dict.keys()) - {"title"}
            if disallowed:
                raise InvalidStateError(
                    message=(
                        f"Can only update {', '.join(sorted(disallowed))}"
                        " on completed recordings"
                    ),
                    current_state=storage_recording.status,
                    expected_state=RecordingStatus.COMPLETE.value,
                )

        if "title" in update_dict:
            storage_recording.title = update_dict["title"]

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

            # Update speakers list
            storage_recording.speakers = list(speaker_mapping.values())

        await self.storage.update(storage_recording)

        logger.info(f"Recording {recording_id} updated successfully")
        return self._storage_to_response(storage_recording)

    async def delete_recording(self, recording_id: str) -> None:
        """Delete a recording.

        Alias for cancel_recording.

        Args:
            recording_id: ID of recording to delete

        Raises:
            RecordingNotFoundError: If recording doesn't exist
        """
        await self.cancel_recording(recording_id)

    def _update_audio_level(self, recording_id: str, level: float) -> None:
        """Update audio level for active recording (called from recorder callback).

        Keeps audio level in memory and throttles disk writes to avoid excessive I/O.

        Args:
            recording_id: Recording ID
            level: Audio level (0-1)
        """
        self._audio_level = level

        # Throttle disk writes to at most once per second
        now = time.monotonic()
        if now - self._last_audio_level_write < _AUDIO_LEVEL_WRITE_INTERVAL:
            return
        self._last_audio_level_write = now

        # Schedule async update from different thread
        if self._event_loop and not self._event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._async_update_audio_level(recording_id, level), self._event_loop
            )

    async def _async_update_audio_level(self, recording_id: str, level: float) -> None:
        """Async update of audio level (throttled).

        Args:
            recording_id: Recording ID
            level: Audio level (0-1)
        """
        try:
            storage_recording = await self.storage.get(recording_id)
            if storage_recording and storage_recording.status == RecordingStatus.RECORDING.value:
                storage_recording.audio_level = level
                await self.storage.update(storage_recording)
        except Exception as e:
            logger.debug(f"Failed to update audio level for {recording_id}: {e}")

    async def _run_transcription_subprocess(
        self,
        recording_id: str,
        audio_path: str,
        model_name: str,
        language: str | None,
        word_timestamps: bool,
        diarize: bool = False,
        hf_token: str | None = None,
    ) -> dict:
        """Run transcription in a separate subprocess.

        Args:
            recording_id: Recording ID (for logging)
            audio_path: Path to audio file
            model_name: Whisper model name
            language: Language code or None
            word_timestamps: Whether to include word timestamps
            diarize: Whether to run speaker diarization
            hf_token: HuggingFace token for diarization models

        Returns:
            Parsed transcription result dict

        Raises:
            RuntimeError: If subprocess fails or times out
        """
        cmd = [
            sys.executable,
            "-m",
            "harkd.services.transcription_worker",
            str(audio_path),
            model_name,
            str(language) if language else "None",
            str(word_timestamps).lower(),
            str(diarize).lower(),
            hf_token or "None",
        ]

        logger.info(f"[{recording_id}] Starting transcription subprocess: {' '.join(cmd)}")

        def run_subprocess():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )

        loop = asyncio.get_event_loop()
        try:
            proc = await loop.run_in_executor(None, run_subprocess)
        except subprocess.TimeoutExpired as e:
            logger.error(
                f"[{recording_id}] Transcription subprocess timed out after {_SUBPROCESS_TIMEOUT}s"
            )
            raise RuntimeError(f"Transcription timed out after {_SUBPROCESS_TIMEOUT}s") from e

        stdout = proc.stdout
        stderr = proc.stderr

        if proc.returncode != 0:
            logger.error(
                f"[{recording_id}] Transcription subprocess failed with code {proc.returncode}. "
                f"stderr: {stderr}"
            )
            raise RuntimeError(f"Transcription failed: {stderr}")

        logger.info(f"[{recording_id}] Transcription subprocess completed successfully")

        # Parse JSON result: look for delimiter first, fall back to last line
        try:
            if _JSON_DELIMITER in stdout:
                json_str = stdout.split(_JSON_DELIMITER)[-1].strip()
            else:
                # Fallback: JSON on the last non-empty line
                json_str = stdout.strip().split("\n")[-1]
            result_dict = json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(
                f"[{recording_id}] Failed to parse transcription output: {e}. stdout: {stdout}"
            )
            raise

        return result_dict

    @staticmethod
    def _convert_transcription_result(
        result_dict: dict,
    ) -> tuple[list[dict], str, float, list[str]]:
        """Convert a transcription result dict to segments, language, confidence, and speakers.

        Args:
            result_dict: Parsed JSON from transcription subprocess

        Returns:
            Tuple of (segments, detected_language, language_confidence, speakers)
        """
        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": seg.get("speaker"),
                "words": [
                    {
                        "start": w["start"],
                        "end": w["end"],
                        "word": w["word"],
                        "speaker": w.get("speaker"),
                    }
                    for w in seg.get("words", [])
                ],
            }
            for seg in result_dict.get("segments", [])
        ]

        detected_language = result_dict.get("language", "unknown")
        language_confidence = result_dict.get("language_probability", 0.0)
        speakers = result_dict.get("speakers", [])

        return segments, detected_language, language_confidence, speakers

    async def _process_recording(self, recording_id: str) -> None:
        """Background task to process recording after stop.

        Updates recording status as it progresses through stages:
        1. preprocessing -> transcription -> diarization -> complete

        Args:
            recording_id: ID of recording to process
        """
        logger.info(f"[{recording_id}] Starting background processing")

        try:
            storage_recording = await self.storage.get(recording_id)
            if storage_recording is None:
                logger.error(f"[{recording_id}] Recording not found during processing")
                return

            # Audio file is stored alongside metadata: {base_path}/recordings/{id}/audio.wav
            audio_path = self.storage.base_path / "recordings" / recording_id / "audio.wav"
            if not await asyncio.to_thread(audio_path.exists):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            settings = storage_recording.settings

            # Stage 1: Transcription
            logger.info(f"[{recording_id}] Stage 1: Transcription")
            storage_recording.processing_stage = "transcription"
            storage_recording.processing_progress = 0.0
            await self.storage.update(storage_recording)

            # Check if diarization is enabled
            diarization_enabled = settings.get("diarization", False)
            word_timestamps = settings.get("word_timestamps", False) or diarization_enabled

            # Prepare transcription parameters
            model_name = settings.get("model", "base")
            language = settings.get("language") if settings.get("language") != "auto" else None

            if diarization_enabled:
                hf_token = self.config.hf_token
                # Run diarization in subprocess
                logger.info(f"[{recording_id}] Using diarization")
                storage_recording.processing_stage = "diarization"
                storage_recording.processing_progress = 0.3
                await self.storage.update(storage_recording)

                result_dict = await self._run_transcription_subprocess(
                    recording_id,
                    str(audio_path),
                    model_name,
                    language,
                    word_timestamps=True,
                    diarize=True,
                    hf_token=hf_token,
                )
            else:
                # Regular transcription
                logger.info(f"[{recording_id}] Running transcription in separate subprocess...")
                result_dict = await self._run_transcription_subprocess(
                    recording_id,
                    str(audio_path),
                    model_name,
                    language,
                    word_timestamps,
                )

            segments, detected_language, language_confidence, speakers = (
                self._convert_transcription_result(result_dict)
            )

            # Generate full transcript
            transcript = " ".join(seg["text"] for seg in segments)

            # Re-read title from storage — user may have updated it via PATCH during processing
            latest = await self.storage.get(recording_id)
            if latest is not None:
                storage_recording.title = latest.title

            # Generate title if not provided
            if storage_recording.title == "Untitled Recording":
                storage_recording.title = generate_title(transcript)
                logger.debug(f"[{recording_id}] Generated title: {storage_recording.title}")

            # Stage: Meeting minutes (if LLM enabled)
            if self.config.llm.enabled:
                storage_recording.processing_stage = "meeting_minutes"
                storage_recording.processing_progress = 0.8
                await self.storage.update(storage_recording)

                try:
                    from harkd.llm.client import LLMClient

                    llm = LLMClient(self.config.llm)
                    minutes = await llm.generate_meeting_minutes(
                        transcript=transcript,
                        speakers=speakers,
                        language=detected_language,
                    )
                    storage_recording.executive_summary = minutes.executive_summary
                    storage_recording.meeting_notes = minutes.meeting_notes
                    storage_recording.tasks = minutes.tasks
                    storage_recording.decisions = minutes.decisions
                except Exception as e:
                    logger.warning(
                        f"[{recording_id}] Meeting minutes generation failed: {e}",
                        exc_info=True,
                    )
                    # Non-fatal: recording still completes without minutes

            # Update recording with final data
            storage_recording.status = "complete"
            storage_recording.processing_stage = None
            storage_recording.processing_progress = None
            storage_recording.input_source = settings.get("input_source")
            storage_recording.model = settings.get("model")
            storage_recording.language = detected_language
            storage_recording.language_confidence = language_confidence
            storage_recording.diarized = diarization_enabled
            storage_recording.speakers = speakers
            storage_recording.segments = segments
            storage_recording.transcript = transcript

            await self.storage.update(storage_recording)

            logger.info(
                f"[{recording_id}] Processing complete: "
                f"{len(segments)} segments, {len(speakers)} speakers, "
                f"language={detected_language}"
            )

        except asyncio.CancelledError:
            logger.info(f"[{recording_id}] Processing cancelled")
            raise
        except Exception as e:
            logger.error(
                f"[{recording_id}] Processing failed: {e}",
                exc_info=True,
            )
            # Update status to error
            try:
                storage_recording = await self.storage.get(recording_id)
                if storage_recording:
                    storage_recording.status = "error"
                    storage_recording.processing_stage = None
                    storage_recording.processing_progress = None
                    await self.storage.update(storage_recording)
            except Exception as update_error:
                logger.error(
                    f"[{recording_id}] Failed to update error status: {update_error}",
                    exc_info=True,
                )
        finally:
            # Clean up task reference
            self._processing_tasks.pop(recording_id, None)

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
            audio_level=storage.audio_level,
            processing_stage=(
                ProcessingStage(storage.processing_stage) if storage.processing_stage else None
            ),
            processing_progress=storage.processing_progress,
            input_source=storage.input_source,
            model=storage.model,
            language=storage.language,
            language_confidence=storage.language_confidence,
            diarized=storage.diarized,
            speakers=storage.speakers,
            segments=segments,
            transcript=storage.transcript,
            tags=list(storage.tags),
            executive_summary=list(storage.executive_summary),
            meeting_notes=cast(Any, storage.meeting_notes),
            tasks=cast(Any, storage.tasks),
            decisions=list(storage.decisions),
            settings=storage.settings,
        )
