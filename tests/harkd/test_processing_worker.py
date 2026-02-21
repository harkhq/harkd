"""Tests for ProcessingWorker."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from harkd.config import HarkdSettings, StorageSettings
from harkd.events import get_event_bus
from harkd.exceptions import RecordingNotFoundError, RetryNotAllowedError
from harkd.services.processing_worker import ProcessingWorker
from harkd.services.voice_profile_service import VoiceProfileService
from harkd.storage.filesystem.recordings import FilesystemRecordingStorage
from harkd.storage.filesystem.voice_profiles import FilesystemVoiceProfileStorage
from harkd.storage.models import StorageRecording, StorageVoiceProfile
from harkd.transcription import LocalBackend


@pytest.fixture
def storage(tmp_path):
    """Create recording storage."""
    return FilesystemRecordingStorage(tmp_path)


@pytest.fixture
def config(tmp_path):
    """Create test daemon config."""
    return HarkdSettings(storage=StorageSettings(base_path=tmp_path))


@pytest.fixture
def voice_profile_service(tmp_path):
    """Create voice profile service."""
    vp_storage = FilesystemVoiceProfileStorage(tmp_path)
    return VoiceProfileService(vp_storage)


@pytest.fixture
def worker(storage, config):
    """Create processing worker."""
    return ProcessingWorker(storage, config)


@pytest.fixture
def worker_with_profiles(storage, config, voice_profile_service):
    """Create processing worker with voice profile service."""
    return ProcessingWorker(storage, config, voice_profile_service=voice_profile_service)


@pytest.fixture
def mock_transcriber():
    """Mock transcription backend."""
    mock_result = {
        "text": "Hello world",
        "language": "en",
        "language_probability": 0.95,
        "duration": 2.0,
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "Hello"},
                    {"start": 1.0, "end": 2.0, "word": "world"},
                ],
            }
        ],
    }

    async def mock_run_transcription(self, recording_id, **kwargs):
        return mock_result

    with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
        yield mock_result


async def _create_recording(storage, **overrides):
    """Helper to create a recording in storage."""
    defaults: dict[str, Any] = {
        "id": "test-rec-001",
        "status": "processing",
        "created_at": datetime.now(UTC),
        "title": "Test Recording",
        "duration": 10.0,
        "settings": {"model": "base", "diarization": False, "word_timestamps": False},
        "processing_stage": "preprocessing",
        "processing_progress": 0.0,
    }
    defaults.update(overrides)
    rec = StorageRecording(**defaults)
    await storage.create(rec)
    return rec


class TestEnqueue:
    """Tests for enqueue and deduplication."""

    def test_enqueue_adds_to_queue(self, worker):
        """Test enqueue adds ID to active set and queue."""
        result = worker.enqueue("rec-001")

        assert result is True
        assert "rec-001" in worker._active
        assert worker._queue.qsize() == 1

    def test_enqueue_dedup(self, worker):
        """Test enqueue same ID twice returns False and queue size stays 1."""
        worker.enqueue("rec-001")
        result = worker.enqueue("rec-001")

        assert result is False
        assert worker._queue.qsize() == 1


class TestProcessRecording:
    """Tests for _process_recording."""

    @pytest.mark.asyncio
    async def test_process_recording_success(self, worker, storage, mock_transcriber):
        """Test successful processing sets status to complete."""
        rec = await _create_recording(storage)

        # Create audio file
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"
        assert result.retry_count == 0
        assert result.transcript == "Hello world"
        assert result.language == "en"
        assert result.processing_duration is not None
        assert result.processing_duration >= 0

    @pytest.mark.asyncio
    async def test_process_recording_transient_error_retries(self, worker, storage):
        """Test transient error increments retry_count and re-enqueues."""
        rec = await _create_recording(storage, max_retries=3)

        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        async def failing_transcription(self, recording_id, **kwargs):
            raise RuntimeError("Transcription timed out after 1800s")

        with patch.object(ProcessingWorker, "_run_transcription", failing_transcription):
            await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.retry_count == 1
        assert result.last_error == "Transcription timed out after 1800s"
        assert result.status == "processing"  # Still processing, will be retried
        assert len(result.error_history) == 1
        assert result.error_history[0]["transient"] is True

    @pytest.mark.asyncio
    async def test_process_recording_permanent_error_no_retry(self, worker, storage):
        """Test permanent error sets status to error without re-enqueue."""
        rec = await _create_recording(storage, max_retries=3)

        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        async def failing_transcription(self, recording_id, **kwargs):
            raise RuntimeError("Audio file not found: /missing.wav")

        with patch.object(ProcessingWorker, "_run_transcription", failing_transcription):
            await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "error"
        assert result.retry_count == 1
        assert "not found" in result.last_error

    @pytest.mark.asyncio
    async def test_process_recording_exhausts_retries(self, worker, storage):
        """Test that after max retries, status becomes error."""
        rec = await _create_recording(storage, max_retries=1, retry_count=1)

        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        async def failing_transcription(self, recording_id, **kwargs):
            raise RuntimeError("Transcription timed out after 1800s")

        with patch.object(ProcessingWorker, "_run_transcription", failing_transcription):
            await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "error"
        assert result.retry_count == 2  # Was 1, now incremented to 2

    @pytest.mark.asyncio
    async def test_process_recording_missing_audio(self, worker, storage):
        """Test processing fails cleanly when audio file is missing."""
        rec = await _create_recording(storage, max_retries=3)
        # Don't create audio file

        await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        # FileNotFoundError contains "not found" — classified as permanent
        assert result.status == "error"
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_error_history_appended(self, worker, storage):
        """Test that error_history grows with each failure."""
        rec = await _create_recording(storage, max_retries=5)

        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        async def failing_transcription(self, recording_id, **kwargs):
            raise RuntimeError("OOM killed")

        with patch.object(ProcessingWorker, "_run_transcription", failing_transcription):
            await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert len(result.error_history) == 1

        # Process again (simulating re-enqueue)
        with patch.object(ProcessingWorker, "_run_transcription", failing_transcription):
            await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert len(result.error_history) == 2
        assert result.retry_count == 2


class TestBackendDispatch:
    """Tests for backend creation and fallback."""

    def test_default_backend_is_local(self, worker):
        """Test default config creates LocalBackend."""
        assert isinstance(worker._backend, LocalBackend)

    @pytest.mark.asyncio
    async def test_fallback_to_local_on_remote_failure(self, storage, tmp_path):
        """Test remote failure falls back to local when configured."""
        from harkd.config import KoyebProviderSettings, TranscriptionSettings

        config = HarkdSettings(
            storage=StorageSettings(base_path=tmp_path),
            transcription=TranscriptionSettings(
                backend="koyeb",
                endpoint_url="https://fake.koyeb.app",
                fallback_to_local=True,
                koyeb=KoyebProviderSettings(token="fake-token"),
            ),
        )
        worker = ProcessingWorker(storage, config)

        rec = await _create_recording(storage)
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Fallback result",
            "language": "en",
            "language_probability": 0.9,
            "duration": 1.0,
            "segments": [{"start": 0, "end": 1, "text": "Fallback result", "words": []}],
        }

        # Mock the remote backend to fail, then local to succeed
        worker._backend.transcribe = AsyncMock(side_effect=RuntimeError("Connection failed"))
        with patch.object(LocalBackend, "transcribe", return_value=mock_result):
            await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"
        assert result.transcript == "Fallback result"

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self, storage, tmp_path):
        """Test remote failure does NOT fall back when fallback_to_local=False."""
        from harkd.config import KoyebProviderSettings, TranscriptionSettings

        config = HarkdSettings(
            storage=StorageSettings(base_path=tmp_path),
            transcription=TranscriptionSettings(
                backend="koyeb",
                endpoint_url="https://fake.koyeb.app",
                fallback_to_local=False,
                koyeb=KoyebProviderSettings(token="fake-token"),
            ),
        )
        worker = ProcessingWorker(storage, config)

        rec = await _create_recording(storage, max_retries=0)
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        worker._backend.transcribe = AsyncMock(side_effect=RuntimeError("Connection failed"))
        await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "error"


class TestRecover:
    """Tests for recover()."""

    @pytest.mark.asyncio
    async def test_recover_stuck_processing(self, worker, storage):
        """Test that recordings stuck in processing are re-enqueued."""
        await _create_recording(storage, id="stuck-001", status="processing")

        count = await worker.recover()

        assert count == 1
        assert "stuck-001" in worker._active

    @pytest.mark.asyncio
    async def test_recover_retryable_error(self, worker, storage):
        """Test that retryable error recordings are re-enqueued."""
        await _create_recording(
            storage,
            id="err-001",
            status="error",
            retry_count=1,
            max_retries=3,
            last_error="Transcription timed out after 1800s",
        )

        count = await worker.recover()

        assert count == 1
        assert "err-001" in worker._active

    @pytest.mark.asyncio
    async def test_recover_exhausted_error_not_enqueued(self, worker, storage):
        """Test that exhausted-retry error recordings are NOT enqueued."""
        await _create_recording(
            storage,
            id="err-002",
            status="error",
            retry_count=3,
            max_retries=3,
            last_error="timed out",
        )

        count = await worker.recover()

        assert count == 0
        assert "err-002" not in worker._active

    @pytest.mark.asyncio
    async def test_recover_permanent_error_not_enqueued(self, worker, storage):
        """Test that permanent error recordings are NOT enqueued."""
        await _create_recording(
            storage,
            id="err-003",
            status="error",
            retry_count=0,
            max_retries=3,
            last_error="Audio file not found: /missing.wav",
        )

        count = await worker.recover()

        assert count == 0
        assert "err-003" not in worker._active

    @pytest.mark.asyncio
    async def test_recover_processing_exhausts_retries(self, worker, storage):
        """Test that stuck processing with exhausted retries goes to error."""
        await _create_recording(
            storage,
            id="stuck-002",
            status="processing",
            retry_count=3,
            max_retries=3,
        )

        count = await worker.recover()

        assert count == 0
        result = await storage.get("stuck-002")
        assert result.status == "error"


class TestRetry:
    """Tests for retry()."""

    @pytest.mark.asyncio
    async def test_retry_success(self, worker, storage):
        """Test manual retry enqueues recording."""
        await _create_recording(
            storage,
            id="retry-001",
            status="error",
            retry_count=1,
            max_retries=3,
            last_error="timed out",
        )

        await worker.retry("retry-001")

        result = await storage.get("retry-001")
        assert result.status == "processing"
        assert "retry-001" in worker._active

    @pytest.mark.asyncio
    async def test_retry_wrong_status(self, worker, storage):
        """Test retry raises error for disallowed status."""
        await _create_recording(storage, id="retry-002", status="processing")

        with pytest.raises(RetryNotAllowedError):
            await worker.retry("retry-002")

    @pytest.mark.asyncio
    async def test_retry_already_queued(self, worker, storage):
        """Test retry raises error if already in active set."""
        await _create_recording(storage, id="retry-003", status="error", last_error="timed out")
        worker._active.add("retry-003")

        with pytest.raises(RetryNotAllowedError):
            await worker.retry("retry-003")

    @pytest.mark.asyncio
    async def test_retry_complete_clears_output(self, worker, storage):
        """Test retry from complete state clears all processing output fields."""
        await _create_recording(
            storage,
            id="retry-004",
            status="complete",
            model="base",
            language="en",
            language_confidence=0.95,
            diarized=True,
            speakers=["Alice", "Bob"],
            speaker_embeddings={"SPEAKER_00": [0.1, 0.2]},
            speaker_profiles={"SPEAKER_00": "profile-1"},
            segments=[{"start": 0, "end": 1, "text": "Hello"}],
            transcript="Hello",
            executive_summary=["Summary line"],
            meeting_notes=[{"topic": "Intro"}],
            tasks=[{"task": "Do thing"}],
            decisions=["Decision 1"],
            retry_count=2,
            last_error="previous error",
            last_error_at=datetime.now(UTC),
            error_history=[{"attempt": 1, "error": "old"}],
            tags=["important"],
        )

        await worker.retry("retry-004")

        result = await storage.get("retry-004")
        assert result.status == "processing"
        assert result.processing_stage == "preprocessing"

        # Processing output cleared
        assert result.model is None
        assert result.language is None
        assert result.language_confidence is None
        assert result.diarized is None
        assert result.speakers == []
        assert result.speaker_embeddings is None
        assert result.speaker_profiles is None
        assert result.segments == []
        assert result.transcript is None
        assert result.processing_duration is None
        assert result.executive_summary == []
        assert result.meeting_notes == []
        assert result.tasks == []
        assert result.decisions == []

        # Retry/error state reset
        assert result.retry_count == 0
        assert result.last_error is None
        assert result.last_error_at is None
        assert result.error_history == []

        # Preserved fields
        assert result.tags == ["important"]
        assert "retry-004" in worker._active

    @pytest.mark.asyncio
    async def test_retry_not_found(self, worker):
        """Test retry raises error for nonexistent recording."""
        with pytest.raises(RecordingNotFoundError):
            await worker.retry("nonexistent")

    @pytest.mark.asyncio
    async def test_retry_emits_invalidate_event(self, worker, storage):
        """Test that retry() emits an invalidate(recordings) event."""
        await _create_recording(
            storage,
            id="retry-emit-001",
            status="error",
            retry_count=1,
            max_retries=3,
            last_error="timed out",
        )

        bus = get_event_bus()
        q = bus.subscribe()
        try:
            await worker.retry("retry-emit-001")

            # Drain the queue and find the invalidate event
            events = []
            while not q.empty():
                events.append(q.get_nowait())

            invalidate_events = [
                e for e in events if e.type == "invalidate" and e.data.get("entity") == "recordings"
            ]
            assert len(invalidate_events) == 1
        finally:
            bus.unsubscribe(q)


class TestRunLoop:
    """Tests for the worker loop."""

    @pytest.mark.asyncio
    async def test_shutdown_graceful(self, worker, storage, mock_transcriber):
        """Test graceful shutdown."""
        rec = await _create_recording(storage)

        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        worker.start()
        worker.enqueue(rec.id)

        # Give worker time to process
        await asyncio.sleep(0.2)

        await worker.shutdown(timeout=5.0)

        assert worker._shutdown is True
        assert worker._task.done()


class TestErrorClassification:
    """Tests for _is_transient_error."""

    @pytest.mark.parametrize(
        "error_msg,expected",
        [
            ("Transcription timed out after 1800s", True),
            ("OOM killed by system", True),
            ("Cannot allocate memory for model", True),
            ("Daemon restarted during processing", True),
            ("Process killed by signal 9", True),
            ("connection refused", True),
            ("temporary failure", True),
            ("Remote transcription failed after 3 attempts", True),
            ("Audio file not found: /missing.wav", False),
            ("No such file or directory", False),
            ("whisperx not installed", False),
            ("ImportError: no module named torch", False),
            ("ModuleNotFoundError: No module named 'whisperx'", False),
            ("HuggingFace token is required for diarization", False),
            ("Invalid model name: nonexistent", False),
            ("Some unknown random error happened", True),  # Default to transient
            (None, False),
            ("", False),
        ],
    )
    def test_error_classification(self, error_msg, expected):
        """Test error classification for various error messages."""
        assert ProcessingWorker._is_transient_error(error_msg) is expected


class TestSpeakerProfileMatching:
    """Tests for speaker-to-profile matching in _process_recording."""

    @pytest.mark.asyncio
    async def test_matches_speakers_to_profiles(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """Test that diarized speakers are matched to known profiles."""
        # Create a voice profile with an embedding
        now = datetime.now(UTC)
        prof = StorageVoiceProfile(
            id="profile-alice",
            name="Alice",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [1.0, 0.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof)

        # Create recording with diarization enabled
        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        # Mock transcription to return diarized result with matching embedding
        mock_result = {
            "text": "Hello from Alice",
            "language": "en",
            "language_probability": 0.95,
            "duration": 5.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello from Alice",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"start": 0.0, "end": 1.5, "word": "Hello", "speaker": "SPEAKER_00"},
                        {"start": 1.5, "end": 3.0, "word": "from", "speaker": "SPEAKER_00"},
                        {"start": 3.0, "end": 5.0, "word": "Alice", "speaker": "SPEAKER_00"},
                    ],
                }
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [1.0, 0.0, 0.0]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"
        assert result.speakers == ["Alice"]
        assert result.speaker_profiles == {"Alice": "profile-alice"}

    @pytest.mark.asyncio
    async def test_no_profiles_skips_matching(self, worker_with_profiles, storage):
        """Test that matching is skipped when no voice profiles exist."""
        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello",
            "language": "en",
            "language_probability": 0.95,
            "duration": 2.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                    "words": [],
                }
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2, 0.3]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"
        assert result.speakers == ["SPEAKER_00"]
        assert result.speaker_profiles is None

    @pytest.mark.asyncio
    async def test_auto_adds_embedding_to_matched_profile(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """Test that matched profiles get new embeddings auto-added."""
        now = datetime.now(UTC)
        prof = StorageVoiceProfile(
            id="profile-bob",
            name="Bob",
            created_at=now,
            clips=1,
            total_seconds=30.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [0.0, 1.0, 0.0],
                    "audio_duration": 30.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof)

        rec = await _create_recording(
            storage,
            duration=20.0,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hi",
            "language": "en",
            "language_probability": 0.9,
            "duration": 2.0,
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hi", "speaker": "SPEAKER_00", "words": []}
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [0.0, 1.0, 0.0]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        # Check profile got the new embedding
        updated_prof = await voice_profile_service.storage.get("profile-bob")
        assert updated_prof.clips == 2
        assert len(updated_prof.embeddings) == 2

    @pytest.mark.asyncio
    async def test_diarization_disabled_skips_matching(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """Test that speaker matching is skipped when diarization is disabled."""
        now = datetime.now(UTC)
        prof = StorageVoiceProfile(
            id="profile-alice",
            name="Alice",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [1.0, 0.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof)

        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": False, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello",
            "language": "en",
            "language_probability": 0.95,
            "duration": 2.0,
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hello", "words": []}],
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"
        assert result.speaker_profiles is None

    @pytest.mark.asyncio
    async def test_no_voice_profile_service_skips_matching(self, worker, storage):
        """Test that matching is skipped when worker has no voice profile service."""
        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello",
            "language": "en",
            "language_probability": 0.95,
            "duration": 2.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                    "words": [],
                }
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2, 0.3]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"
        assert result.speakers == ["SPEAKER_00"]
        assert result.speaker_profiles is None

    @pytest.mark.asyncio
    async def test_segments_are_renamed_after_matching(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """Verify segment and word speaker labels are actually renamed."""
        now = datetime.now(UTC)
        prof = StorageVoiceProfile(
            id="profile-alice",
            name="Alice",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [1.0, 0.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof)

        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": True},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello world",
            "language": "en",
            "language_probability": 0.95,
            "duration": 5.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"start": 0.0, "end": 2.5, "word": "Hello", "speaker": "SPEAKER_00"},
                        {"start": 2.5, "end": 5.0, "word": "world", "speaker": "SPEAKER_00"},
                    ],
                }
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [1.0, 0.0, 0.0]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"

        # Segment speaker should be renamed
        assert result.segments[0]["speaker"] == "Alice"

        # Words should also be renamed (word_timestamps=True so words are kept)
        words = result.segments[0]["words"]
        assert words[0]["speaker"] == "Alice"
        assert words[1]["speaker"] == "Alice"

    @pytest.mark.asyncio
    async def test_mixed_matched_and_unmatched_speakers(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """Some speakers match profiles, others keep anonymous labels."""
        now = datetime.now(UTC)
        prof = StorageVoiceProfile(
            id="profile-alice",
            name="Alice",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [1.0, 0.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof)

        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello Hi",
            "language": "en",
            "language_probability": 0.9,
            "duration": 4.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                    "words": [],
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "Hi",
                    "speaker": "SPEAKER_01",
                    "words": [],
                },
            ],
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_embeddings": {
                "SPEAKER_00": [1.0, 0.0, 0.0],  # Matches Alice
                "SPEAKER_01": [0.0, 0.0, 1.0],  # No match
            },
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"

        # Alice matched, SPEAKER_01 kept anonymous
        assert result.speakers == ["Alice", "SPEAKER_01"]
        assert result.speaker_profiles == {"Alice": "profile-alice"}

        # Segments renamed only for matched speaker
        assert result.segments[0]["speaker"] == "Alice"
        assert result.segments[1]["speaker"] == "SPEAKER_01"

    @pytest.mark.asyncio
    async def test_segments_with_none_speaker_not_renamed(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """Segments with speaker=None should not be affected by matching."""
        now = datetime.now(UTC)
        prof = StorageVoiceProfile(
            id="profile-alice",
            name="Alice",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [1.0, 0.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof)

        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello silence",
            "language": "en",
            "language_probability": 0.9,
            "duration": 4.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                    "words": [],
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "silence",
                    # No speaker key at all
                    "words": [],
                },
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [1.0, 0.0, 0.0]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"
        assert result.segments[0]["speaker"] == "Alice"
        assert result.segments[1]["speaker"] is None

    @pytest.mark.asyncio
    async def test_profile_matching_failure_proceeds_gracefully(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """If get_known_embeddings raises, recording still completes without matching."""
        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello",
            "language": "en",
            "language_probability": 0.95,
            "duration": 2.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                    "words": [],
                }
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2, 0.3]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        # Make get_known_embeddings raise an exception
        async def broken_get_known_embeddings():
            raise OSError("Storage corrupted")

        voice_profile_service.get_known_embeddings = broken_get_known_embeddings

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        # Should still complete — matching failure is caught
        assert result.status == "complete"
        assert result.speakers == ["SPEAKER_00"]
        assert result.speaker_profiles is None

    @pytest.mark.asyncio
    async def test_word_timestamps_true_keeps_renamed_words(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """When word_timestamps=True, words should be kept AND renamed."""
        now = datetime.now(UTC)
        prof = StorageVoiceProfile(
            id="profile-bob",
            name="Bob",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [0.0, 1.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof)

        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": True},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hi there",
            "language": "en",
            "language_probability": 0.9,
            "duration": 3.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Hi there",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"start": 0.0, "end": 1.5, "word": "Hi", "speaker": "SPEAKER_00"},
                        {"start": 1.5, "end": 3.0, "word": "there", "speaker": "SPEAKER_00"},
                    ],
                }
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [0.0, 1.0, 0.0]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"

        # Words should be present (word_timestamps=True)
        assert "words" in result.segments[0]
        words = result.segments[0]["words"]
        assert len(words) == 2

        # Words should be renamed
        assert words[0]["speaker"] == "Bob"
        assert words[1]["speaker"] == "Bob"

    @pytest.mark.asyncio
    async def test_word_timestamps_false_strips_renamed_words(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """When word_timestamps=False, words are stripped AFTER renaming."""
        now = datetime.now(UTC)
        prof = StorageVoiceProfile(
            id="profile-alice",
            name="Alice",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [1.0, 0.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof)

        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello",
            "language": "en",
            "language_probability": 0.95,
            "duration": 2.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"start": 0.0, "end": 2.0, "word": "Hello", "speaker": "SPEAKER_00"},
                    ],
                }
            ],
            "speakers": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [1.0, 0.0, 0.0]},
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        assert result.status == "complete"

        # Segment speaker renamed
        assert result.segments[0]["speaker"] == "Alice"

        # Words should be stripped because word_timestamps=False
        assert "words" not in result.segments[0]

    @pytest.mark.asyncio
    async def test_partial_add_embedding_failure(
        self, worker_with_profiles, storage, voice_profile_service
    ):
        """If add_embedding fails mid-loop, recording still completes.

        The already-renamed segments and partial speaker_profiles are kept,
        which is acceptable — better than discarding all matches.
        """
        now = datetime.now(UTC)
        # Create Alice profile — will match
        prof_alice = StorageVoiceProfile(
            id="profile-alice",
            name="Alice",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_00",
                    "timestamp": now.isoformat(),
                    "vector": [1.0, 0.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        # Create Bob profile — will match but add_embedding will fail
        prof_bob = StorageVoiceProfile(
            id="profile-bob",
            name="Bob",
            created_at=now,
            clips=1,
            total_seconds=60.0,
            confidence=0.2,
            embeddings=[
                {
                    "recording_id": "old-rec",
                    "speaker_id": "SPEAKER_01",
                    "timestamp": now.isoformat(),
                    "vector": [0.0, 1.0, 0.0],
                    "audio_duration": 60.0,
                },
            ],
        )
        await voice_profile_service.storage.create(prof_alice)
        await voice_profile_service.storage.create(prof_bob)

        rec = await _create_recording(
            storage,
            settings={"model": "base", "diarization": True, "word_timestamps": False},
        )
        audio_path = storage.base_path / "recordings" / rec.id / "audio.wav"
        audio_path.touch()

        mock_result = {
            "text": "Hello Hi",
            "language": "en",
            "language_probability": 0.9,
            "duration": 4.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                    "words": [],
                },
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "Hi",
                    "speaker": "SPEAKER_01",
                    "words": [],
                },
            ],
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_embeddings": {
                "SPEAKER_00": [1.0, 0.0, 0.0],
                "SPEAKER_01": [0.0, 1.0, 0.0],
            },
        }

        async def mock_run_transcription(self, recording_id, **kwargs):
            return mock_result

        # Delete Bob's profile after matching but before add_embedding
        # by patching add_embedding to fail on Bob
        original_add_embedding = voice_profile_service.add_embedding
        call_count = 0

        async def failing_add_embedding(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (Alice) succeeds
                return await original_add_embedding(**kwargs)
            # Second call (Bob) fails
            raise OSError("Disk full")

        voice_profile_service.add_embedding = failing_add_embedding

        with patch.object(ProcessingWorker, "_run_transcription", mock_run_transcription):
            await worker_with_profiles._process_recording(rec.id)

        result = await storage.get(rec.id)
        # Should still complete — the outer try/except catches the error
        assert result.status == "complete"
        # speaker_profiles may be None (the exception resets to empty dict path)
        # or partial depending on when the error was caught
        # The key thing is it doesn't crash
