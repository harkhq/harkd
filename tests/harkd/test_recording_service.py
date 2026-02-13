"""Tests for recording service."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harkd.api.models.recording import (
    ProcessingStage,
    RecordingCreate,
    RecordingOverrides,
    RecordingStatus,
    RecordingUpdate,
)
from harkd.config import HarkdSettings, LLMSettings, RecordingDefaults, StorageSettings
from harkd.exceptions import (
    InvalidStateError,
    NoActiveRecordingError,
    RecordingInProgressError,
    RecordingNotFoundError,
)
from harkd.services.recording_service import RecordingService
from harkd.state.recording_state import RecordingState
from harkd.storage.filesystem.recordings import FilesystemRecordingStorage
from harkd.storage.models import StorageRecording


@pytest.fixture
def storage(tmp_path):
    """Create recording storage."""
    return FilesystemRecordingStorage(tmp_path)


@pytest.fixture
def recording_state():
    """Create fresh recording state."""
    return RecordingState()


@pytest.fixture
def config(tmp_path):
    """Create test daemon config."""
    return HarkdSettings(storage=StorageSettings(base_path=tmp_path))


@pytest.fixture
def service(storage, config, recording_state):
    """Create recording service."""
    return RecordingService(
        storage=storage, config=config, recording_state=recording_state
    )


@pytest.fixture
def mock_recorder():
    """Mock AudioRecorder."""
    with patch("harkd.services.recording_service.AudioRecorder") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_transcriber():
    """Mock transcription subprocess."""
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

    async def mock_run_subprocess(
        self, recording_id, audio_path, model_name, language, word_timestamps,
        diarize=False, hf_token=None,
    ):
        return mock_result

    with patch.object(
        RecordingService, "_run_transcription_subprocess", mock_run_subprocess
    ):
        yield mock_result


class TestStartRecording:
    """Tests for start_recording."""

    @pytest.mark.asyncio
    async def test_start_recording_success(self, service, mock_recorder):
        """Test successful recording start."""
        request = RecordingCreate(title="Test Recording")

        result = await service.start_recording(request)

        assert result.status == RecordingStatus.RECORDING
        assert result.title == "Test Recording"
        assert result.duration == 0.0
        assert result.audio_level == 0.0
        assert result.processing_stage is None
        assert result.processing_progress is None
        assert result.transcript is None
        assert result.segments is None

        # Verify recorder was started
        mock_recorder.start.assert_called_once()

        # Verify state updated
        assert service.recording_state.is_recording
        assert service.recording_state.active_recording_id == result.id
        assert service.recording_state.start_time is not None

    @pytest.mark.asyncio
    async def test_start_recording_generates_id(self, service, mock_recorder):
        """Test recording ID generation."""
        request = RecordingCreate()

        result = await service.start_recording(request)

        # ID format: YYYY-MM-DD-XXXXXXXX
        assert len(result.id) > 10
        assert result.id.startswith(datetime.now(UTC).strftime("%Y-%m-%d"))

    @pytest.mark.asyncio
    async def test_start_recording_default_title(self, service, mock_recorder):
        """Test default title when not provided."""
        request = RecordingCreate()  # No title

        result = await service.start_recording(request)

        assert result.title == "Untitled Recording"

    @pytest.mark.asyncio
    async def test_start_recording_creates_directory(self, service, mock_recorder):
        """Test audio directory creation."""
        request = RecordingCreate()

        result = await service.start_recording(request)

        # Directory is created at {base_path}/recordings/{id}/
        audio_dir = service.storage.base_path / "recordings" / result.id
        assert audio_dir.exists()
        assert audio_dir.is_dir()

    @pytest.mark.asyncio
    async def test_start_recording_already_recording(self, service, mock_recorder):
        """Test error when already recording."""
        request = RecordingCreate()

        # Start first recording
        first = await service.start_recording(request)

        # Try to start second recording
        with pytest.raises(RecordingInProgressError) as exc_info:
            await service.start_recording(request)

        # Verify the active recording ID is in the error details
        assert exc_info.value.details["active_recording_id"] == first.id

    @pytest.mark.asyncio
    async def test_start_recording_cleans_up_on_error(self, service):
        """Test cleanup when recorder.start() fails."""
        request = RecordingCreate()

        with patch("harkd.services.recording_service.AudioRecorder") as mock_recorder:
            mock_recorder.return_value.start.side_effect = RuntimeError("Audio error")

            with pytest.raises(RuntimeError):
                await service.start_recording(request)

        # Verify state not updated
        assert not service.recording_state.is_recording

    @pytest.mark.asyncio
    async def test_start_recording_uses_daemon_defaults(self, service, mock_recorder):
        """Test recording uses daemon defaults when no overrides given."""
        request = RecordingCreate()

        result = await service.start_recording(request)

        # Get stored settings
        stored = await service.storage.get(result.id)
        assert stored.settings["model"] == "large-v3"
        assert stored.settings["input_source"] == "both"
        assert stored.settings["language"] == "auto"
        assert stored.settings["diarization"] is True

    @pytest.mark.asyncio
    async def test_start_recording_partial_overrides(self, service, mock_recorder):
        """Test per-recording overrides merge with daemon defaults."""
        request = RecordingCreate(
            settings=RecordingOverrides(language="en", diarization=False)
        )

        result = await service.start_recording(request)

        stored = await service.storage.get(result.id)
        # Overridden
        assert stored.settings["language"] == "en"
        assert stored.settings["diarization"] is False
        # From daemon defaults
        assert stored.settings["model"] == "large-v3"
        assert stored.settings["input_source"] == "both"
        assert stored.settings["word_timestamps"] is False

    @pytest.mark.asyncio
    async def test_start_recording_all_overrides(self, service, mock_recorder):
        """Test all overridable settings can be overridden."""
        request = RecordingCreate(
            settings=RecordingOverrides(
                language="de",
                input_source="mic",
                diarization=False,
                noise_reduction=False,
                normalization=False,
                word_timestamps=True,
            )
        )

        result = await service.start_recording(request)

        stored = await service.storage.get(result.id)
        assert stored.settings["language"] == "de"
        assert stored.settings["input_source"] == "mic"
        assert stored.settings["diarization"] is False
        assert stored.settings["noise_reduction"] is False
        assert stored.settings["normalization"] is False
        assert stored.settings["word_timestamps"] is True
        # model always from daemon
        assert stored.settings["model"] == "large-v3"

    @pytest.mark.asyncio
    async def test_start_recording_custom_daemon_config(
        self, storage, recording_state, mock_recorder, tmp_path
    ):
        """Test that custom daemon config is respected."""
        custom_config = HarkdSettings(
            storage=StorageSettings(base_path=tmp_path),
            recording=RecordingDefaults(model="small", language="fr"),
        )
        svc = RecordingService(
            storage=storage, config=custom_config, recording_state=recording_state
        )

        result = await svc.start_recording(RecordingCreate())

        stored = await storage.get(result.id)
        assert stored.settings["model"] == "small"
        assert stored.settings["language"] == "fr"


class TestStopRecording:
    """Tests for stop_recording."""

    @pytest.mark.asyncio
    async def test_stop_recording_success(
        self, service, mock_recorder, mock_transcriber
    ):
        """Test successful recording stop."""
        # Start recording
        start_result = await service.start_recording(RecordingCreate())
        recording_id = start_result.id

        # Wait a bit for duration
        await asyncio.sleep(0.1)

        # Stop recording
        stop_result = await service.stop_recording(recording_id)

        assert stop_result.status == RecordingStatus.PROCESSING
        assert stop_result.duration > 0
        assert stop_result.processing_stage == ProcessingStage.PREPROCESSING
        assert stop_result.processing_progress == 0.0

        # Verify state cleared
        assert not service.recording_state.is_recording

    @pytest.mark.asyncio
    async def test_stop_recording_not_found(self, service):
        """Test error when recording doesn't exist."""
        with pytest.raises(RecordingNotFoundError):
            await service.stop_recording("nonexistent-id")

    @pytest.mark.asyncio
    async def test_stop_recording_invalid_state(self, service, storage):
        """Test error when recording not in recording state."""
        # Create a completed recording
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            settings={},
        )
        await storage.create(recording)

        with pytest.raises(InvalidStateError):
            await service.stop_recording("test-123")

    @pytest.mark.asyncio
    async def test_stop_recording_starts_processing(
        self, service, mock_recorder, mock_transcriber
    ):
        """Test that processing task is started."""
        # Start recording
        start_result = await service.start_recording(RecordingCreate())
        recording_id = start_result.id

        # Create audio file (required for processing)
        audio_path = (
            service.storage.base_path / "recordings" / recording_id / "audio.wav"
        )
        audio_path.touch()

        # Stop recording
        await service.stop_recording(recording_id)

        # Verify processing task exists
        assert recording_id in service._processing_tasks


class TestGetActiveRecording:
    """Tests for get_active_recording."""

    @pytest.mark.asyncio
    async def test_get_active_recording_success(self, service, mock_recorder):
        """Test getting the active recording."""
        # Start recording
        start_result = await service.start_recording(RecordingCreate(title="Test"))
        recording_id = start_result.id

        # Get active recording
        result = await service.get_active_recording()

        assert result.id == recording_id
        assert result.status == RecordingStatus.RECORDING
        assert result.title == "Test"

    @pytest.mark.asyncio
    async def test_get_active_recording_no_active(self, service):
        """Test error when no recording is active."""
        with pytest.raises(NoActiveRecordingError):
            await service.get_active_recording()

    @pytest.mark.asyncio
    async def test_get_active_recording_real_time_duration(
        self, service, mock_recorder
    ):
        """Test that duration is updated in real-time."""
        # Start recording
        await service.start_recording(RecordingCreate())

        # Wait a bit
        await asyncio.sleep(0.1)

        # Get active recording
        result = await service.get_active_recording()

        # Duration should be > 0
        assert result.duration > 0


class TestStopActiveRecording:
    """Tests for stop_active_recording."""

    @pytest.mark.asyncio
    async def test_stop_active_recording_success(
        self, service, mock_recorder, mock_transcriber
    ):
        """Test stopping the active recording."""
        # Start recording
        start_result = await service.start_recording(RecordingCreate())
        recording_id = start_result.id

        # Wait a bit for duration
        await asyncio.sleep(0.1)

        # Stop active recording
        stop_result = await service.stop_active_recording()

        assert stop_result.id == recording_id
        assert stop_result.status == RecordingStatus.PROCESSING
        assert stop_result.duration > 0
        assert stop_result.processing_stage == ProcessingStage.PREPROCESSING
        assert stop_result.processing_progress == 0.0

        # Verify state cleared
        assert not service.recording_state.is_recording

    @pytest.mark.asyncio
    async def test_stop_active_recording_no_active(self, service):
        """Test error when no recording is active."""
        with pytest.raises(NoActiveRecordingError):
            await service.stop_active_recording()

    @pytest.mark.asyncio
    async def test_stop_active_recording_same_behavior_as_stop(
        self, service, mock_recorder, mock_transcriber, storage
    ):
        """Test that stop_active_recording delegates to stop_recording."""
        # Start recording
        start_result = await service.start_recording(RecordingCreate())
        recording_id = start_result.id

        # Create audio file
        audio_path = (
            service.storage.base_path / "recordings" / recording_id / "audio.wav"
        )
        audio_path.touch()

        # Stop using active endpoint
        result_active = await service.stop_active_recording()

        # Verify behavior is the same
        assert result_active.status == RecordingStatus.PROCESSING
        assert not service.recording_state.is_recording
        assert recording_id in service._processing_tasks


class TestCancelRecording:
    """Tests for cancel_recording."""

    @pytest.mark.asyncio
    async def test_cancel_active_recording(self, service, mock_recorder):
        """Test canceling an active recording."""
        # Start recording
        result = await service.start_recording(RecordingCreate())
        recording_id = result.id

        # Cancel
        await service.cancel_recording(recording_id)

        # Verify deleted
        with pytest.raises(RecordingNotFoundError):
            await service.get_recording(recording_id)

        # Verify state cleared
        assert not service.recording_state.is_recording

    @pytest.mark.asyncio
    async def test_cancel_processing_recording(
        self, service, mock_recorder, mock_transcriber
    ):
        """Test canceling a processing recording."""
        # Start and stop recording
        result = await service.start_recording(RecordingCreate())
        recording_id = result.id

        # Create audio file
        audio_path = (
            service.storage.base_path / "recordings" / recording_id / "audio.wav"
        )
        audio_path.touch()

        await service.stop_recording(recording_id)

        # Cancel while processing
        await service.cancel_recording(recording_id)

        # Verify deleted
        with pytest.raises(RecordingNotFoundError):
            await service.get_recording(recording_id)

    @pytest.mark.asyncio
    async def test_cancel_completed_recording(self, service, storage):
        """Test canceling a completed recording."""
        # Create completed recording
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            settings={},
        )
        await storage.create(recording)

        # Cancel
        await service.cancel_recording("test-123")

        # Verify deleted
        with pytest.raises(RecordingNotFoundError):
            await service.get_recording("test-123")

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_recording(self, service):
        """Test error when canceling nonexistent recording."""
        with pytest.raises(RecordingNotFoundError):
            await service.cancel_recording("nonexistent-id")


class TestGetRecording:
    """Tests for get_recording."""

    @pytest.mark.asyncio
    async def test_get_recording_success(self, service, storage):
        """Test getting a recording."""
        # Create recording
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            settings={},
        )
        await storage.create(recording)

        # Get recording
        result = await service.get_recording("test-123")

        assert result.id == "test-123"
        assert result.title == "Test"
        assert result.status == RecordingStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_get_recording_not_found(self, service):
        """Test error when recording doesn't exist."""
        with pytest.raises(RecordingNotFoundError):
            await service.get_recording("nonexistent-id")

    @pytest.mark.asyncio
    async def test_get_active_recording_updates_duration(
        self, service, mock_recorder, recording_state
    ):
        """Test that active recording duration is updated from state."""
        # Start recording
        result = await service.start_recording(RecordingCreate())
        recording_id = result.id

        # Wait a bit
        await asyncio.sleep(0.1)

        # Get recording
        updated = await service.get_recording(recording_id)

        # Duration should be > 0 (from state)
        assert updated.duration > 0


class TestListRecordings:
    """Tests for list_recordings."""

    @pytest.mark.asyncio
    async def test_list_recordings_empty(self, service):
        """Test listing when no recordings."""
        result = await service.list_recordings()

        assert result.total == 0
        assert result.recordings == []

    @pytest.mark.asyncio
    async def test_list_recordings(self, service, storage):
        """Test listing recordings."""
        # Create recordings
        for i in range(3):
            recording = StorageRecording(
                id=f"test-{i}",
                status=RecordingStatus.COMPLETE.value,
                created_at=datetime.now(UTC),
                title=f"Recording {i}",
                duration=10.0,
                settings={},
            )
            await storage.create(recording)

        # List
        result = await service.list_recordings()

        assert result.total == 3
        assert len(result.recordings) == 3

    @pytest.mark.asyncio
    async def test_list_recordings_pagination(self, service, storage):
        """Test pagination."""
        # Create recordings
        for i in range(5):
            recording = StorageRecording(
                id=f"test-{i}",
                status=RecordingStatus.COMPLETE.value,
                created_at=datetime.now(UTC),
                title=f"Recording {i}",
                duration=10.0,
                settings={},
            )
            await storage.create(recording)

        # Get first page
        result = await service.list_recordings(limit=2, offset=0)
        assert len(result.recordings) <= 2

        # Get second page
        result = await service.list_recordings(limit=2, offset=2)
        assert len(result.recordings) <= 2

    @pytest.mark.asyncio
    async def test_list_recordings_filter_by_status(self, service, storage):
        """Test filtering by status."""
        # Create recordings with different statuses
        await storage.create(
            StorageRecording(
                id="recording-1",
                status=RecordingStatus.COMPLETE.value,
                created_at=datetime.now(UTC),
                title="Complete",
                duration=10.0,
                settings={},
            )
        )
        await storage.create(
            StorageRecording(
                id="processing-1",
                status=RecordingStatus.PROCESSING.value,
                created_at=datetime.now(UTC),
                title="Processing",
                duration=5.0,
                settings={},
            )
        )

        # Filter by complete
        result = await service.list_recordings(status=RecordingStatus.COMPLETE)
        assert all(r.status == RecordingStatus.COMPLETE for r in result.recordings)


class TestUpdateRecording:
    """Tests for update_recording."""

    @pytest.mark.asyncio
    async def test_update_recording_title(self, service, storage):
        """Test updating recording title."""
        # Create recording
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Old Title",
            duration=10.0,
            settings={},
        )
        await storage.create(recording)

        # Update
        update = RecordingUpdate(title="New Title")
        result = await service.update_recording("test-123", update)

        assert result.title == "New Title"

    @pytest.mark.asyncio
    async def test_update_recording_speakers(self, service, storage):
        """Test updating speaker names."""
        # Create recording with segments
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.COMPLETE.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            segments=[
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "word": "Hello",
                            "speaker": "SPEAKER_01",
                        }
                    ],
                }
            ],
            speakers=["SPEAKER_01"],
            settings={},
        )
        await storage.create(recording)

        # Update speakers
        update = RecordingUpdate(speakers={"SPEAKER_01": "Alice"})
        result = await service.update_recording("test-123", update)

        assert result.speakers == ["Alice"]
        assert result.segments[0].speaker == "Alice"
        assert result.segments[0].words[0].speaker == "Alice"

    @pytest.mark.asyncio
    async def test_update_recording_not_complete(self, service, storage):
        """Test error when updating non-complete recording."""
        # Create processing recording
        recording = StorageRecording(
            id="test-123",
            status=RecordingStatus.PROCESSING.value,
            created_at=datetime.now(UTC),
            title="Test",
            duration=10.0,
            settings={},
        )
        await storage.create(recording)

        # Try to update
        update = RecordingUpdate(title="New Title")
        with pytest.raises(InvalidStateError):
            await service.update_recording("test-123", update)


class TestProcessing:
    """Tests for background processing."""

    @pytest.mark.asyncio
    async def test_processing_workflow(
        self, service, mock_recorder, mock_transcriber, tmp_path
    ):
        """Test complete processing workflow."""
        # Start recording
        result = await service.start_recording(RecordingCreate(title="Test"))
        recording_id = result.id

        # Create audio file
        audio_path = (
            service.storage.base_path / "recordings" / recording_id / "audio.wav"
        )
        audio_path.touch()

        # Stop recording
        await service.stop_recording(recording_id)

        # Wait for processing to complete (with timeout)
        task = service._processing_tasks.get(recording_id)
        if task:
            await asyncio.wait_for(task, timeout=5.0)

        # Verify final state
        final = await service.get_recording(recording_id)
        assert final.status == RecordingStatus.COMPLETE
        assert final.transcript == "Hello world"
        assert final.language == "en"
        assert len(final.segments) == 1

    @pytest.mark.asyncio
    async def test_processing_generates_title(
        self, service, mock_recorder, mock_transcriber
    ):
        """Test title generation for untitled recordings."""
        # Start recording without title
        result = await service.start_recording(RecordingCreate())
        recording_id = result.id

        # Create audio file
        audio_path = (
            service.storage.base_path / "recordings" / recording_id / "audio.wav"
        )
        audio_path.touch()

        # Stop and process
        await service.stop_recording(recording_id)

        # Wait for processing
        task = service._processing_tasks.get(recording_id)
        if task:
            await asyncio.wait_for(task, timeout=5.0)

        # Verify title generated
        final = await service.get_recording(recording_id)
        assert final.title != "Untitled Recording"
        assert len(final.title) > 0

    @pytest.mark.asyncio
    async def test_processing_error_sets_error_status(
        self, service, mock_recorder, storage
    ):
        """Test that processing failure sets error status and clears processing fields."""
        # Start recording
        result = await service.start_recording(RecordingCreate(title="Test"))
        recording_id = result.id

        # Create audio file
        audio_path = (
            service.storage.base_path / "recordings" / recording_id / "audio.wav"
        )
        audio_path.touch()

        # Make transcription subprocess raise an error
        async def failing_subprocess(self, *args, **kwargs):
            raise RuntimeError("Transcription failed")

        with patch.object(
            RecordingService,
            "_run_transcription_subprocess",
            failing_subprocess,
        ):
            await service.stop_recording(recording_id)

            # Wait for processing to complete
            task = service._processing_tasks.get(recording_id)
            if task:
                await asyncio.wait_for(task, timeout=5.0)

        # Verify error state
        final = await service.get_recording(recording_id)
        assert final.status == RecordingStatus.ERROR
        assert final.processing_stage is None
        assert final.processing_progress is None


class TestConvertTranscriptionResult:
    """Tests for _convert_transcription_result static method."""

    def test_empty_segments(self):
        """Test conversion with empty segments."""
        result_dict = {"segments": [], "language": "en", "language_probability": 0.9}
        segments, language, confidence, speakers = (
            RecordingService._convert_transcription_result(result_dict)
        )

        assert segments == []
        assert language == "en"
        assert confidence == 0.9
        assert speakers == []

    def test_conversion_with_words(self):
        """Test conversion with segments containing words."""
        result_dict = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Hello world",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {"start": 0.0, "end": 1.5, "word": "Hello"},
                        {"start": 1.5, "end": 3.0, "word": "world"},
                    ],
                }
            ],
            "language": "en",
            "language_probability": 0.95,
            "speakers": ["SPEAKER_01"],
        }
        segments, language, confidence, speakers = (
            RecordingService._convert_transcription_result(result_dict)
        )

        assert len(segments) == 1
        assert segments[0]["text"] == "Hello world"
        assert segments[0]["speaker"] == "SPEAKER_01"
        assert len(segments[0]["words"]) == 2
        assert segments[0]["words"][0]["word"] == "Hello"
        assert segments[0]["words"][0]["start"] == 0.0
        assert segments[0]["words"][0]["end"] == 1.5

    def test_missing_optional_fields(self):
        """Test conversion when optional fields are missing."""
        result_dict = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
            ],
        }
        segments, language, confidence, speakers = (
            RecordingService._convert_transcription_result(result_dict)
        )

        assert len(segments) == 1
        assert segments[0]["speaker"] is None
        assert segments[0]["words"] == []
        assert language == "unknown"
        assert confidence == 0.0
        assert speakers == []


class TestAudioLevelCallback:
    """Tests for audio level callback."""

    @pytest.mark.asyncio
    async def test_audio_level_callback_from_thread(
        self, service, mock_recorder, storage
    ):
        """Test that audio level callback works when called from a different thread.

        This tests the cross-thread async call handling that occurs when
        the audio recorder's thread calls the level callback.
        """
        import threading

        # Start recording
        result = await service.start_recording(RecordingCreate())
        recording_id = result.id

        # Simulate level callback being called from different thread
        # (as would happen with real AudioRecorder)
        def call_from_thread():
            service._update_audio_level(recording_id, 0.75)

        thread = threading.Thread(target=call_from_thread)
        thread.start()
        thread.join(timeout=2.0)

        # Give async task time to complete
        await asyncio.sleep(0.1)

        # Verify in-memory level was updated
        assert service._audio_level == 0.75

        # Verify the thread completed without error
        assert not thread.is_alive()

        # Verify the recording still exists and is in recording state
        recording = await service.get_recording(recording_id)
        assert recording.status == RecordingStatus.RECORDING


class TestDiarizationSubprocess:
    """Tests for diarization parameter passing to subprocess."""

    @pytest.mark.asyncio
    async def test_diarization_params_passed_to_subprocess(
        self, service, mock_recorder, tmp_path
    ):
        """Test that diarize and hf_token are passed to subprocess correctly."""
        # Set up config with diarization and hf_token
        service.config = HarkdSettings(
            storage=StorageSettings(base_path=tmp_path),
            recording=RecordingDefaults(diarization=True),
            hf_token="hf_test_token",
        )

        diarized_result = {
            "text": "Hello from speaker 1",
            "language": "en",
            "language_probability": 0.9,
            "duration": 3.0,
            "speakers": ["SPEAKER_01"],
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Hello from speaker 1",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "word": "Hello",
                            "speaker": "SPEAKER_01",
                        }
                    ],
                }
            ],
        }

        captured_kwargs = {}

        async def mock_subprocess(self, recording_id, audio_path, model_name,
                                  language, word_timestamps, diarize=False, hf_token=None):
            captured_kwargs["diarize"] = diarize
            captured_kwargs["hf_token"] = hf_token
            return diarized_result

        with patch.object(
            RecordingService, "_run_transcription_subprocess", mock_subprocess
        ):
            result = await service.start_recording(RecordingCreate())
            recording_id = result.id

            audio_path = (
                service.storage.base_path / "recordings" / recording_id / "audio.wav"
            )
            audio_path.touch()

            await service.stop_recording(recording_id)

            task = service._processing_tasks.get(recording_id)
            if task:
                await asyncio.wait_for(task, timeout=5.0)

        assert captured_kwargs["diarize"] is True
        assert captured_kwargs["hf_token"] == "hf_test_token"

        final = await service.get_recording(recording_id)
        assert final.status == RecordingStatus.COMPLETE
        assert final.speakers == ["SPEAKER_01"]


class TestMeetingMinutesIntegration:
    """Tests for meeting minutes in recording pipeline."""

    @pytest.mark.asyncio
    async def test_meeting_minutes_called_when_llm_enabled(
        self, storage, recording_state, mock_recorder, tmp_path
    ):
        """Test that meeting minutes are generated when LLM is enabled."""
        config = HarkdSettings(
            storage=StorageSettings(base_path=tmp_path),
            recording=RecordingDefaults(diarization=False),
            llm=LLMSettings(enabled=True, provider="openai", api_key="sk-test"),
        )
        svc = RecordingService(
            storage=storage, config=config, recording_state=recording_state
        )

        mock_transcription = {
            "text": "We decided to use React",
            "language": "en",
            "language_probability": 0.95,
            "duration": 5.0,
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "We decided to use React", "words": []},
            ],
        }

        async def mock_subprocess(self, *args, **kwargs):
            return mock_transcription

        from harkd.llm.types import MeetingMinutesResult

        mock_minutes = MeetingMinutesResult(
            executive_summary=["Team decided to use React"],
            meeting_notes=[{"topic": "Tech Stack", "content": "Chose React"}],
            tasks=[{"task": "Set up React", "assignee": None, "due": None}],
            decisions=["Use React for frontend"],
        )

        with (
            patch.object(
                RecordingService, "_run_transcription_subprocess", mock_subprocess
            ),
            patch("harkd.llm.client.LLMClient") as mock_llm_cls,
        ):
            mock_llm = MagicMock()
            mock_llm.generate_meeting_minutes = AsyncMock(return_value=mock_minutes)
            mock_llm_cls.return_value = mock_llm

            result = await svc.start_recording(RecordingCreate())
            recording_id = result.id

            audio_path = storage.base_path / "recordings" / recording_id / "audio.wav"
            audio_path.touch()

            await svc.stop_recording(recording_id)

            task = svc._processing_tasks.get(recording_id)
            if task:
                await asyncio.wait_for(task, timeout=5.0)

        final = await svc.get_recording(recording_id)
        assert final.status == RecordingStatus.COMPLETE
        assert final.executive_summary == ["Team decided to use React"]
        assert len(final.meeting_notes) == 1
        assert len(final.tasks) == 1
        assert final.decisions == ["Use React for frontend"]

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_prevent_completion(
        self, storage, recording_state, mock_recorder, tmp_path
    ):
        """Test that LLM failure doesn't prevent recording completion."""
        config = HarkdSettings(
            storage=StorageSettings(base_path=tmp_path),
            recording=RecordingDefaults(diarization=False),
            llm=LLMSettings(enabled=True, provider="openai", api_key="sk-test"),
        )
        svc = RecordingService(
            storage=storage, config=config, recording_state=recording_state
        )

        mock_transcription = {
            "text": "Hello world",
            "language": "en",
            "language_probability": 0.95,
            "duration": 2.0,
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello world", "words": []},
            ],
        }

        async def mock_subprocess(self, *args, **kwargs):
            return mock_transcription

        with (
            patch.object(
                RecordingService, "_run_transcription_subprocess", mock_subprocess
            ),
            patch("harkd.llm.client.LLMClient") as mock_llm_cls,
        ):
            mock_llm = MagicMock()
            mock_llm.generate_meeting_minutes = AsyncMock(
                side_effect=RuntimeError("LLM service unavailable")
            )
            mock_llm_cls.return_value = mock_llm

            result = await svc.start_recording(RecordingCreate())
            recording_id = result.id

            audio_path = storage.base_path / "recordings" / recording_id / "audio.wav"
            audio_path.touch()

            await svc.stop_recording(recording_id)

            task = svc._processing_tasks.get(recording_id)
            if task:
                await asyncio.wait_for(task, timeout=5.0)

        # Recording should still complete despite LLM failure
        final = await svc.get_recording(recording_id)
        assert final.status == RecordingStatus.COMPLETE
        assert final.transcript == "Hello world"
        # Meeting minutes fields remain empty
        assert final.executive_summary == []
        assert final.meeting_notes == []

    @pytest.mark.asyncio
    async def test_meeting_minutes_not_called_when_llm_disabled(
        self, service, mock_recorder, mock_transcriber
    ):
        """Test that meeting minutes are skipped when LLM is disabled."""
        # Default config has LLM disabled
        assert service.config.llm.enabled is False

        result = await service.start_recording(RecordingCreate())
        recording_id = result.id

        audio_path = (
            service.storage.base_path / "recordings" / recording_id / "audio.wav"
        )
        audio_path.touch()

        await service.stop_recording(recording_id)

        task = service._processing_tasks.get(recording_id)
        if task:
            await asyncio.wait_for(task, timeout=5.0)

        final = await service.get_recording(recording_id)
        assert final.status == RecordingStatus.COMPLETE
        # No meeting minutes
        assert final.executive_summary == []
