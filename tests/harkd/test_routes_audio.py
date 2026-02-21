"""Tests for audio clip extraction endpoint."""

import struct
import wave

import pytest
from fastapi.testclient import TestClient

from harkd.api.app import create_app
from harkd.config import HarkdSettings, StorageSettings

# Clips from 16kHz sources are upsampled to 48kHz for browser playback
OUTPUT_RATE = 48000
UPSAMPLE_RATIO = OUTPUT_RATE // 16000  # 3


def create_test_wav(path, duration=1.0, sample_rate=16000):
    """Create a valid WAV file with silence."""
    n_frames = int(duration * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))


@pytest.fixture
def settings(tmp_path):
    """Create test settings."""
    return HarkdSettings(storage=StorageSettings(base_path=tmp_path))


@pytest.fixture
def client(settings):
    """Create test client."""
    from harkd.api import deps
    from harkd.config import get_settings

    deps._recording_services.clear()
    deps._voice_profile_services.clear()
    deps._processing_workers.clear()

    from harkd.state.recording_state import get_recording_state

    get_recording_state.cache_clear()

    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def _create_recording_with_audio(settings, recording_id="rec-001", duration=1.0):
    """Create a recording with metadata and WAV file."""
    import json

    rec_dir = settings.storage.base_path / "recordings" / recording_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": recording_id,
        "status": "complete",
        "created_at": "2025-01-01T00:00:00",
        "title": "Test Recording",
        "duration": duration,
        "settings": {},
        "segments": [],
        "speakers": [],
    }
    with open(rec_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    create_test_wav(rec_dir / "audio.wav", duration=duration)
    return recording_id


def test_get_audio_clip_success(client, settings):
    """Test extracting a valid audio clip (upsampled to 48kHz)."""
    _create_recording_with_audio(settings, duration=1.0)

    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=0&end=0.5")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    import io

    buf = io.BytesIO(response.content)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == OUTPUT_RATE
        # 0.5s at 48kHz = 24000 frames
        assert wf.getnframes() == 24000


def test_get_audio_clip_recording_not_found(client):
    """Test 404 for nonexistent recording."""
    response = client.get("/api/v1/recordings/nonexistent/audio/clip?start=0&end=1")
    assert response.status_code == 404

    data = response.json()
    assert data["detail"]["error"]["code"] == "RECORDING_NOT_FOUND"


def test_get_audio_clip_no_audio_file(client, settings):
    """Test 404 when recording exists but audio file is missing."""
    import json

    rec_dir = settings.storage.base_path / "recordings" / "rec-no-audio"
    rec_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": "rec-no-audio",
        "status": "complete",
        "created_at": "2025-01-01T00:00:00",
        "title": "No Audio",
        "duration": 0,
        "settings": {},
    }
    with open(rec_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    response = client.get("/api/v1/recordings/rec-no-audio/audio/clip?start=0&end=1")
    assert response.status_code == 404

    data = response.json()
    assert data["detail"]["error"]["code"] == "AUDIO_NOT_FOUND"


def test_get_audio_clip_invalid_range(client, settings):
    """Test 422 when start >= end."""
    _create_recording_with_audio(settings)

    # start == end (end must be > 0 and > start)
    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=0.5&end=0.5")
    assert response.status_code == 422

    # start > end
    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=1&end=0.5")
    assert response.status_code == 422


def test_get_audio_clip_range_clamped(client, settings):
    """Test that end beyond duration is clamped to file length."""
    _create_recording_with_audio(settings, duration=1.0)

    # Request beyond the actual duration
    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=0&end=5.0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    # 1s at 16kHz → upsampled to 48kHz = 48000 frames
    import io

    buf = io.BytesIO(response.content)
    with wave.open(buf, "rb") as wf:
        assert wf.getnframes() == OUTPUT_RATE


def test_get_audio_clip_content_disposition_inline(client, settings):
    """Test that Content-Disposition is set to inline."""
    _create_recording_with_audio(settings)

    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=0&end=0.5")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == "inline"


def test_get_audio_clip_cache_control(client, settings):
    """Test that Cache-Control prevents caching."""
    _create_recording_with_audio(settings)

    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=0&end=0.5")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_get_audio_clip_start_beyond_duration(client, settings):
    """Test that start beyond file duration returns valid empty WAV."""
    _create_recording_with_audio(settings, duration=1.0)

    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=5&end=10")
    assert response.status_code == 200

    import io

    buf = io.BytesIO(response.content)
    with wave.open(buf, "rb") as wf:
        assert wf.getnframes() == 0


def test_get_audio_clip_stereo_downmixed_to_mono(client, settings):
    """Test that stereo WAV files are downmixed to mono and upsampled."""
    import io
    import json

    rec_dir = settings.storage.base_path / "recordings" / "rec-stereo"
    rec_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": "rec-stereo",
        "status": "complete",
        "created_at": "2025-01-01T00:00:00",
        "title": "Stereo",
        "duration": 1.0,
        "settings": {},
    }
    with open(rec_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create stereo WAV with distinct L/R values
    n_frames = 16000
    # L=1000, R=3000 for each frame → mono avg should be 2000
    stereo_samples = []
    for _ in range(n_frames):
        stereo_samples.extend([1000, 3000])
    with wave.open(str(rec_dir / "audio.wav"), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n_frames * 2}h", *stereo_samples))

    response = client.get("/api/v1/recordings/rec-stereo/audio/clip?start=0&end=0.5")
    assert response.status_code == 200

    buf = io.BytesIO(response.content)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == OUTPUT_RATE
        # 0.5s at 48kHz = 24000 frames
        assert wf.getnframes() == 24000

        # Verify the downmix averaged the channels (constant value → preserved)
        raw = wf.readframes(1)
        (sample,) = struct.unpack("<h", raw)
        assert sample == 2000


def test_get_audio_clip_48khz_no_resample(client, settings):
    """Test with 48kHz source — no resampling needed."""
    _create_recording_with_audio(settings, recording_id="rec-48k", duration=1.0)

    # Recreate at 48kHz
    rec_dir = settings.storage.base_path / "recordings" / "rec-48k"
    n_frames = 48000
    with wave.open(str(rec_dir / "audio.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))

    response = client.get("/api/v1/recordings/rec-48k/audio/clip?start=0&end=0.5")
    assert response.status_code == 200

    import io

    buf = io.BytesIO(response.content)
    with wave.open(buf, "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnframes() == 24000  # 0.5s at 48kHz


def test_get_audio_clip_very_short(client, settings):
    """Test extracting a very short clip (1ms)."""
    _create_recording_with_audio(settings, duration=1.0)

    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=0&end=0.001")
    assert response.status_code == 200

    import io

    buf = io.BytesIO(response.content)
    with wave.open(buf, "rb") as wf:
        # 0.001s * 16000 = 16 frames → upsampled to 48 at 48kHz
        assert wf.getnframes() == 48


def test_get_audio_clip_middle_section(client, settings):
    """Test extracting from the middle of a file."""
    _create_recording_with_audio(settings, duration=3.0)

    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=1.0&end=2.0")
    assert response.status_code == 200

    import io

    buf = io.BytesIO(response.content)
    with wave.open(buf, "rb") as wf:
        # 1 second at 48kHz
        assert wf.getnframes() == OUTPUT_RATE


def test_get_audio_clip_negative_start_rejected(client, settings):
    """Test that negative start is rejected by FastAPI validation."""
    _create_recording_with_audio(settings)

    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=-1&end=1")
    assert response.status_code == 422


def test_get_audio_clip_duration_preserved(client, settings):
    """Test that clip duration is preserved through downmix + upsample."""
    import io

    _create_recording_with_audio(settings, duration=5.0)

    response = client.get("/api/v1/recordings/rec-001/audio/clip?start=1.5&end=3.5")
    assert response.status_code == 200

    buf = io.BytesIO(response.content)
    with wave.open(buf, "rb") as wf:
        duration = wf.getnframes() / wf.getframerate()
        assert abs(duration - 2.0) < 0.001
