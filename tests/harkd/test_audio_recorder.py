"""Tests for audio recorder."""

import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from harkd.audio.recorder import AudioRecorder
from harkd.audio.sources import AudioSourceInfo


@pytest.fixture
def temp_output(tmp_path):
    """Create temporary output path."""
    return tmp_path / "test_recording.wav"


def test_recorder_initialization(temp_output):
    """Test recorder can be initialized."""
    recorder = AudioRecorder(
        output_path=temp_output,
        sample_rate=16000,
    )

    assert recorder.output_path == temp_output
    assert recorder.sample_rate == 16000
    assert recorder.is_recording() is False


def test_recorder_with_level_callback(temp_output):
    """Test recorder with level callback."""
    callback = MagicMock()
    recorder = AudioRecorder(
        output_path=temp_output,
        level_callback=callback,
    )

    assert recorder.level_callback == callback


@patch("harkd.audio.recorder.find_loopback_device")
@patch("harkd.audio.recorder.find_microphone")
@patch("harkd.audio.recorder.sd.InputStream")
@patch("harkd.audio.recorder.sf.SoundFile")
def test_start_recording(
    mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
):
    """Test starting a recording."""
    # Mock microphone device
    mock_find_mic.return_value = AudioSourceInfo(
        device_index=0,
        name="Test Mic",
        channels=1,
        sample_rate=48000,
        is_loopback=False,
    )

    # Mock loopback device
    mock_find_loopback.return_value = AudioSourceInfo(
        device_index=1,
        name="Test Loopback",
        channels=2,
        sample_rate=48000,
        is_loopback=True,
    )

    recorder = AudioRecorder(output_path=temp_output)
    recorder.start()

    assert recorder.is_recording() is True
    mock_soundfile.assert_called_once()

    # Verify audio data is queued for writing via callback
    audio_data = np.random.random((1024, 1)).astype(np.float32) * 0.1
    recorder._mic_dual_callback(audio_data, 1024, {}, MagicMock())
    # Data goes to the mic buffer, not directly to soundfile
    assert len(recorder._mic_buffer) > 0


@patch("harkd.audio.recorder.find_loopback_device")
@patch("harkd.audio.recorder.find_microphone")
@patch("harkd.audio.recorder.sd.InputStream")
@patch("harkd.audio.recorder.sf.SoundFile")
def test_start_recording_already_recording(
    mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
):
    """Test that starting when already recording raises error."""
    mock_find_mic.return_value = AudioSourceInfo(
        device_index=0,
        name="Test Mic",
        channels=1,
        sample_rate=48000,
        is_loopback=False,
    )
    mock_find_loopback.return_value = AudioSourceInfo(
        device_index=1,
        name="Test Loopback",
        channels=2,
        sample_rate=48000,
        is_loopback=True,
    )

    recorder = AudioRecorder(output_path=temp_output)
    recorder.start()

    with pytest.raises(RuntimeError) as exc_info:
        recorder.start()

    assert "Already recording" in str(exc_info.value)


@patch("harkd.audio.recorder.find_loopback_device")
@patch("harkd.audio.recorder.find_microphone")
@patch("harkd.audio.recorder.sd.InputStream")
@patch("harkd.audio.recorder.sf.SoundFile")
def test_stop_recording(
    mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
):
    """Test stopping a recording."""
    mock_find_mic.return_value = AudioSourceInfo(
        device_index=0,
        name="Test Mic",
        channels=1,
        sample_rate=48000,
        is_loopback=False,
    )
    mock_find_loopback.return_value = AudioSourceInfo(
        device_index=1,
        name="Test Loopback",
        channels=2,
        sample_rate=48000,
        is_loopback=True,
    )

    recorder = AudioRecorder(output_path=temp_output)
    recorder.start()
    recorder.stop()

    assert recorder.is_recording() is False


@patch("harkd.audio.recorder.sd.InputStream")
@patch("harkd.audio.recorder.sf.SoundFile")
def test_stop_when_not_recording(mock_soundfile, mock_stream, temp_output):
    """Test that stopping when not recording is safe."""
    recorder = AudioRecorder(output_path=temp_output)
    recorder.stop()  # Should not raise

    assert recorder.is_recording() is False


@patch("harkd.audio.recorder.find_loopback_device")
@patch("harkd.audio.recorder.find_microphone")
@patch("harkd.audio.recorder.sd.InputStream")
@patch("harkd.audio.recorder.sf.SoundFile")
def test_audio_callback_calculates_level(
    mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
):
    """Test that audio callback calculates and reports level."""
    mock_find_mic.return_value = AudioSourceInfo(
        device_index=0,
        name="Test Mic",
        channels=1,
        sample_rate=48000,
        is_loopback=False,
    )
    mock_find_loopback.return_value = AudioSourceInfo(
        device_index=1,
        name="Test Loopback",
        channels=2,
        sample_rate=48000,
        is_loopback=True,
    )

    level_callback = MagicMock()
    recorder = AudioRecorder(output_path=temp_output, level_callback=level_callback)
    recorder.start()

    # Simulate audio data through the mic callback
    audio_data = np.random.random((1024, 1)).astype(np.float32) * 0.1
    recorder._mic_dual_callback(audio_data, 1024, {}, MagicMock())

    # mic_level should have been updated
    assert 0 <= recorder.mic_level <= 1


@patch("harkd.audio.recorder.find_loopback_device")
@patch("harkd.audio.recorder.find_microphone")
@patch("harkd.audio.recorder.sd.InputStream")
@patch("harkd.audio.recorder.sf.SoundFile")
def test_recorder_always_stereo(
    mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
):
    """Test that recorder always uses 2 channels (stereo)."""
    mock_find_mic.return_value = AudioSourceInfo(
        device_index=0,
        name="Test Mic",
        channels=1,
        sample_rate=48000,
        is_loopback=False,
    )
    mock_find_loopback.return_value = AudioSourceInfo(
        device_index=1,
        name="Test Loopback",
        channels=2,
        sample_rate=48000,
        is_loopback=True,
    )

    recorder = AudioRecorder(output_path=temp_output)
    recorder.start()

    # Check that SoundFile was created with 2 channels (always stereo)
    call_kwargs = mock_soundfile.call_args[1]
    assert call_kwargs["channels"] == 2


@patch("harkd.audio.recorder.find_loopback_device")
@patch("harkd.audio.recorder.find_microphone")
@patch("harkd.audio.recorder.sd.InputStream")
@patch("harkd.audio.recorder.sf.SoundFile")
def test_recorder_creates_output_directory(
    mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, tmp_path
):
    """Test that recorder creates output directory if needed."""
    mock_find_mic.return_value = AudioSourceInfo(
        device_index=0,
        name="Test Mic",
        channels=1,
        sample_rate=48000,
        is_loopback=False,
    )
    mock_find_loopback.return_value = AudioSourceInfo(
        device_index=1,
        name="Test Loopback",
        channels=2,
        sample_rate=48000,
        is_loopback=True,
    )

    output_path = tmp_path / "subdir" / "recording.wav"
    recorder = AudioRecorder(output_path=output_path)
    recorder.start()

    assert output_path.parent.exists()


@patch("harkd.audio.recorder.find_loopback_device")
@patch("harkd.audio.recorder.find_microphone")
@patch("harkd.audio.recorder.sd.InputStream")
@patch("harkd.audio.recorder.sf.SoundFile")
def test_recorder_sample_rate(
    mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
):
    """Test that custom sample rate is used."""
    mock_find_mic.return_value = AudioSourceInfo(
        device_index=0,
        name="Test Mic",
        channels=1,
        sample_rate=48000,
        is_loopback=False,
    )
    mock_find_loopback.return_value = AudioSourceInfo(
        device_index=1,
        name="Test Loopback",
        channels=2,
        sample_rate=48000,
        is_loopback=True,
    )

    recorder = AudioRecorder(output_path=temp_output, sample_rate=48000)
    recorder.start()

    # Check that SoundFile was created with correct sample rate
    call_kwargs = mock_soundfile.call_args[1]
    assert call_kwargs["samplerate"] == 48000


class TestReportLevel:
    """Tests for _report_level method."""

    def test_level_callback_exception_does_not_crash(self, temp_output):
        """Test that callback exceptions don't crash the recorder."""

        def failing_callback(mic_level, speaker_level):
            raise ValueError("Callback error")

        recorder = AudioRecorder(output_path=temp_output, level_callback=failing_callback)

        # Should not raise — error is caught and logged
        audio_data = np.ones((1024, 1), dtype=np.float32) * 0.1
        recorder._report_level(audio_data)

    def test_level_rms_calculation(self, temp_output):
        """Test that RMS level is calculated correctly."""
        levels = []

        def capture_level(mic_level, speaker_level):
            levels.append(mic_level)

        recorder = AudioRecorder(output_path=temp_output, level_callback=capture_level)

        # Silent audio should give level ~0
        silent = np.zeros((1024, 1), dtype=np.float32)
        recorder._report_level(silent)
        assert levels[-1] == 0.0

        # Known amplitude should give predictable RMS
        # RMS of constant 0.25 = 0.25, level = min(1.0, 0.25 * 2) = 0.5
        constant = np.full((1024, 1), 0.25, dtype=np.float32)
        recorder._report_level(constant)
        assert abs(levels[-1] - 0.5) < 0.01

    def test_level_clamped_to_one(self, temp_output):
        """Test that level is clamped to 1.0 max."""
        levels = []

        def capture_level(mic_level, speaker_level):
            levels.append(mic_level)

        recorder = AudioRecorder(output_path=temp_output, level_callback=capture_level)

        # Very loud audio (RMS > 0.5) should be clamped to 1.0
        loud = np.ones((1024, 1), dtype=np.float32)
        recorder._report_level(loud)
        assert levels[-1] == 1.0

    def test_no_callback_does_nothing(self, temp_output):
        """Test that _report_level is safe when no callback is set."""
        recorder = AudioRecorder(output_path=temp_output, level_callback=None)

        # Should not raise
        audio_data = np.ones((1024, 1), dtype=np.float32)
        recorder._report_level(audio_data)


# ============================================================================
# Regression tests for bugs found during real-world testing
# ============================================================================


class TestCallbackNeverWritesSoundfileDirectly:
    """Regression: PortAudio segfault from file I/O in audio callback.

    Audio callbacks run on PortAudio's C thread. Calling soundfile.write()
    (which uses cffi/libsndfile) from that thread causes segfaults when running
    inside uvicorn/FastAPI. All callbacks must enqueue data for the writer thread.
    """

    @patch("harkd.audio.recorder.find_loopback_device")
    @patch("harkd.audio.recorder.find_microphone")
    @patch("harkd.audio.recorder.sd.InputStream")
    @patch("harkd.audio.recorder.sf.SoundFile")
    def test_mic_callback_enqueues_not_writes(
        self, mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
    ):
        """Mic callback must put data on buffer, never call soundfile.write."""
        mock_find_mic.return_value = AudioSourceInfo(
            device_index=0,
            name="Mic",
            channels=1,
            sample_rate=16000,
            is_loopback=False,
        )
        mock_find_loopback.return_value = AudioSourceInfo(
            device_index=1,
            name="Monitor",
            channels=2,
            sample_rate=48000,
            is_loopback=True,
        )
        recorder = AudioRecorder(output_path=temp_output)
        recorder.start()

        sf_instance = mock_soundfile.return_value
        audio = np.random.random((1024, 1)).astype(np.float32)
        recorder._mic_dual_callback(audio, 1024, {}, MagicMock())

        sf_instance.write.assert_not_called()
        assert len(recorder._mic_buffer) > 0

    @patch("harkd.audio.recorder.find_loopback_device")
    @patch("harkd.audio.recorder.find_microphone")
    @patch("harkd.audio.recorder.sd.InputStream")
    @patch("harkd.audio.recorder.sf.SoundFile")
    def test_speaker_callback_enqueues_not_writes(
        self, mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
    ):
        """Speaker callback must put data on buffer, never call soundfile.write."""
        mock_find_mic.return_value = AudioSourceInfo(
            device_index=0,
            name="Mic",
            channels=1,
            sample_rate=16000,
            is_loopback=False,
        )
        mock_find_loopback.return_value = AudioSourceInfo(
            device_index=1,
            name="Monitor",
            channels=1,
            sample_rate=48000,
            is_loopback=True,
        )
        recorder = AudioRecorder(output_path=temp_output)
        recorder.start()

        sf_instance = mock_soundfile.return_value
        audio = np.random.random((1024, 1)).astype(np.float32)
        recorder._speaker_dual_callback(audio, 1024, {}, MagicMock())

        sf_instance.write.assert_not_called()
        assert len(recorder._speaker_buffer) > 0

    @patch("harkd.audio.recorder.find_loopback_device")
    @patch("harkd.audio.recorder.find_microphone")
    @patch("harkd.audio.recorder.sd.InputStream")
    @patch("harkd.audio.recorder.sf.SoundFile")
    def test_dual_callbacks_enqueue_not_write(
        self,
        mock_soundfile,
        mock_stream,
        mock_find_mic,
        mock_find_loopback,
        temp_output,
    ):
        """Both-mode callbacks must buffer data, never call soundfile.write."""
        mock_find_mic.return_value = AudioSourceInfo(
            device_index=0,
            name="Mic",
            channels=1,
            sample_rate=16000,
            is_loopback=False,
        )
        mock_find_loopback.return_value = AudioSourceInfo(
            device_index=1,
            name="Monitor",
            channels=1,
            sample_rate=48000,
            is_loopback=True,
        )
        recorder = AudioRecorder(output_path=temp_output)
        recorder.start()

        sf_instance = mock_soundfile.return_value
        audio = np.random.random((1024, 1)).astype(np.float32)
        recorder._mic_dual_callback(audio, 1024, {}, MagicMock())
        recorder._speaker_dual_callback(audio, 1024, {}, MagicMock())

        sf_instance.write.assert_not_called()


class TestWriterThreadDrainsQueue:
    """Regression: verify the writer thread actually writes queued audio to disk."""

    @patch("harkd.audio.recorder.find_loopback_device")
    @patch("harkd.audio.recorder.find_microphone")
    @patch("harkd.audio.recorder.sd.InputStream")
    @patch("harkd.audio.recorder.sf.SoundFile")
    def test_writer_thread_writes_queued_data(
        self, mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
    ):
        """Writer thread must drain the queue and call soundfile.write."""
        mock_find_mic.return_value = AudioSourceInfo(
            device_index=0,
            name="Mic",
            channels=1,
            sample_rate=16000,
            is_loopback=False,
        )
        mock_find_loopback.return_value = AudioSourceInfo(
            device_index=1,
            name="Monitor",
            channels=1,
            sample_rate=16000,
            is_loopback=True,
        )
        recorder = AudioRecorder(output_path=temp_output)
        recorder.start()

        sf_instance = mock_soundfile.return_value
        audio = np.random.random((1024, 1)).astype(np.float32)

        # Simulate 3 callbacks on both mic and speaker to produce interleaved output
        for _ in range(3):
            recorder._mic_dual_callback(audio, 1024, {}, MagicMock())
            recorder._speaker_dual_callback(audio, 1024, {}, MagicMock())

        # Give interleave + writer threads time to process
        time.sleep(0.5)

        assert sf_instance.write.call_count == 3


class TestPulseSourceEnvVar:
    """Regression: PULSE_SOURCE env var management for PulseAudio monitors.

    On Linux, PortAudio's ALSA backend can't see PulseAudio monitor sources.
    We route audio via PULSE_SOURCE + the 'pulse' device. The env var must be
    set before opening the stream and restored after stopping.
    """

    def test_set_pulse_source_sets_env_var(self, temp_output):
        """_set_pulse_source must set PULSE_SOURCE when pulse_source is provided."""
        recorder = AudioRecorder(output_path=temp_output)
        source = AudioSourceInfo(
            device_index=17,
            name="monitor.source",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source="alsa_output.some_device.monitor",
        )

        old_val = os.environ.get("PULSE_SOURCE")
        try:
            recorder._set_pulse_source(source)
            assert os.environ["PULSE_SOURCE"] == "alsa_output.some_device.monitor"
            assert recorder._pulse_source_set is True
        finally:
            # Clean up
            recorder._restore_pulse_source()
            assert os.environ.get("PULSE_SOURCE") == old_val

    def test_set_pulse_source_noop_without_pulse_source(self, temp_output):
        """_set_pulse_source must be a no-op when pulse_source is None."""
        recorder = AudioRecorder(output_path=temp_output)
        source = AudioSourceInfo(
            device_index=5,
            name="BlackHole 2ch",
            channels=2,
            sample_rate=48000,
            is_loopback=True,
            pulse_source=None,
        )

        old_val = os.environ.get("PULSE_SOURCE")
        recorder._set_pulse_source(source)
        assert os.environ.get("PULSE_SOURCE") == old_val
        assert recorder._pulse_source_set is False

    def test_restore_removes_env_var_if_was_unset(self, temp_output):
        """_restore_pulse_source must remove PULSE_SOURCE if it wasn't set before."""
        recorder = AudioRecorder(output_path=temp_output)
        source = AudioSourceInfo(
            device_index=17,
            name="monitor",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source="some.monitor",
        )

        # Ensure PULSE_SOURCE is not set
        old_val = os.environ.pop("PULSE_SOURCE", None)
        try:
            recorder._set_pulse_source(source)
            assert "PULSE_SOURCE" in os.environ

            recorder._restore_pulse_source()
            assert "PULSE_SOURCE" not in os.environ
        finally:
            if old_val is not None:
                os.environ["PULSE_SOURCE"] = old_val

    def test_restore_reverts_to_original_value(self, temp_output):
        """_restore_pulse_source must revert to the original PULSE_SOURCE value."""
        recorder = AudioRecorder(output_path=temp_output)
        source = AudioSourceInfo(
            device_index=17,
            name="monitor",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source="new.monitor",
        )

        old_val = os.environ.get("PULSE_SOURCE")
        try:
            os.environ["PULSE_SOURCE"] = "original.monitor"
            recorder._set_pulse_source(source)
            assert os.environ["PULSE_SOURCE"] == "new.monitor"

            recorder._restore_pulse_source()
            assert os.environ["PULSE_SOURCE"] == "original.monitor"
        finally:
            if old_val is not None:
                os.environ["PULSE_SOURCE"] = old_val
            else:
                os.environ.pop("PULSE_SOURCE", None)

    @patch("harkd.audio.recorder.find_loopback_device")
    @patch("harkd.audio.recorder.find_microphone")
    @patch("harkd.audio.recorder.sd.InputStream")
    @patch("harkd.audio.recorder.sf.SoundFile")
    def test_pulse_source_set_before_stream_opened(
        self, mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
    ):
        """PULSE_SOURCE must be set before sd.InputStream is created."""
        pulse_source_at_stream_open = []

        def capture_pulse_source(**kwargs):
            pulse_source_at_stream_open.append(os.environ.get("PULSE_SOURCE"))
            return MagicMock()

        mock_stream.side_effect = capture_pulse_source
        mock_find_mic.return_value = None  # No mic
        mock_find_loopback.return_value = AudioSourceInfo(
            device_index=17,
            name="monitor",
            channels=2,
            sample_rate=44100,
            is_loopback=True,
            pulse_source="alsa_output.device.monitor",
        )

        old_val = os.environ.get("PULSE_SOURCE")
        try:
            recorder = AudioRecorder(output_path=temp_output, mic_enabled=False)
            recorder.start()
            assert pulse_source_at_stream_open[0] == "alsa_output.device.monitor"
        finally:
            # Ensure cleanup
            recorder.stop()
            if old_val is not None:
                os.environ["PULSE_SOURCE"] = old_val
            else:
                os.environ.pop("PULSE_SOURCE", None)


class TestSpeakerNativeSampleRate:
    """Regression: speaker devices may not support 16kHz.

    When the speaker device's native rate differs from the target rate (16kHz),
    the stream must use the device's native rate. Resampling happens in the
    interleave thread.
    """

    @patch("harkd.audio.recorder.find_loopback_device")
    @patch("harkd.audio.recorder.find_microphone")
    @patch("harkd.audio.recorder.sd.InputStream")
    @patch("harkd.audio.recorder.sf.SoundFile")
    def test_speaker_uses_native_rate(
        self, mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
    ):
        """Speaker stream must open at device native rate, not target rate."""
        mock_find_mic.return_value = None  # No mic
        mock_find_loopback.return_value = AudioSourceInfo(
            device_index=1,
            name="Monitor",
            channels=1,
            sample_rate=48000,
            is_loopback=True,
        )

        recorder = AudioRecorder(output_path=temp_output, mic_enabled=False, sample_rate=16000)
        recorder.start()

        # One InputStream call (speaker only since mic is unavailable)
        # Find the call that opened the speaker stream (48kHz)
        found_48k = False
        for call in mock_stream.call_args_list:
            if call[1].get("samplerate") == 48000:
                found_48k = True
        assert found_48k, "Speaker stream should be opened at 48kHz native rate"

        assert recorder._speaker_native_rate == 48000

    @patch("harkd.audio.recorder.find_loopback_device")
    @patch("harkd.audio.recorder.find_microphone")
    @patch("harkd.audio.recorder.sd.InputStream")
    @patch("harkd.audio.recorder.sf.SoundFile")
    def test_speaker_matching_rate_uses_target(
        self, mock_soundfile, mock_stream, mock_find_mic, mock_find_loopback, temp_output
    ):
        """When device native rate matches target, use the target rate."""
        mock_find_mic.return_value = None  # No mic
        mock_find_loopback.return_value = AudioSourceInfo(
            device_index=1,
            name="Monitor",
            channels=1,
            sample_rate=16000,
            is_loopback=True,
        )

        recorder = AudioRecorder(output_path=temp_output, mic_enabled=False, sample_rate=16000)
        recorder.start()

        # Find the speaker stream call
        found_16k = False
        for call in mock_stream.call_args_list:
            if call[1].get("samplerate") == 16000:
                found_16k = True
        assert found_16k


class TestMuteToggle:
    """Tests for mid-recording mute/unmute toggle."""

    def test_set_mic_enabled(self, temp_output):
        """Test toggling mic enabled state."""
        recorder = AudioRecorder(output_path=temp_output, mic_enabled=True)
        assert not recorder._mic_muted

        recorder.set_mic_enabled(False)
        assert recorder._mic_muted

        recorder.set_mic_enabled(True)
        assert not recorder._mic_muted

    def test_set_speaker_enabled(self, temp_output):
        """Test toggling speaker enabled state."""
        recorder = AudioRecorder(output_path=temp_output, speaker_enabled=True)
        assert not recorder._speaker_muted

        recorder.set_speaker_enabled(False)
        assert recorder._speaker_muted

        recorder.set_speaker_enabled(True)
        assert not recorder._speaker_muted

    def test_muted_mic_produces_silence(self, temp_output):
        """Test that muted mic callback produces zero data."""
        recorder = AudioRecorder(output_path=temp_output, mic_enabled=False)

        audio_data = np.ones((1024, 1), dtype=np.float32) * 0.5
        recorder._mic_dual_callback(audio_data, 1024, {}, MagicMock())

        # Buffer should contain zeros
        assert len(recorder._mic_buffer) == 1
        expected = np.zeros((1024, 1), dtype=np.float32)
        np.testing.assert_array_equal(recorder._mic_buffer[0], expected)

    def test_muted_speaker_produces_silence(self, temp_output):
        """Test that muted speaker callback produces zero data."""
        recorder = AudioRecorder(output_path=temp_output, speaker_enabled=False)

        audio_data = np.ones((1024, 1), dtype=np.float32) * 0.5
        recorder._speaker_dual_callback(audio_data, 1024, {}, MagicMock())

        # Buffer should contain zeros
        assert len(recorder._speaker_buffer) == 1
        expected = np.zeros((1024, 1), dtype=np.float32)
        np.testing.assert_array_equal(recorder._speaker_buffer[0], expected)

    def test_mic_level_property(self, temp_output):
        """Test mic_level property returns current level."""
        recorder = AudioRecorder(output_path=temp_output)
        assert recorder.mic_level == 0.0

        audio_data = np.ones((1024, 1), dtype=np.float32) * 0.3
        recorder._mic_dual_callback(audio_data, 1024, {}, MagicMock())
        assert recorder.mic_level > 0.0

    def test_speaker_level_property(self, temp_output):
        """Test speaker_level property returns current level."""
        recorder = AudioRecorder(output_path=temp_output)
        assert recorder.speaker_level == 0.0

        audio_data = np.ones((1024, 1), dtype=np.float32) * 0.3
        recorder._speaker_dual_callback(audio_data, 1024, {}, MagicMock())
        assert recorder.speaker_level > 0.0


class TestLevelCallbackReturnsNativeFloat:
    """Regression: numpy.float32 caused Pydantic JSON serialization errors.

    The level callback must receive native Python floats, not numpy.float32,
    because the values are stored in Pydantic models and serialized to JSON.
    """

    def test_level_is_native_float_not_numpy(self, temp_output):
        """Level values must be native Python floats, not numpy.float32."""
        mic_levels: list[float] = []
        speaker_levels: list[float] = []

        def collect(m: float, s: float) -> None:
            mic_levels.append(m)
            speaker_levels.append(s)

        recorder = AudioRecorder(
            output_path=temp_output,
            level_callback=collect,
        )

        audio = np.random.random((1024, 1)).astype(np.float32) * 0.3
        recorder._report_level(audio)

        assert len(mic_levels) == 1
        assert type(mic_levels[0]) is float  # noqa: E721 — must be exact type, not isinstance
        assert type(speaker_levels[0]) is float  # noqa: E721

    def test_level_is_json_serializable(self, temp_output):
        """Level values must be JSON-serializable (native float, not numpy)."""
        import json

        mic_levels: list[float] = []
        speaker_levels: list[float] = []

        def collect(m: float, s: float) -> None:
            mic_levels.append(m)
            speaker_levels.append(s)

        recorder = AudioRecorder(
            output_path=temp_output,
            level_callback=collect,
        )

        audio = np.ones((1024, 1), dtype=np.float32) * 0.4
        recorder._report_level(audio)

        # This would raise TypeError if level is numpy.float32
        json.dumps({"mic_level": mic_levels[0], "speaker_level": speaker_levels[0]})
