"""Audio recorder for harkd daemon with full platform support.

Always records in stereo (L=mic, R=speaker) with independent mute flags
for each input. Supports mid-recording toggle of mic/speaker inputs.
"""

import logging
import os
import queue
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from harkd.audio.sources import (
    AudioSourceInfo,
    find_loopback_device,
    find_microphone,
    get_loopback_instructions,
)
from harkd.exceptions import NoLoopbackDeviceError, NoMicrophoneError

__all__ = ["AudioRecorder"]

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Audio recorder with full platform support.

    Always records in stereo (L=mic, R=speaker). Each input can be
    independently muted/unmuted during recording. Muted inputs produce
    silence frames (streams stay open).
    """

    def __init__(
        self,
        output_path: Path,
        mic_enabled: bool = True,
        speaker_enabled: bool = True,
        sample_rate: int = 16000,
        level_callback: Callable[[float, float], None] | None = None,
    ):
        """Initialize audio recorder.

        Args:
            output_path: Path to save the recording WAV file
            mic_enabled: Whether microphone input is initially enabled
            speaker_enabled: Whether speaker input is initially enabled
            sample_rate: Sample rate in Hz (default: 16000 for Whisper)
            level_callback: Optional callback for real-time audio levels (mic_level, speaker_level)

        Raises:
            TypeError: If output_path is not a Path
        """
        # Type validation
        if not isinstance(output_path, Path):
            try:
                output_path = Path(output_path)
            except (TypeError, ValueError) as e:
                raise TypeError(f"output_path must be a Path or path string: {e}") from e

        # Validate sample rate
        valid_rates = (8000, 11025, 16000, 22050, 44100, 48000)
        if sample_rate not in valid_rates:
            logger.warning(f"Unusual sample rate: {sample_rate}. Common values: {valid_rates}")

        self.output_path = output_path
        self.sample_rate = sample_rate
        self.level_callback = level_callback

        # Mute state (toggled mid-recording)
        self._mic_muted: bool = not mic_enabled
        self._speaker_muted: bool = not speaker_enabled

        # Per-input levels
        self._mic_level: float = 0.0
        self._speaker_level: float = 0.0

        # Device availability (set during start)
        self._mic_available: bool = False
        self._speaker_available: bool = False

        self._recording = False
        self._stop_event = threading.Event()
        self._mic_stream: sd.InputStream | None = None
        self._speaker_stream: sd.InputStream | None = None
        self._soundfile: sf.SoundFile | None = None
        self._lock = threading.Lock()
        self._interleave_thread: threading.Thread | None = None

        # Buffers to interleave mic and speaker
        self._mic_buffer: list[np.ndarray] = []
        self._speaker_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()

        # Speaker device may use a different native sample rate (e.g. 48kHz)
        self._speaker_native_rate: int | None = None

        # PulseAudio source routing (Linux): set PULSE_SOURCE before opening stream
        self._original_pulse_source: str | None = None
        self._pulse_source_set = False

        # Write queue: audio callbacks put data here, writer thread writes to disk.
        # This avoids doing file I/O in the PortAudio callback (which runs in a
        # real-time C thread and can cause segfaults with concurrent cffi calls).
        self._write_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._writer_thread: threading.Thread | None = None

    def set_mic_enabled(self, enabled: bool) -> None:
        """Toggle microphone input on/off during recording."""
        self._mic_muted = not enabled

    def set_speaker_enabled(self, enabled: bool) -> None:
        """Toggle speaker input on/off during recording."""
        self._speaker_muted = not enabled

    @property
    def mic_level(self) -> float:
        """Current mic audio level (0-1)."""
        return self._mic_level

    @property
    def speaker_level(self) -> float:
        """Current speaker audio level (0-1)."""
        return self._speaker_level

    def start(self) -> None:
        """Start recording.

        Always opens both mic and speaker streams in stereo mode.
        If a device for an enabled input is unavailable, raises an error.
        If a device for a disabled input is unavailable, proceeds but marks
        that input as permanently unavailable.

        Raises:
            RuntimeError: If already recording
            NoMicrophoneError: If mic is enabled and no microphone found
            NoLoopbackDeviceError: If speaker is enabled and no loopback device found
        """
        mic_enabled = not self._mic_muted
        speaker_enabled = not self._speaker_muted

        logger.info(
            f"Starting recording: mic={mic_enabled}, speaker={speaker_enabled}, "
            f"sample_rate={self.sample_rate}, output={self.output_path}"
        )

        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording")

            self._recording = True
            self._stop_event.clear()

            # Find audio sources
            mic_source = find_microphone()
            if mic_source is not None:
                self._mic_available = True
                logger.debug(f"Using microphone: {mic_source.name}")
            else:
                self._mic_available = False
                if mic_enabled:
                    self._recording = False
                    try:
                        devices = sd.query_devices()
                        input_devices = [
                            f"  [{i}] {d['name']}"
                            for i, d in enumerate(devices)
                            if d["max_input_channels"] > 0
                        ]
                        if input_devices:
                            available = "\n\nAvailable input devices:\n" + "\n".join(input_devices)
                        else:
                            available = "\n\nNo input devices detected."
                    except Exception:
                        available = ""
                    message = f"No microphone device found.{available}"
                    logger.error(message)
                    raise NoMicrophoneError(message)
                logger.debug("No microphone found (disabled, proceeding)")

            speaker_source = find_loopback_device()
            if speaker_source is not None:
                self._speaker_available = True
                logger.debug(f"Using loopback device: {speaker_source.name}")
            else:
                self._speaker_available = False
                if speaker_enabled:
                    self._recording = False
                    instructions = get_loopback_instructions()
                    message = f"No loopback device found.\n\n{instructions}"
                    logger.error(message)
                    raise NoLoopbackDeviceError(message)
                logger.debug("No loopback found (disabled, proceeding)")

            # Always stereo output
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                self._soundfile = sf.SoundFile(
                    self.output_path,
                    mode="w",
                    samplerate=self.sample_rate,
                    channels=2,
                    format="WAV",
                    subtype="PCM_16",
                )
                self._start_writer_thread()
                self._start_both(mic_source, speaker_source)
            except Exception:
                if self._soundfile:
                    self._soundfile.close()
                    self._soundfile = None
                self._restore_pulse_source()
                self._recording = False
                raise

    def _set_pulse_source(self, speaker_source: AudioSourceInfo) -> None:
        """Set PULSE_SOURCE env var to route PulseAudio monitor to the pulse device.

        On Linux, PortAudio's ALSA backend can't see PulseAudio monitors directly.
        We use the 'pulse' sounddevice and set PULSE_SOURCE to tell PulseAudio
        which monitor source to capture from.
        """
        if speaker_source.pulse_source:
            self._original_pulse_source = os.environ.get("PULSE_SOURCE")
            os.environ["PULSE_SOURCE"] = speaker_source.pulse_source
            self._pulse_source_set = True
            logger.debug(f"Set PULSE_SOURCE={speaker_source.pulse_source}")

    def _restore_pulse_source(self) -> None:
        """Restore original PULSE_SOURCE env var."""
        if self._pulse_source_set:
            if self._original_pulse_source is not None:
                os.environ["PULSE_SOURCE"] = self._original_pulse_source
            else:
                os.environ.pop("PULSE_SOURCE", None)
            self._pulse_source_set = False
            logger.debug("Restored PULSE_SOURCE")

    def _start_both(
        self,
        mic_source: AudioSourceInfo | None,
        speaker_source: AudioSourceInfo | None,
    ) -> None:
        """Start dual-stream stereo recording.

        Either source may be None if that device is unavailable (and disabled).
        """
        self._mic_buffer = []
        self._speaker_buffer = []

        # IMPORTANT: Open mic stream BEFORE setting PULSE_SOURCE.
        # PULSE_SOURCE redirects PulseAudio capture globally, so if it's set
        # when the mic stream opens, the mic will capture from the speaker
        # monitor instead of the actual microphone.
        if mic_source is not None:
            self._mic_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                device=mic_source.device_index,
                callback=self._mic_dual_callback,
            )
            self._mic_stream.start()

        # Now set PULSE_SOURCE for the speaker monitor and open speaker stream
        if speaker_source is not None:
            self._set_pulse_source(speaker_source)

            speaker_rate = int(speaker_source.sample_rate)
            if speaker_rate != self.sample_rate:
                logger.info(
                    f"Speaker device native rate is {speaker_rate}Hz, "
                    f"will resample to {self.sample_rate}Hz in interleave thread"
                )
            self._speaker_native_rate = speaker_rate

            self._speaker_stream = sd.InputStream(
                samplerate=speaker_rate,
                channels=1,
                device=speaker_source.device_index,
                callback=self._speaker_dual_callback,
            )
            self._speaker_stream.start()

        # Start interleaving thread
        self._interleave_thread = threading.Thread(target=self._interleave_audio)
        self._interleave_thread.daemon = True
        self._interleave_thread.start()

    def stop(self) -> None:
        """Stop recording and close file."""
        logger.info("Stopping recording")

        with self._lock:
            if not self._recording:
                logger.debug("Stop called but not recording")
                return

            self._recording = False
            self._stop_event.set()

        # Wait for interleaving thread to finish (outside lock to avoid deadlock)
        if self._interleave_thread and self._interleave_thread.is_alive():
            logger.debug("Waiting for interleaving thread to finish")
            self._interleave_thread.join(timeout=2.0)
            if self._interleave_thread.is_alive():
                logger.warning("Interleaving thread did not terminate within timeout")

        with self._lock:
            # Stop and close streams (stops callbacks from producing more data)
            if self._mic_stream:
                self._mic_stream.stop()
                self._mic_stream.close()
                self._mic_stream = None
                logger.debug("Microphone stream stopped")

            if self._speaker_stream:
                self._speaker_stream.stop()
                self._speaker_stream.close()
                self._speaker_stream = None
                logger.debug("Speaker stream stopped")

        # Stop writer thread (send sentinel, wait for it to drain)
        if self._writer_thread and self._writer_thread.is_alive():
            self._write_queue.put(None)  # Sentinel
            self._writer_thread.join(timeout=5.0)
            if self._writer_thread.is_alive():
                logger.warning("Writer thread did not terminate within timeout")

        with self._lock:
            # Close sound file
            if self._soundfile:
                self._soundfile.close()
                self._soundfile = None
                logger.debug(f"Audio file closed: {self.output_path}")

            self._interleave_thread = None
            self._writer_thread = None

            # Restore PULSE_SOURCE env var
            self._restore_pulse_source()

        logger.info("Recording stopped successfully")

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically stop recording."""
        self.stop()
        return False

    def _resample(self, audio: np.ndarray, orig_rate: int) -> np.ndarray:
        """Resample audio from orig_rate to target sample_rate.

        Args:
            audio: Audio data (samples, 1) or (samples,)
            orig_rate: Original sample rate

        Returns:
            Resampled audio at self.sample_rate
        """
        if orig_rate == self.sample_rate:
            return audio
        import librosa

        # librosa expects (samples,) shape
        flat = audio.flatten()
        resampled = librosa.resample(flat, orig_sr=orig_rate, target_sr=self.sample_rate)
        return resampled.reshape(-1, 1)

    def _start_writer_thread(self) -> None:
        """Start the background writer thread that drains the write queue."""
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        """Background thread that writes audio data from the queue to disk."""
        while True:
            try:
                data = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if data is None:  # Sentinel to stop
                break

            try:
                with self._lock:
                    if self._soundfile:
                        self._soundfile.write(data)
            except Exception as e:
                logger.error(f"Error writing audio data: {e}", exc_info=True)

    def _mic_dual_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for mic in dual-stream mode. Applies mute if needed."""
        if status:
            logger.warning(f"Microphone dual-stream callback status: {status}")

        try:
            data = np.zeros_like(indata) if self._mic_muted else indata.copy()

            # Compute level from the data we'll actually use
            rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
            self._mic_level = float(min(1.0, rms * 2.0))

            with self._buffer_lock:
                self._mic_buffer.append(data)
        except Exception as e:
            logger.error(f"Error in mic dual-stream callback: {e}", exc_info=True)

    def _speaker_dual_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for speaker in dual-stream mode. Applies mute if needed."""
        if status:
            logger.warning(f"Speaker dual-stream callback status: {status}")

        try:
            data = np.zeros_like(indata) if self._speaker_muted else indata.copy()

            # Compute level from the data we'll actually use
            rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
            self._speaker_level = float(min(1.0, rms * 2.0))

            with self._buffer_lock:
                self._speaker_buffer.append(data)
        except Exception as e:
            logger.error(f"Error in speaker dual-stream callback: {e}", exc_info=True)

    def _interleave_audio(self) -> None:
        """Interleave mic and speaker audio into stereo."""
        # Continue processing while recording OR while buffers have data
        while not self._stop_event.is_set() or self._mic_buffer or self._speaker_buffer:
            chunks_to_write = []

            with self._buffer_lock:
                # Process available chunks
                min_chunks = min(len(self._mic_buffer), len(self._speaker_buffer))

                for _ in range(min_chunks):
                    mic_chunk = self._mic_buffer.pop(0)
                    speaker_chunk = self._speaker_buffer.pop(0)

                    # Resample speaker if at different rate
                    if self._speaker_native_rate and self._speaker_native_rate != self.sample_rate:
                        speaker_chunk = self._resample(speaker_chunk, self._speaker_native_rate)

                    # Ensure same length (may differ after resampling)
                    min_len = min(len(mic_chunk), len(speaker_chunk))
                    mic_chunk = mic_chunk[:min_len]
                    speaker_chunk = speaker_chunk[:min_len]

                    # Create stereo: L=mic, R=speaker
                    stereo = np.column_stack((mic_chunk, speaker_chunk))
                    chunks_to_write.append(stereo)

            # Send chunks to writer thread via queue
            for stereo in chunks_to_write:
                try:
                    self._write_queue.put_nowait(stereo)
                except (ValueError, AttributeError):
                    # Queue or file was closed, stop writing
                    break

            # Report per-input levels
            if self.level_callback and chunks_to_write:
                try:
                    self.level_callback(self._mic_level, self._speaker_level)
                except Exception as e:
                    logger.debug(f"Error in level callback: {e}")

            # Use event wait for cleaner shutdown signaling
            self._stop_event.wait(timeout=0.01)

    def _report_level(self, audio_data: np.ndarray) -> None:
        """Calculate and report audio level (legacy helper for tests)."""
        if self.level_callback:
            try:
                # Calculate RMS level
                rms = np.sqrt(np.mean(audio_data**2))
                # Normalize to 0-1 range (assuming max level is around 0.5)
                # Convert to native float to avoid numpy.float32 serialization issues
                level = float(min(1.0, rms * 2.0))
                self.level_callback(level, level)
            except Exception as e:
                logger.debug(f"Error in level callback: {e}")
