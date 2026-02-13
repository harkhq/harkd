"""Transcriber using WhisperX for harkd daemon."""

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Fix for PyTorch 2.6 weights_only default change
# Must be set before torch is imported
os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

__all__ = [
    "WordSegment",
    "TranscriptionSegment",
    "TranscriptionResult",
    "Transcriber",
]

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    """A single word with timing information."""

    start: float
    end: float
    word: str


@dataclass
class TranscriptionSegment:
    """A single transcription segment."""

    start: float
    end: float
    text: str
    words: list[WordSegment] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    text: str
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float


class Transcriber:
    """WhisperX-based transcription engine for harkd daemon.

    Uses WhisperX for improved word-level timestamps through wav2vec2 alignment.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str | None = None,
        compute_type: str = "auto",
    ):
        """Initialize transcriber.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use ("cpu", "cuda", or "auto")
            language: Language code (e.g., "en") or None for auto-detection
            compute_type: Compute type ("int8", "float16", or "auto")
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.compute_type = compute_type
        self._model = None
        self._model_lock = threading.Lock()
        self._actual_device: str | None = None

    def close(self) -> None:
        """Release the loaded model and free memory."""
        with self._model_lock:
            self._model = None
            self._actual_device = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release model."""
        self.close()
        return False

    def _load_model(self):
        """Lazy-load the WhisperX model with thread safety."""
        if self._model is None:
            with self._model_lock:
                # Double-check pattern for thread safety
                if self._model is None:
                    logger.info(
                        f"Loading WhisperX model: {self.model_name} "
                        f"(device={self.device}, compute_type={self.compute_type})"
                    )

                    try:
                        import whisperx  # type: ignore
                    except ImportError as e:
                        logger.error("WhisperX not installed")
                        raise RuntimeError(
                            "WhisperX is required for transcription. "
                            "Install with: pip install whisperx"
                        ) from e

                    try:
                        # Auto-detect device if needed
                        device = self.device
                        if device == "auto":
                            try:
                                import torch

                                device = "cuda" if torch.cuda.is_available() else "cpu"
                            except ImportError:
                                device = "cpu"
                            logger.debug(f"Auto-detected device: {device}")

                        self._actual_device = device

                        # Auto-detect compute type if needed
                        compute_type = self.compute_type
                        if compute_type == "auto":
                            compute_type = "float16" if device == "cuda" else "int8"
                            logger.debug(f"Auto-detected compute type: {compute_type}")

                        self._model = whisperx.load_model(
                            self.model_name,
                            device=device,
                            compute_type=compute_type,
                        )
                        logger.info("WhisperX model loaded successfully")
                    except Exception as e:
                        logger.error(
                            f"Failed to load WhisperX model: {e}", exc_info=True
                        )
                        raise

    def transcribe(
        self,
        audio_path: Path,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file (WAV format)
            word_timestamps: Whether to include word-level timestamps

        Returns:
            Transcription result with text, segments, and metadata

        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(
            f"Transcribing audio: {audio_path} "
            f"(language={self.language or 'auto'}, word_timestamps={word_timestamps})"
        )

        self._load_model()

        try:
            import whisperx  # type: ignore

            # Load audio
            logger.debug("Loading audio file")
            audio = whisperx.load_audio(str(audio_path))

            # Transcribe with WhisperX
            logger.debug("Running transcription")
            if self._model is None:
                raise RuntimeError("Model not loaded")
            result = self._model.transcribe(
                audio, batch_size=16, language=self.language
            )
            detected_language = result.get("language", "unknown")
            logger.debug(f"Detected language: {detected_language}")

            # Align for word-level timestamps if requested
            if word_timestamps:
                logger.debug("Aligning for word-level timestamps")
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self._actual_device,
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self._actual_device,
                    return_char_alignments=False,
                )

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise

        # Convert segments to our format
        segments = []
        full_text_parts = []

        for segment in result.get("segments", []):
            # Convert words if available
            words = []
            if word_timestamps and "words" in segment:
                words = [
                    WordSegment(
                        start=word.get("start", 0.0),
                        end=word.get("end", 0.0),
                        word=word.get("word", ""),
                    )
                    for word in segment.get("words", [])
                ]

            segments.append(
                TranscriptionSegment(
                    start=segment.get("start", 0.0),
                    end=segment.get("end", 0.0),
                    text=segment.get("text", "").strip(),
                    words=words,
                )
            )
            full_text_parts.append(segment.get("text", "").strip())

        # Calculate duration from the segment with the latest end time
        duration = max((s.end for s in segments), default=0.0)

        # WhisperX doesn't expose language probability via its API
        language_probability = 0.0

        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"{duration:.1f}s duration, language={detected_language}"
        )

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            segments=segments,
            language=detected_language,
            language_probability=language_probability,
            duration=duration,
        )
