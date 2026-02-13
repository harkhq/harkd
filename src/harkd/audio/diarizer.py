"""Speaker diarization using WhisperX for harkd daemon.

Provides speaker identification and word-level speaker timestamps.
Uses WhisperX pipeline: transcribe -> align -> diarize -> assign speakers.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "WordSegment",
    "DiarizedSegment",
    "DiarizationResult",
    "Diarizer",
]

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    """A single word with timing and speaker information."""

    start: float
    end: float
    word: str
    speaker: str | None = None


@dataclass
class DiarizedSegment:
    """A transcription segment with speaker information."""

    start: float
    end: float
    text: str
    speaker: str  # "SPEAKER_01", "SPEAKER_02", or custom name
    words: list[WordSegment] = field(default_factory=list)


@dataclass
class DiarizationResult:
    """Complete diarization result."""

    segments: list[DiarizedSegment]
    speakers: list[str]  # Unique speaker labels detected
    language: str
    language_probability: float
    duration: float


def _renumber_speaker(speaker: str) -> str:
    """Renumber speaker labels from 0-indexed to 1-indexed.

    Converts SPEAKER_00 -> SPEAKER_01, SPEAKER_01 -> SPEAKER_02, etc.

    Args:
        speaker: Speaker label (e.g., "SPEAKER_00")

    Returns:
        Renumbered speaker label, or original if not in expected format
    """
    if speaker.startswith("SPEAKER_"):
        try:
            num = int(speaker.split("_")[1])
            return f"SPEAKER_{num + 1:02d}"
        except (IndexError, ValueError):
            pass
    return speaker


class Diarizer:
    """WhisperX-based transcription with speaker diarization.

    This is a daemon-focused diarizer that wraps WhisperX for audio
    transcription with speaker identification.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        hf_token: str | None = None,
        num_speakers: int | None = None,
        compute_type: str = "auto",
    ):
        """Initialize diarizer.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use ("cpu", "cuda", or "auto")
            hf_token: HuggingFace token for pyannote models (required for diarization)
            num_speakers: Expected number of speakers (helps accuracy, None = auto-detect)
            compute_type: Compute type ("int8", "float16", or "auto")
        """
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.num_speakers = num_speakers
        self.compute_type = compute_type
        self._model = None
        self._actual_device: str | None = None

    def close(self) -> None:
        """Release the loaded model and free memory."""
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
        """Lazy-load the WhisperX model."""
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
                    "WhisperX is required for diarization. Install with: pip install whisperx"
                ) from e

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

            try:
                self._model = whisperx.load_model(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info("WhisperX model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load WhisperX model: {e}", exc_info=True)
                raise

    def transcribe_and_diarize(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> DiarizationResult:
        """Transcribe audio with speaker diarization.

        Uses WhisperX pipeline:
        1. Transcribe with faster-whisper
        2. Align with wav2vec2 for word timestamps
        3. Diarize with pyannote for speaker detection
        4. Assign speakers to words

        Args:
            audio_path: Path to audio file (WAV format)
            language: Language code (e.g., "en") or None for auto-detection

        Returns:
            Diarization result with speaker-labeled segments

        Raises:
            RuntimeError: If WhisperX is not installed or HF token is missing
            FileNotFoundError: If audio file doesn't exist
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not self.hf_token:
            logger.error("HuggingFace token not provided for diarization")
            raise RuntimeError(
                "HuggingFace token is required for diarization. "
                "Get one at https://huggingface.co/settings/tokens and "
                "accept pyannote model terms at https://huggingface.co/pyannote/speaker-diarization"
            )

        logger.info(
            f"Starting diarization: {audio_path} "
            f"(language={language or 'auto'}, num_speakers={self.num_speakers or 'auto'})"
        )

        self._load_model()

        try:
            import whisperx  # type: ignore
            import whisperx.diarize  # type: ignore
        except ImportError as e:
            logger.error("WhisperX not installed")
            raise RuntimeError("WhisperX is required for diarization") from e

        try:
            # Load audio
            logger.debug("Loading audio file")
            audio = whisperx.load_audio(str(audio_path))

            # Transcribe
            logger.info("Transcribing audio")
            if self._model is None:
                raise RuntimeError("Model not loaded")
            result = self._model.transcribe(audio, batch_size=16, language=language)
            detected_language = result.get("language", "unknown")
            logger.debug(f"Detected language: {detected_language}")

            # Align (get word-level timestamps)
            logger.info("Aligning transcription for word timestamps")
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

            # Diarize (identify speakers)
            logger.info("Running speaker diarization")
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self._actual_device,
            )

            if diarize_model is None:
                logger.error("Failed to load diarization model")
                raise RuntimeError(
                    "Failed to load diarization model. "
                    "Make sure you've accepted the pyannote model terms at "
                    "https://huggingface.co/pyannote/speaker-diarization"
                )

            # Set speaker constraints if specified
            diarize_kwargs: dict[str, int] = {}
            if self.num_speakers is not None:
                diarize_kwargs["min_speakers"] = self.num_speakers
                diarize_kwargs["max_speakers"] = self.num_speakers
                logger.debug(f"Speaker constraint: {self.num_speakers} speakers")

            diarize_segments = diarize_model(audio, **diarize_kwargs)  # type: ignore[arg-type]

            # Assign speakers to words
            logger.debug("Assigning speakers to words")
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Convert to our format
            diarization_result = self._convert_result(result, detected_language, language)

            logger.info(
                f"Diarization complete: {len(diarization_result.segments)} segments, "
                f"{len(diarization_result.speakers)} unique speakers, "
                f"{diarization_result.duration:.1f}s duration"
            )

            return diarization_result
        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True)
            raise

    def _convert_result(
        self,
        whisperx_result: dict,
        detected_language: str,
        explicit_language: str | None,
    ) -> DiarizationResult:
        """Convert WhisperX output to DiarizationResult.

        Args:
            whisperx_result: Raw result from WhisperX
            detected_language: Language detected by WhisperX
            explicit_language: Language explicitly specified by user (or None)

        Returns:
            Standardized DiarizationResult
        """
        segments: list[DiarizedSegment] = []
        speakers_seen: set[str] = set()

        for seg in whisperx_result.get("segments", []):
            speaker = _renumber_speaker(seg.get("speaker", "UNKNOWN"))
            speakers_seen.add(speaker)

            # Extract word-level information
            words: list[WordSegment] = []
            for word_info in seg.get("words", []):
                word_speaker = word_info.get("speaker")
                if word_speaker:
                    word_speaker = _renumber_speaker(word_speaker)

                words.append(
                    WordSegment(
                        start=word_info.get("start", 0.0),
                        end=word_info.get("end", 0.0),
                        word=word_info.get("word", ""),
                        speaker=word_speaker,
                    )
                )

            segments.append(
                DiarizedSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                    speaker=speaker,
                    words=words,
                )
            )

        # Calculate duration from the segment with the latest end time
        duration = max((s.end for s in segments), default=0.0)

        # WhisperX doesn't expose language probability via its API
        language_probability = 0.0

        return DiarizationResult(
            segments=segments,
            speakers=sorted(speakers_seen),
            language=detected_language,
            language_probability=language_probability,
            duration=duration,
        )
