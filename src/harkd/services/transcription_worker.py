"""Transcription worker that runs in a separate process."""

import sys
from pathlib import Path

# JSON delimiter for structured output parsing
_JSON_DELIMITER = "---HARKD_JSON_RESULT---"


def _patch_torch_serialization():
    """Patch torch serialization to use weights_only=False by default for WhisperX compatibility.

    This is needed because WhisperX models use pickle-based serialization that
    is not compatible with PyTorch 2.6+'s weights_only=True default.
    """
    try:
        import torch.serialization

        _original_load = torch.serialization.load

        def _patched_load(
            f,
            map_location=None,
            pickle_module=None,
            *,
            weights_only=None,
            **pickle_load_args,
        ):
            if weights_only is None:
                weights_only = False
            return _original_load(
                f,
                map_location,
                pickle_module,
                weights_only=weights_only,
                **pickle_load_args,
            )

        torch.serialization.load = _patched_load
        torch.load = _patched_load
    except ImportError:
        pass


# Apply patch before importing Transcriber (which imports WhisperX/PyTorch)
_patch_torch_serialization()


def transcribe_audio_worker(
    audio_path_str: str,
    model_name: str,
    language: str | None,
    word_timestamps: bool,
    diarize: bool = False,
    hf_token: str | None = None,
):
    """Worker function to run transcription in a separate process.

    This runs in a completely separate process to avoid threading/async issues
    with PyTorch/WhisperX in the uvicorn server context.

    Args:
        audio_path_str: Path to audio file as string
        model_name: Whisper model name
        language: Language code or None for auto-detect
        word_timestamps: Whether to include word timestamps
        diarize: Whether to run speaker diarization
        hf_token: HuggingFace token for diarization models

    Returns:
        Dict with transcription result
    """
    audio_path = Path(audio_path_str)

    if diarize and hf_token:
        from harkd.audio.diarizer import Diarizer

        with Diarizer(
            model_name=model_name,
            device="auto",
            hf_token=hf_token,
            compute_type="auto",
        ) as diarizer:
            result = diarizer.transcribe_and_diarize(audio_path, language=language)

        return {
            "text": " ".join(seg.text for seg in result.segments),
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "speakers": result.speakers,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "word": w.word,
                            "speaker": w.speaker,
                        }
                        for w in seg.words
                    ],
                }
                for seg in result.segments
            ],
        }
    else:
        # Import here to avoid loading in main process
        from harkd.audio.transcriber import Transcriber

        with Transcriber(
            model_name=model_name,
            device="auto",
            language=language,
            compute_type="auto",
        ) as transcriber:
            result = transcriber.transcribe(audio_path, word_timestamps=word_timestamps)

        # Convert to serializable dict
        return {
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "word": w.word,
                        }
                        for w in seg.words
                    ],
                }
                for seg in result.segments
            ],
        }


if __name__ == "__main__":
    # Allow running as standalone script for testing
    import json

    if len(sys.argv) != 7:
        print(
            "Usage: transcription_worker.py <audio_path> <model_name> <language> "
            "<word_timestamps> <diarize> <hf_token>"
        )
        sys.exit(1)

    audio_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3] if sys.argv[3] != "None" else None
    word_timestamps = sys.argv[4].lower() == "true"
    diarize = sys.argv[5].lower() == "true"
    hf_token = sys.argv[6] if sys.argv[6] != "None" else None

    result = transcribe_audio_worker(
        audio_path, model_name, language, word_timestamps, diarize, hf_token
    )
    # Print delimiter before JSON for reliable parsing
    print(_JSON_DELIMITER)
    print(json.dumps(result))
