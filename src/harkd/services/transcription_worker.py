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
    beam_size: int = 3,
    batch_size: int = 16,
    vad_onset: float = 0.5,
    vad_offset: float = 0.363,
    vad_method: str = "pyannote",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    clustering_threshold: float | None = None,
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
        beam_size: Beam size for decoding
        batch_size: Batch size for transcription
        vad_onset: VAD onset threshold
        vad_offset: VAD offset threshold
        vad_method: VAD method ("pyannote" or "silero")
        num_speakers: Exact speaker count hint for diarization
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        clustering_threshold: Agglomerative clustering threshold

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
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            clustering_threshold=clustering_threshold,
            compute_type="auto",
            beam_size=beam_size,
            batch_size=batch_size,
            vad_onset=vad_onset,
            vad_offset=vad_offset,
            vad_method=vad_method,
        ) as diarizer:
            result = diarizer.transcribe_and_diarize(audio_path, language=language)

        return {
            "text": " ".join(seg.text for seg in result.segments),
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "speakers": result.speakers,
            "speaker_embeddings": result.speaker_embeddings,
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
            beam_size=beam_size,
            batch_size=batch_size,
            vad_onset=vad_onset,
            vad_offset=vad_offset,
            vad_method=vad_method,
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

    if len(sys.argv) < 7:
        print(
            "Usage: transcription_worker.py <audio_path> <model_name> <language> "
            "<word_timestamps> <diarize> <hf_token> "
            "[beam_size] [batch_size] [vad_onset] [vad_offset] [vad_method]"
        )
        sys.exit(1)

    audio_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3] if sys.argv[3] != "None" else None
    word_timestamps = sys.argv[4].lower() == "true"
    diarize = sys.argv[5].lower() == "true"
    hf_token = sys.argv[6] if sys.argv[6] != "None" else None

    # Optional performance params (with defaults matching config)
    beam_size = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    batch_size = int(sys.argv[8]) if len(sys.argv) > 8 else 16
    vad_onset = float(sys.argv[9]) if len(sys.argv) > 9 else 0.5
    vad_offset = float(sys.argv[10]) if len(sys.argv) > 10 else 0.363
    vad_method = sys.argv[11] if len(sys.argv) > 11 else "pyannote"

    # Optional diarization tuning params
    num_speakers = int(sys.argv[12]) if len(sys.argv) > 12 and sys.argv[12] != "None" else None
    min_speakers = int(sys.argv[13]) if len(sys.argv) > 13 and sys.argv[13] != "None" else None
    max_speakers = int(sys.argv[14]) if len(sys.argv) > 14 and sys.argv[14] != "None" else None
    clustering_threshold = (
        float(sys.argv[15]) if len(sys.argv) > 15 and sys.argv[15] != "None" else None
    )

    result = transcribe_audio_worker(
        audio_path,
        model_name,
        language,
        word_timestamps,
        diarize,
        hf_token,
        beam_size=beam_size,
        batch_size=batch_size,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        vad_method=vad_method,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        clustering_threshold=clustering_threshold,
    )
    # Print delimiter before JSON for reliable parsing
    print(_JSON_DELIMITER)
    print(json.dumps(result))
