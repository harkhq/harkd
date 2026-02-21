"""Tests for TranscriptionRequest and backend protocol."""

from pathlib import Path

from harkd.transcription.backend import TranscriptionRequest


class TestTranscriptionRequest:
    """Tests for TranscriptionRequest dataclass."""

    def test_basic_construction(self):
        """Test creating a basic request with required fields."""
        req = TranscriptionRequest(
            audio_path=Path("/tmp/audio.wav"),
            model_name="large-v3",
            language=None,
            word_timestamps=False,
            diarize=False,
            hf_token=None,
            beam_size=3,
            batch_size=16,
            vad_onset=0.5,
            vad_offset=0.363,
            vad_method="pyannote",
            timeout=1800,
        )
        assert req.audio_path == Path("/tmp/audio.wav")
        assert req.model_name == "large-v3"
        assert req.language is None
        assert req.num_speakers is None
        assert req.min_speakers is None
        assert req.max_speakers is None
        assert req.clustering_threshold is None

    def test_with_diarization_params(self):
        """Test request with diarization tuning params."""
        req = TranscriptionRequest(
            audio_path=Path("/tmp/audio.wav"),
            model_name="large-v3",
            language="en",
            word_timestamps=True,
            diarize=True,
            hf_token="hf_test_token",
            beam_size=5,
            batch_size=8,
            vad_onset=0.5,
            vad_offset=0.363,
            vad_method="pyannote",
            timeout=3600,
            num_speakers=2,
            min_speakers=1,
            max_speakers=5,
            clustering_threshold=0.7,
        )
        assert req.diarize is True
        assert req.hf_token == "hf_test_token"
        assert req.num_speakers == 2
        assert req.min_speakers == 1
        assert req.max_speakers == 5
        assert req.clustering_threshold == 0.7

    def test_to_params_dict_basic(self):
        """Test converting request to params dict."""
        req = TranscriptionRequest(
            audio_path=Path("/tmp/audio.wav"),
            model_name="base",
            language=None,
            word_timestamps=False,
            diarize=False,
            hf_token=None,
            beam_size=3,
            batch_size=16,
            vad_onset=0.5,
            vad_offset=0.363,
            vad_method="pyannote",
            timeout=1800,
        )
        params = req.to_params_dict()

        assert params["model_name"] == "base"
        assert params["language"] is None
        assert params["word_timestamps"] is False
        assert params["diarize"] is False
        assert params["beam_size"] == 3
        assert params["batch_size"] == 16
        assert "hf_token" not in params
        assert "num_speakers" not in params
        assert "audio_path" not in params

    def test_to_params_dict_with_optional_fields(self):
        """Test params dict includes optional fields when set."""
        req = TranscriptionRequest(
            audio_path=Path("/tmp/audio.wav"),
            model_name="large-v3",
            language="en",
            word_timestamps=True,
            diarize=True,
            hf_token="hf_test",
            beam_size=5,
            batch_size=8,
            vad_onset=0.5,
            vad_offset=0.363,
            vad_method="silero",
            timeout=3600,
            num_speakers=3,
            min_speakers=2,
            max_speakers=4,
            clustering_threshold=0.8,
        )
        params = req.to_params_dict()

        assert params["hf_token"] == "hf_test"
        assert params["num_speakers"] == 3
        assert params["min_speakers"] == 2
        assert params["max_speakers"] == 4
        assert params["clustering_threshold"] == 0.8
        assert params["vad_method"] == "silero"

    def test_to_params_dict_keys_match_worker_contract(self):
        """Verify to_params_dict() keys match what the worker reads.

        The worker (app.py) reads params via params_dict.get(key, default).
        This test ensures no key drift between the two sides.
        """
        # Keys the worker reads from the params dict (from worker/app.py)
        worker_keys = {
            "model_name",
            "language",
            "word_timestamps",
            "diarize",
            "hf_token",
            "beam_size",
            "batch_size",
            "vad_onset",
            "vad_offset",
            "vad_method",
            "num_speakers",
            "min_speakers",
            "max_speakers",
            "clustering_threshold",
        }

        # Build a request with ALL optional fields set so to_params_dict includes them
        req = TranscriptionRequest(
            audio_path=Path("/tmp/audio.wav"),
            model_name="large-v3",
            language="en",
            word_timestamps=True,
            diarize=True,
            hf_token="hf_test",
            beam_size=5,
            batch_size=8,
            vad_onset=0.6,
            vad_offset=0.4,
            vad_method="silero",
            timeout=3600,
            num_speakers=3,
            min_speakers=2,
            max_speakers=4,
            clustering_threshold=0.8,
        )
        params = req.to_params_dict()

        assert set(params.keys()) == worker_keys
