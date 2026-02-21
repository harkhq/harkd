"""Local transcription backend — runs WhisperX in a subprocess."""

import asyncio
import json
import logging
import subprocess
import sys
from typing import Any

from harkd.transcription.backend import BackendHealth, TranscriptionBackend, TranscriptionRequest

__all__ = ["LocalBackend"]

logger = logging.getLogger(__name__)

# JSON delimiter used to find structured output in subprocess stdout
_JSON_DELIMITER = "---HARKD_JSON_RESULT---"

# Default subprocess timeout (30 minutes)
_DEFAULT_SUBPROCESS_TIMEOUT = 1800


class LocalBackend(TranscriptionBackend):
    """Runs transcription in a local subprocess via transcription_worker.py.

    This is the default backend — identical behavior to the original
    ProcessingWorker._run_transcription_subprocess().
    """

    async def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]:
        effective_timeout = request.timeout or _DEFAULT_SUBPROCESS_TIMEOUT

        cmd = [
            sys.executable,
            "-m",
            "harkd.services.transcription_worker",
            str(request.audio_path),
            request.model_name,
            str(request.language) if request.language else "None",
            str(request.word_timestamps).lower(),
            str(request.diarize).lower(),
            request.hf_token or "None",
            str(request.beam_size),
            str(request.batch_size),
            str(request.vad_onset),
            str(request.vad_offset),
            request.vad_method,
            str(request.num_speakers) if request.num_speakers is not None else "None",
            str(request.min_speakers) if request.min_speakers is not None else "None",
            str(request.max_speakers) if request.max_speakers is not None else "None",
            (
                str(request.clustering_threshold)
                if request.clustering_threshold is not None
                else "None"
            ),
        ]

        logger.info(
            f"Starting transcription subprocess (timeout={effective_timeout}s): {' '.join(cmd)}"
        )

        def run_subprocess():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )

        loop = asyncio.get_event_loop()
        try:
            proc = await loop.run_in_executor(None, run_subprocess)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Transcription timed out after {effective_timeout}s") from e

        stdout = proc.stdout
        stderr = proc.stderr

        if proc.returncode != 0:
            raise RuntimeError(f"Transcription failed: {stderr}")

        logger.info("Transcription subprocess completed successfully")

        try:
            if _JSON_DELIMITER in stdout:
                json_str = stdout.split(_JSON_DELIMITER)[-1].strip()
            else:
                json_str = stdout.strip().split("\n")[-1]
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse transcription output: {e}. stdout: {stdout}")
            raise

    async def health_check(self) -> BackendHealth:
        return BackendHealth(healthy=True, details={"backend": "local"})
