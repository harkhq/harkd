"""Base remote transcription backend — HTTP-based with retries."""

import json
import logging
from typing import Any

import httpx

from harkd.exceptions import RemoteTranscriptionError
from harkd.transcription.backend import BackendHealth, TranscriptionBackend, TranscriptionRequest

__all__ = ["RemoteBackend"]

logger = logging.getLogger(__name__)


class RemoteBackend(TranscriptionBackend):
    """Base class for HTTP-based remote transcription backends.

    Handles file upload, retry with exponential backoff, and health checks.
    Provider subclasses override _provider_headers() for platform-specific auth.
    """

    def __init__(
        self,
        endpoint_url: str,
        worker_api_key: str | None = None,
        timeout: int = 3600,
        max_retries: int = 2,
    ):
        self._endpoint = endpoint_url.rstrip("/")
        self._worker_api_key = worker_api_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0))
        self._max_retries = max_retries

    def _build_headers(self) -> dict[str, str]:
        """Combine worker auth + provider-specific headers."""
        headers: dict[str, str] = {}
        if self._worker_api_key:
            headers["Authorization"] = f"Bearer {self._worker_api_key}"
        headers.update(self._provider_headers())
        return headers

    def _provider_headers(self) -> dict[str, str]:
        """Override in subclasses for provider-specific headers."""
        return {}

    async def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]:
        headers = self._build_headers()
        params_json = json.dumps(request.to_params_dict())
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 2):  # +2 because range is exclusive + 1 initial
            try:
                with open(request.audio_path, "rb") as f:
                    files = {"audio": (request.audio_path.name, f, "audio/wav")}
                    data = {"params": params_json}
                    response = await self._client.post(
                        f"{self._endpoint}/transcribe",
                        files=files,
                        data=data,
                        headers=headers,
                    )

                if response.status_code >= 500:
                    last_error = RemoteTranscriptionError(
                        message=f"HTTP {response.status_code}: {response.text[:500]}",
                        provider=self._provider_name(),
                    )
                    if attempt <= self._max_retries:
                        logger.warning(
                            f"Remote transcription returned {response.status_code}, "
                            f"retrying ({attempt}/{self._max_retries})..."
                        )
                        continue
                    # Last attempt — fall through to after-loop error
                elif response.status_code >= 400:
                    raise RemoteTranscriptionError(
                        message=f"Remote transcription failed: HTTP {response.status_code}",
                        provider=self._provider_name(),
                        details={
                            "status_code": response.status_code,
                            "body": response.text[:1000],
                        },
                    )
                else:
                    return response.json()

            except httpx.TimeoutException as e:
                last_error = e
                if attempt <= self._max_retries:
                    logger.warning(
                        f"Remote transcription timed out, "
                        f"retrying ({attempt}/{self._max_retries})..."
                    )
                    continue
            except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt <= self._max_retries:
                    logger.warning(
                        f"Remote transcription connection error: {e}, "
                        f"retrying ({attempt}/{self._max_retries})..."
                    )
                    continue
            except RemoteTranscriptionError:
                raise
            except httpx.HTTPError as e:
                last_error = e
                if attempt <= self._max_retries:
                    logger.warning(
                        f"Remote transcription HTTP error: {e}, "
                        f"retrying ({attempt}/{self._max_retries})..."
                    )
                    continue

        total = self._max_retries + 1
        raise RemoteTranscriptionError(
            message=f"Remote transcription failed after {total} attempts: {last_error}",
            provider=self._provider_name(),
        )

    def _provider_name(self) -> str:
        """Return the provider name for error messages."""
        return "remote"

    async def health_check(self) -> BackendHealth:
        try:
            response = await self._client.get(
                f"{self._endpoint}/health", headers=self._build_headers()
            )
            if response.status_code == 200:
                return BackendHealth(healthy=True, details=response.json())
            return BackendHealth(
                healthy=False,
                details={"status_code": response.status_code},
            )
        except httpx.HTTPError as e:
            return BackendHealth(healthy=False, details={"error": str(e)})

    async def close(self) -> None:
        await self._client.aclose()
