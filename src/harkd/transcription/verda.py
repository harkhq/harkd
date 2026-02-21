"""Verda (DataCrunch) transcription backend with async polling."""

import asyncio
import json
import logging
from typing import Any

import httpx

from harkd.config import FileStorageSettings, VerdaProviderSettings
from harkd.exceptions import RemoteTranscriptionError
from harkd.transcription.backend import BackendHealth, TranscriptionBackend, TranscriptionRequest
from harkd.transcription.file_storage import FileStorageClient

__all__ = ["VerdaBackend"]

logger = logging.getLogger(__name__)

# Verda status codes
_STATUS_INITIALIZED = 0
_STATUS_QUEUED = 1
_STATUS_PROCESSING = 2
_STATUS_COMPLETE = 3


class VerdaBackend(TranscriptionBackend):
    """Verda (DataCrunch) GPU cloud backend with async job polling.

    Uses Prefer: respond-async for long-running transcription jobs.
    Audio is uploaded to S3-compatible storage and passed as a presigned URL.
    EU region: Helsinki (L40S 48GB).
    """

    def __init__(
        self,
        endpoint_url: str,
        provider: VerdaProviderSettings,
        worker_api_key: str | None = None,
        file_storage: FileStorageSettings | None = None,
        timeout: int = 3600,
        max_retries: int = 2,
    ):
        self._endpoint = endpoint_url.rstrip("/")
        self._worker_api_key = worker_api_key
        self._api_key = provider.api_key
        self._poll_interval = provider.poll_interval
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0))
        self._max_retries = max_retries
        self._file_storage = FileStorageClient(file_storage) if file_storage else None

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if self._worker_api_key:
            headers["X-Worker-Api-Key"] = self._worker_api_key
        return headers

    async def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]:
        if self._file_storage:
            return await self._transcribe_via_url(request)
        return await self._transcribe_via_upload(request)

    async def _transcribe_via_url(self, request: TranscriptionRequest) -> dict[str, Any]:
        """Upload to S3, submit async job with URL, poll until complete."""
        assert self._file_storage is not None
        audio_url, s3_key = await self._file_storage.upload(request.audio_path)
        try:
            return await self._submit_and_poll(request, audio_url=audio_url)
        finally:
            await self._file_storage.delete(s3_key)

    async def _transcribe_via_upload(self, request: TranscriptionRequest) -> dict[str, Any]:
        """Direct multipart upload with async polling."""
        headers = {**self._build_headers(), "Prefer": "respond-async"}
        params_json = json.dumps(request.to_params_dict())

        with open(request.audio_path, "rb") as f:
            files = {"audio": (request.audio_path.name, f, "audio/wav")}
            data = {"params": params_json}
            response = await self._client.post(
                f"{self._endpoint}/transcribe",
                files=files,
                data=data,
                headers=headers,
            )

        if response.status_code >= 400:
            raise RemoteTranscriptionError(
                message=f"Verda job submission failed: HTTP {response.status_code}",
                provider="verda",
                details={"status_code": response.status_code, "body": response.text[:1000]},
            )

        task_info = response.json()
        return await self._poll_until_complete(task_info)

    async def _submit_and_poll(
        self,
        request: TranscriptionRequest,
        audio_url: str,
    ) -> dict[str, Any]:
        """Submit async job with audio URL and poll until complete."""
        headers = {**self._build_headers(), "Prefer": "respond-async"}
        payload = {
            "audio_url": audio_url,
            "params": request.to_params_dict(),
        }

        response = await self._client.post(
            f"{self._endpoint}/transcribe",
            json=payload,
            headers=headers,
        )

        if response.status_code >= 400:
            raise RemoteTranscriptionError(
                message=f"Verda job submission failed: HTTP {response.status_code}",
                provider="verda",
                details={"status_code": response.status_code, "body": response.text[:1000]},
            )

        task_info = response.json()
        return await self._poll_until_complete(task_info)

    async def _poll_until_complete(self, task_info: dict[str, Any]) -> dict[str, Any]:
        """Poll Verda status endpoint until job completes.

        Raises RemoteTranscriptionError if the job doesn't complete within
        the client timeout (default 3600s).
        """
        task_id: str = str(task_info.get("Id", ""))
        status_path: str = str(task_info.get("StatusPath", ""))
        result_path: str = str(task_info.get("ResultPath", ""))

        if not status_path or not result_path:
            raise RemoteTranscriptionError(
                message="Verda response missing StatusPath or ResultPath",
                provider="verda",
                details={"task_info": task_info},
            )

        logger.info(f"Verda job submitted: {task_id}, polling for completion...")

        # Use configured timeout as max poll duration to prevent infinite polling
        max_poll_seconds = self._timeout
        elapsed = 0.0

        while elapsed < max_poll_seconds:
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

            headers = {
                **self._build_headers(),
                "X-Inference-Id": task_id,
            }
            status_resp = await self._client.get(
                f"{self._endpoint}{status_path}",
                headers=headers,
            )

            if status_resp.status_code >= 400:
                raise RemoteTranscriptionError(
                    message=f"Verda status poll failed: HTTP {status_resp.status_code}",
                    provider="verda",
                    details={"task_id": task_id},
                )

            status_data = status_resp.json()
            status_code = status_data.get("Status", -1)

            if status_code == _STATUS_COMPLETE:
                break
            elif status_code in (_STATUS_INITIALIZED, _STATUS_QUEUED, _STATUS_PROCESSING):
                logger.debug(f"Verda job {task_id}: status={status_code}")
                continue
            else:
                raise RemoteTranscriptionError(
                    message=f"Verda job failed with status {status_code}",
                    provider="verda",
                    details={"task_id": task_id, "status": status_data},
                )
        else:
            # Loop exited without break — polling timed out
            raise RemoteTranscriptionError(
                message=f"Verda job {task_id} timed out after {max_poll_seconds}s of polling",
                provider="verda",
                details={"task_id": task_id, "elapsed": elapsed},
            )

        # Fetch result
        result_resp = await self._client.get(
            f"{self._endpoint}{result_path}",
            headers={**self._build_headers(), "X-Inference-Id": task_id},
        )

        if result_resp.status_code >= 400:
            raise RemoteTranscriptionError(
                message=f"Verda result fetch failed: HTTP {result_resp.status_code}",
                provider="verda",
                details={"task_id": task_id},
            )

        return result_resp.json()

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
