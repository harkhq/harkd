"""Koyeb transcription backend."""

from harkd.transcription.remote import RemoteBackend

__all__ = ["KoyebBackend"]


class KoyebBackend(RemoteBackend):
    """Koyeb GPU cloud backend.

    Synchronous HTTP POST — the container processes and returns the result.
    Auth via Bearer token. EU region: Frankfurt (RTX-4000-SFF-ADA).
    """

    def __init__(
        self,
        endpoint_url: str,
        token: str,
        worker_api_key: str | None = None,
        timeout: int = 3600,
        max_retries: int = 2,
    ):
        super().__init__(
            endpoint_url=endpoint_url,
            worker_api_key=worker_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._token = token

    def _provider_headers(self) -> dict[str, str]:
        return {"X-Koyeb-Token": self._token}

    def _provider_name(self) -> str:
        return "koyeb"
