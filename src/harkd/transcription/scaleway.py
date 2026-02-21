"""Scaleway transcription backend."""

from harkd.transcription.remote import RemoteBackend

__all__ = ["ScalewayBackend"]


class ScalewayBackend(RemoteBackend):
    """Scaleway GPU cloud backend.

    Synchronous HTTP POST to a user-managed endpoint.
    Auth via X-Auth-Token header. EU region: Paris (L4 24GB).
    Also works as a generic backend for any self-hosted worker.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
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
        self._api_key = api_key

    def _provider_headers(self) -> dict[str, str]:
        if self._api_key:
            return {"X-Auth-Token": self._api_key}
        return {}

    def _provider_name(self) -> str:
        return "scaleway"
