"""Infrastructure provider abstraction for cloud GPU lifecycle management."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class InfraState(str, Enum):
    """State of cloud infrastructure."""

    DORMANT = "dormant"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    TEARING_DOWN = "tearing_down"


@dataclass
class InfraStatus:
    """Current status of cloud infrastructure."""

    state: InfraState
    provider: str
    instance_id: str | None = None
    endpoint_url: str | None = None
    created_at: datetime | None = None
    last_activity: datetime | None = None


@dataclass
class InfraProviderConfig:
    """Configuration passed to InfraProvider for provisioning."""

    docker_image: str
    worker_api_key: str
    worker_model: str = "large-v3"
    hf_token: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


class InfraProvider(ABC):
    """Abstract base class for cloud infrastructure lifecycle management.

    Each provider implements provisioning, teardown, and status checking
    for its specific cloud platform.

    Key distinction:
    - ``teardown()``: cheap/reversible (pause, stop) -- preserves resources
    - ``destroy()``: permanent (delete all resources)
    """

    @abstractmethod
    async def provision(self, config: InfraProviderConfig) -> InfraStatus:
        """Provision or resume cloud infrastructure.

        Returns InfraStatus with state=RUNNING and endpoint_url set.
        """
        ...

    @abstractmethod
    async def teardown(self, instance_id: str) -> None:
        """Stop/pause infrastructure (reversible).

        For providers without pause, this is equivalent to destroy().
        """
        ...

    @abstractmethod
    async def destroy(self, instance_id: str) -> None:
        """Permanently delete infrastructure and all resources."""
        ...

    @abstractmethod
    async def get_status(self, instance_id: str) -> InfraStatus:
        """Get current status of a provisioned instance."""
        ...

    @abstractmethod
    async def discover_running(self) -> list[InfraStatus]:
        """Discover any running instances (for orphan recovery)."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g. 'koyeb', 'scaleway', 'datacrunch')."""
        ...

    async def _log_error_response(self, response: httpx.Response) -> None:
        """Log response body when an API request returns an error."""
        if response.is_error:
            logger.error(
                "[%s] API error %s %s — %s: %s",
                self.provider_name(),
                response.request.method,
                response.request.url,
                response.status_code,
                response.text,
            )
