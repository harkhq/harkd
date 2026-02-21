"""ManagedBackend — TranscriptionBackend wrapper with infrastructure lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harkd.exceptions import InfraProvisioningError
from harkd.transcription.backend import BackendHealth, TranscriptionBackend, TranscriptionRequest
from harkd.transcription.infra.provider import InfraProvider, InfraProviderConfig, InfraState

__all__ = ["ManagedBackend"]

logger = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = Path.home() / ".local/share/hark"


class ManagedBackend(TranscriptionBackend):
    """Transcription backend that auto-provisions and tears down cloud infrastructure.

    Wraps an inner TranscriptionBackend that is created dynamically after
    infrastructure provisioning discovers an endpoint URL.

    State machine::

        dormant ──[transcribe/pre_warm]──> provisioning ──[success]──> running
           ^                                     |                       |
           |                              [failure]                [idle timeout /
           |                                     |                  max runtime /
           +──────────────[done]──── tearing_down <─────────────── close()]
    """

    def __init__(
        self,
        infra_provider: InfraProvider,
        infra_config: InfraProviderConfig,
        backend_factory: Callable[[str], TranscriptionBackend],
        idle_timeout: int = 300,
        max_runtime: int = 0,
        provisioning_timeout: int = 600,
        state_file: Path | None = None,
    ):
        self._infra = infra_provider
        self._infra_config = infra_config
        self._backend_factory = backend_factory
        self._idle_timeout = idle_timeout
        self._max_runtime = max_runtime
        self._provisioning_timeout = provisioning_timeout
        self._state_file = state_file or (
            _DEFAULT_STATE_DIR / f"infra_state_{infra_provider.provider_name()}.json"
        )

        self._state = InfraState.DORMANT
        self._instance_id: str | None = None
        self._endpoint_url: str | None = None
        self._created_at: datetime | None = None
        self._inner: TranscriptionBackend | None = None
        self._lock = asyncio.Lock()
        self._idle_task: asyncio.Task[None] | None = None
        self._runtime_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> InfraState:
        return self._state

    async def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]:
        """Provision if needed, then delegate to inner backend."""
        await self._ensure_ready()
        assert self._inner is not None
        try:
            result = await self._inner.transcribe(request)
        finally:
            self._reset_idle_timer()
        return result

    async def health_check(self) -> BackendHealth:
        """Return health status. Dormant is healthy (can provision on demand)."""
        if self._state == InfraState.DORMANT:
            return BackendHealth(
                healthy=True,
                details={
                    "state": self._state.value,
                    "provider": self._infra.provider_name(),
                    "managed": True,
                },
            )
        if self._inner is not None:
            inner_health = await self._inner.health_check()
            inner_health.details["state"] = self._state.value
            inner_health.details["managed"] = True
            return inner_health
        return BackendHealth(
            healthy=True,
            details={
                "state": self._state.value,
                "provider": self._infra.provider_name(),
                "managed": True,
            },
        )

    async def close(self) -> None:
        """Cancel timers, tear down infrastructure, clean up inner backend."""
        self._cancel_timers()
        if self._state == InfraState.RUNNING and self._instance_id:
            await self._do_teardown()
        if self._inner is not None:
            await self._inner.close()
            self._inner = None

    async def pre_warm(self) -> None:
        """Start provisioning in the background (non-blocking)."""
        if self._state != InfraState.DORMANT:
            return
        asyncio.create_task(self._ensure_ready())

    async def recover_orphans(self) -> None:
        """Check for orphaned infrastructure from previous runs."""
        if not self._state_file.exists():
            return

        try:
            data = json.loads(self._state_file.read_text())
            instance_id = data.get("instance_id")
            if not instance_id:
                self._state_file.unlink(missing_ok=True)
                return

            logger.info(
                "[%s] Found orphaned infra state: instance=%s",
                self._infra.provider_name(),
                instance_id,
            )

            status = await self._infra.get_status(instance_id)

            if status.state == InfraState.RUNNING and status.endpoint_url:
                logger.info(
                    "[%s] Adopting running orphan instance %s at %s",
                    self._infra.provider_name(),
                    instance_id,
                    status.endpoint_url,
                )
                self._state = InfraState.RUNNING
                self._instance_id = instance_id
                self._endpoint_url = status.endpoint_url
                self._created_at = status.created_at
                self._inner = self._backend_factory(status.endpoint_url)
                self._reset_idle_timer()
                if self._max_runtime > 0 and self._created_at:
                    self._start_runtime_timer()
            else:
                logger.info(
                    "[%s] Orphan instance %s is %s, tearing down",
                    self._infra.provider_name(),
                    instance_id,
                    status.state.value,
                )
                try:
                    await self._infra.teardown(instance_id)
                except Exception:
                    logger.warning(
                        "[%s] Failed to teardown orphan %s",
                        self._infra.provider_name(),
                        instance_id,
                        exc_info=True,
                    )
                self._state_file.unlink(missing_ok=True)

        except Exception:
            logger.warning(
                "[%s] Failed to recover orphaned infra",
                self._infra.provider_name(),
                exc_info=True,
            )
            self._state_file.unlink(missing_ok=True)

    async def _ensure_ready(self) -> None:
        """Ensure infrastructure is provisioned and inner backend is ready.

        Serialized by lock so concurrent callers wait rather than double-provision.
        """
        if self._state == InfraState.RUNNING and self._inner is not None:
            return

        async with self._lock:
            if self._state == InfraState.RUNNING and self._inner is not None:
                return

            self._state = InfraState.PROVISIONING
            logger.info("[%s] Provisioning infrastructure...", self._infra.provider_name())

            try:
                status = await asyncio.wait_for(
                    self._infra.provision(self._infra_config),
                    timeout=self._provisioning_timeout,
                )
            except TimeoutError as e:
                self._state = InfraState.DORMANT
                raise InfraProvisioningError(
                    f"Provisioning timed out after {self._provisioning_timeout}s",
                    provider=self._infra.provider_name(),
                ) from e
            except Exception as e:
                self._state = InfraState.DORMANT
                raise InfraProvisioningError(
                    f"Provisioning failed: {e}",
                    provider=self._infra.provider_name(),
                    details={"original_error": str(e)},
                ) from e

            if not status.endpoint_url:
                self._state = InfraState.DORMANT
                raise InfraProvisioningError(
                    "Provisioning succeeded but no endpoint URL returned",
                    provider=self._infra.provider_name(),
                )

            self._instance_id = status.instance_id
            self._endpoint_url = status.endpoint_url
            self._created_at = status.created_at or datetime.now(UTC)
            self._inner = self._backend_factory(status.endpoint_url)
            self._state = InfraState.RUNNING

            self._save_state()
            self._reset_idle_timer()
            if self._max_runtime > 0:
                self._start_runtime_timer()

            logger.info(
                "[%s] Infrastructure ready: instance=%s endpoint=%s",
                self._infra.provider_name(),
                self._instance_id,
                self._endpoint_url,
            )

    async def _do_teardown(self) -> None:
        """Tear down infrastructure and return to dormant state."""
        async with self._lock:
            if self._state not in (InfraState.RUNNING, InfraState.TEARING_DOWN):
                return

            self._state = InfraState.TEARING_DOWN
            instance_id = self._instance_id

            logger.info(
                "[%s] Tearing down infrastructure: instance=%s",
                self._infra.provider_name(),
                instance_id,
            )

            if self._inner is not None:
                try:
                    await self._inner.close()
                except Exception:
                    logger.warning(
                        "[%s] Error closing inner backend",
                        self._infra.provider_name(),
                        exc_info=True,
                    )
                self._inner = None

            if instance_id:
                try:
                    await self._infra.teardown(instance_id)
                except Exception:
                    logger.error(
                        "[%s] Error tearing down instance %s",
                        self._infra.provider_name(),
                        instance_id,
                        exc_info=True,
                    )

            self._instance_id = None
            self._endpoint_url = None
            self._created_at = None
            self._state = InfraState.DORMANT
            self._cancel_timers()
            self._clear_state()

            logger.info("[%s] Infrastructure torn down, now dormant", self._infra.provider_name())

    def _reset_idle_timer(self) -> None:
        """Reset the idle timeout timer."""
        if self._idle_timeout <= 0:
            return

        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()

        self._idle_task = asyncio.create_task(self._idle_timer())

    async def _idle_timer(self) -> None:
        """Sleep for idle_timeout then tear down."""
        try:
            await asyncio.sleep(self._idle_timeout)
            logger.info(
                "[%s] Idle timeout (%ds) reached, tearing down",
                self._infra.provider_name(),
                self._idle_timeout,
            )
            await self._do_teardown()
        except asyncio.CancelledError:
            pass

    def _start_runtime_timer(self) -> None:
        """Start a max-runtime timer."""
        if self._runtime_task is not None and not self._runtime_task.done():
            self._runtime_task.cancel()
        self._runtime_task = asyncio.create_task(self._runtime_timer())

    async def _runtime_timer(self) -> None:
        """Sleep for max_runtime then tear down."""
        try:
            await asyncio.sleep(self._max_runtime)
            logger.info(
                "[%s] Max runtime (%ds) reached, tearing down",
                self._infra.provider_name(),
                self._max_runtime,
            )
            await self._do_teardown()
        except asyncio.CancelledError:
            pass

    def _cancel_timers(self) -> None:
        """Cancel all running timers."""
        for task in (self._idle_task, self._runtime_task):
            if task is not None and not task.done():
                task.cancel()
        self._idle_task = None
        self._runtime_task = None

    def _save_state(self) -> None:
        """Persist infrastructure state for orphan recovery."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "provider": self._infra.provider_name(),
                "instance_id": self._instance_id,
                "endpoint_url": self._endpoint_url,
                "created_at": self._created_at.isoformat() if self._created_at else None,
            }
            self._state_file.write_text(json.dumps(data))
        except Exception:
            logger.warning("Failed to save infra state", exc_info=True)

    def _clear_state(self) -> None:
        """Remove persisted infrastructure state."""
        try:
            self._state_file.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to clear infra state file", exc_info=True)
