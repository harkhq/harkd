"""DataCrunch infrastructure provider.

Creates/deletes instances per usage cycle. No stop/start available.
Longest cold start (2-5min) but cheapest hourly GPU cost.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import httpx

from harkd.transcription.infra.provider import (
    InfraProvider,
    InfraProviderConfig,
    InfraState,
    InfraStatus,
)

__all__ = ["DataCrunchInfraProvider"]

logger = logging.getLogger(__name__)

_DC_API = "https://api.datacrunch.io/v1"
_HOSTNAME = "hark-worker"


class DataCrunchInfraProvider(InfraProvider):
    """DataCrunch infrastructure provider.

    Strategy: create/delete instances per usage cycle.
    No stop/start available so both teardown() and destroy() delete the instance.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        instance_type: str = "1L40S.6V",
        location: str = "FIN-01",
        ssh_key_ids: list[str] | None = None,
        os_volume_id: str | None = None,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._instance_type = instance_type
        self._location = location
        self._ssh_key_ids = ssh_key_ids or []
        self._os_volume_id = os_volume_id
        self._access_token: str | None = None

    def provider_name(self) -> str:
        return "datacrunch"

    async def provision(self, config: InfraProviderConfig) -> InfraStatus:
        async with self._client() as client:
            await self._ensure_token(client)

            logger.info("[datacrunch] Creating instance (type=%s)", self._instance_type)
            create_body = self._build_create_body(config)
            resp = await client.post(
                f"{_DC_API}/instances",
                json=create_body,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            instance_id = data["id"]

            # Wait for running
            await self._wait_for_running(client, instance_id)

            # Get instance details for IP
            instance = await self._get_instance(client, instance_id)
            ip = instance.get("ip")

            if ip:
                await self._wait_for_health(ip)

            return InfraStatus(
                state=InfraState.RUNNING,
                provider="datacrunch",
                instance_id=instance_id,
                endpoint_url=f"http://{ip}:8000" if ip else None,
                created_at=datetime.now(UTC),
            )

    async def teardown(self, instance_id: str) -> None:
        # DataCrunch has no stop/start — teardown is same as destroy
        await self.destroy(instance_id)

    async def destroy(self, instance_id: str) -> None:
        async with self._client() as client:
            await self._ensure_token(client)
            logger.info("[datacrunch] Deleting instance %s", instance_id)
            resp = await client.delete(
                f"{_DC_API}/instances/{instance_id}",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()

    async def get_status(self, instance_id: str) -> InfraStatus:
        async with self._client() as client:
            await self._ensure_token(client)
            instance = await self._get_instance(client, instance_id)
            state = self._map_status(instance.get("status", ""))
            ip = instance.get("ip")
            return InfraStatus(
                state=state,
                provider="datacrunch",
                instance_id=instance_id,
                endpoint_url=f"http://{ip}:8000" if ip else None,
            )

    async def discover_running(self) -> list[InfraStatus]:
        async with self._client() as client:
            await self._ensure_token(client)
            resp = await client.get(
                f"{_DC_API}/instances",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            instances = resp.json()
            if isinstance(instances, dict):
                instances = instances.get("instances", [])

            results = []
            for inst in instances:
                hostname = inst.get("hostname", "")
                description = inst.get("description", "")
                if _HOSTNAME in hostname or _HOSTNAME in description:
                    status = inst.get("status", "")
                    if status in ("running", "deploying"):
                        ip = inst.get("ip")
                        results.append(
                            InfraStatus(
                                state=self._map_status(status),
                                provider="datacrunch",
                                instance_id=inst["id"],
                                endpoint_url=f"http://{ip}:8000" if ip else None,
                            )
                        )
            return results

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=30.0,
            event_hooks={"response": [self._log_error_response]},
        )

    async def _ensure_token(self, client: httpx.AsyncClient) -> None:
        if self._access_token:
            return
        resp = await client.post(
            f"{_DC_API}/oauth2/token",
            json={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _get_instance(self, client: httpx.AsyncClient, instance_id: str) -> dict:
        resp = await client.get(
            f"{_DC_API}/instances/{instance_id}",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def _wait_for_running(
        self, client: httpx.AsyncClient, instance_id: str, timeout: int = 600
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            instance = await self._get_instance(client, instance_id)
            status = instance.get("status", "")
            if status == "running":
                return
            if status in ("error", "terminated"):
                raise RuntimeError(f"DataCrunch instance entered {status} state")
            await asyncio.sleep(10)
        raise TimeoutError(
            f"DataCrunch instance {instance_id} did not reach running state within {timeout}s"
        )

    @staticmethod
    async def _wait_for_health(ip: str, timeout: int = 300) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(f"http://{ip}:8000/health")
                    if resp.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                await asyncio.sleep(10)
        logger.warning("[datacrunch] Health check timed out for %s, proceeding anyway", ip)

    def _build_create_body(self, config: InfraProviderConfig) -> dict:
        body: dict = {
            "instance_type": self._instance_type,
            "hostname": _HOSTNAME,
            "description": "hark-worker managed by harkd",
            "location": self._location,
            "image": config.docker_image,
        }
        if self._ssh_key_ids:
            body["ssh_key_ids"] = self._ssh_key_ids
        if self._os_volume_id:
            body["os_volume_id"] = self._os_volume_id

        # Startup script to run the worker container
        env_flags = (
            f"-e HARKD_WORKER_API_KEY={config.worker_api_key} "
            f"-e HARKD_WORKER_MODEL={config.worker_model}"
        )
        if config.hf_token:
            env_flags += f" -e HARKD_HF_TOKEN={config.hf_token}"
        for key, value in config.extra_env.items():
            env_flags += f" -e {key}={value}"

        body["startup_script"] = (
            f"#!/bin/bash\n"
            f"docker pull {config.docker_image}\n"
            f"docker run -d --gpus all -p 8000:8000 {env_flags} {config.docker_image}\n"
        )
        return body

    @staticmethod
    def _map_status(status: str) -> InfraState:
        mapping = {
            "running": InfraState.RUNNING,
            "deploying": InfraState.PROVISIONING,
            "terminated": InfraState.DORMANT,
            "error": InfraState.DORMANT,
        }
        return mapping.get(status, InfraState.PROVISIONING)
