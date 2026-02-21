"""Scaleway infrastructure provider.

Creates GPU instances that are stopped/started between usage cycles.
Stopped instances preserve disk and cost almost nothing (block storage only).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import UTC, datetime

import httpx

from harkd.transcription.infra.provider import (
    InfraProvider,
    InfraProviderConfig,
    InfraState,
    InfraStatus,
)

__all__ = ["ScalewayInfraProvider"]

logger = logging.getLogger(__name__)

_SCW_API = "https://api.scaleway.com/instance/v1"
_TAG = "hark-worker"


class ScalewayInfraProvider(InfraProvider):
    """Scaleway infrastructure provider.

    Strategy: create instance once, stop/start around usage. Stopped
    instances cost almost nothing (only block storage ~EUR 0.01/GB/month).
    """

    def __init__(
        self,
        secret_key: str,
        organization_id: str,
        project_id: str,
        zone: str = "fr-par-2",
        instance_type: str = "L4-1-24G",
        image_id: str | None = None,
    ):
        self._secret_key = secret_key
        self._organization_id = organization_id
        self._project_id = project_id
        self._zone = zone
        self._instance_type = instance_type
        self._image_id = image_id

    def provider_name(self) -> str:
        return "scaleway"

    async def provision(self, config: InfraProviderConfig) -> InfraStatus:
        async with self._client() as client:
            # Check for existing tagged instance
            existing = await self._find_instance(client)

            if existing:
                server_id = existing["id"]
                state = existing.get("state", "")

                if state == "stopped":
                    logger.info("[scaleway] Starting stopped instance %s", server_id)
                    await self._server_action(client, server_id, "poweron")
                    await self._wait_for_running(client, server_id)
                elif state == "running":
                    logger.info("[scaleway] Instance %s already running", server_id)
                else:
                    logger.info("[scaleway] Instance %s in state %s, waiting...", server_id, state)
                    await self._wait_for_running(client, server_id)

                server = await self._get_server(client, server_id)
                ip = self._extract_ip(server)
                if ip:
                    await self._wait_for_health(ip)

                return InfraStatus(
                    state=InfraState.RUNNING,
                    provider="scaleway",
                    instance_id=server_id,
                    endpoint_url=f"http://{ip}:8000" if ip else None,
                    created_at=datetime.now(UTC),
                )

            # Create new instance
            logger.info("[scaleway] Creating new instance (type=%s)", self._instance_type)
            user_data = self._build_cloud_init(config)
            create_body = self._build_create_body(config, user_data)
            resp = await client.post(f"{_SCW_API}/zones/{self._zone}/servers", json=create_body)
            resp.raise_for_status()
            server = resp.json()["server"]
            server_id = server["id"]

            # Start the instance
            await self._server_action(client, server_id, "poweron")
            await self._wait_for_running(client, server_id)

            server = await self._get_server(client, server_id)
            ip = self._extract_ip(server)
            if ip:
                await self._wait_for_health(ip)

            return InfraStatus(
                state=InfraState.RUNNING,
                provider="scaleway",
                instance_id=server_id,
                endpoint_url=f"http://{ip}:8000" if ip else None,
                created_at=datetime.now(UTC),
            )

    async def teardown(self, instance_id: str) -> None:
        async with self._client() as client:
            logger.info("[scaleway] Stopping instance %s", instance_id)
            await self._server_action(client, instance_id, "poweroff")

    async def destroy(self, instance_id: str) -> None:
        async with self._client() as client:
            # Stop first if running
            server = await self._get_server(client, instance_id)
            if server.get("state") == "running":
                await self._server_action(client, instance_id, "poweroff")
                await self._wait_for_stopped(client, instance_id)

            # Collect volume IDs before deletion
            volumes = server.get("volumes", {})
            volume_ids = [v["id"] for v in volumes.values() if "id" in v]

            # Delete server
            logger.info("[scaleway] Deleting instance %s", instance_id)
            resp = await client.delete(f"{_SCW_API}/zones/{self._zone}/servers/{instance_id}")
            resp.raise_for_status()

            # Delete volumes
            for vol_id in volume_ids:
                logger.info("[scaleway] Deleting volume %s", vol_id)
                try:
                    resp = await client.delete(f"{_SCW_API}/zones/{self._zone}/volumes/{vol_id}")
                    resp.raise_for_status()
                except httpx.HTTPStatusError:
                    logger.warning("[scaleway] Failed to delete volume %s", vol_id)

    async def get_status(self, instance_id: str) -> InfraStatus:
        async with self._client() as client:
            server = await self._get_server(client, instance_id)
            state = self._map_state(server.get("state", ""))
            ip = self._extract_ip(server)
            return InfraStatus(
                state=state,
                provider="scaleway",
                instance_id=instance_id,
                endpoint_url=f"http://{ip}:8000" if ip else None,
            )

    async def discover_running(self) -> list[InfraStatus]:
        async with self._client() as client:
            resp = await client.get(
                f"{_SCW_API}/zones/{self._zone}/servers",
                params={"tags": _TAG},
            )
            resp.raise_for_status()
            servers = resp.json().get("servers", [])
            results = []
            for srv in servers:
                state = srv.get("state", "")
                if state in ("running", "starting"):
                    ip = self._extract_ip(srv)
                    results.append(
                        InfraStatus(
                            state=self._map_state(state),
                            provider="scaleway",
                            instance_id=srv["id"],
                            endpoint_url=f"http://{ip}:8000" if ip else None,
                        )
                    )
            return results

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"X-Auth-Token": self._secret_key},
            timeout=30.0,
        )

    async def _find_instance(self, client: httpx.AsyncClient) -> dict | None:
        resp = await client.get(
            f"{_SCW_API}/zones/{self._zone}/servers",
            params={"tags": _TAG},
        )
        resp.raise_for_status()
        servers = resp.json().get("servers", [])
        for srv in servers:
            if srv.get("state") not in ("terminated",):
                return srv
        return None

    async def _get_server(self, client: httpx.AsyncClient, server_id: str) -> dict:
        resp = await client.get(f"{_SCW_API}/zones/{self._zone}/servers/{server_id}")
        resp.raise_for_status()
        return resp.json()["server"]

    async def _server_action(self, client: httpx.AsyncClient, server_id: str, action: str) -> None:
        resp = await client.post(
            f"{_SCW_API}/zones/{self._zone}/servers/{server_id}/action",
            json={"action": action},
        )
        resp.raise_for_status()

    async def _wait_for_running(
        self, client: httpx.AsyncClient, server_id: str, timeout: int = 300
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            server = await self._get_server(client, server_id)
            if server.get("state") == "running":
                return
            await asyncio.sleep(5)
        raise TimeoutError(f"Scaleway server {server_id} did not reach running state")

    async def _wait_for_stopped(
        self, client: httpx.AsyncClient, server_id: str, timeout: int = 120
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            server = await self._get_server(client, server_id)
            if server.get("state") == "stopped":
                return
            await asyncio.sleep(5)
        raise TimeoutError(f"Scaleway server {server_id} did not stop")

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
                await asyncio.sleep(5)
        logger.warning("[scaleway] Health check timed out for %s, proceeding anyway", ip)

    def _build_cloud_init(self, config: InfraProviderConfig) -> str:
        env_flags = (
            f"-e HARKD_WORKER_API_KEY={config.worker_api_key} "
            f"-e HARKD_WORKER_MODEL={config.worker_model}"
        )
        if config.hf_token:
            env_flags += f" -e HARKD_HF_TOKEN={config.hf_token}"
        for key, value in config.extra_env.items():
            env_flags += f" -e {key}={value}"

        cloud_init = f"""#cloud-config
runcmd:
  - docker pull {config.docker_image}
  - docker run -d --gpus all -p 8000:8000 {env_flags} {config.docker_image}
"""
        return base64.b64encode(cloud_init.encode()).decode()

    def _build_create_body(self, config: InfraProviderConfig, user_data_b64: str) -> dict:
        body: dict = {
            "name": "hark-worker",
            "commercial_type": self._instance_type,
            "organization": self._organization_id,
            "project": self._project_id,
            "tags": [_TAG],
            "dynamic_ip_required": True,
        }
        if self._image_id:
            body["image"] = self._image_id
        # Cloud-init via user_data
        body["user_data"] = user_data_b64
        return body

    @staticmethod
    def _extract_ip(server: dict) -> str | None:
        public_ip = server.get("public_ip")
        if public_ip and isinstance(public_ip, dict):
            return public_ip.get("address")
        public_ips = server.get("public_ips", [])
        if public_ips:
            return public_ips[0].get("address")
        return None

    @staticmethod
    def _map_state(state: str) -> InfraState:
        mapping = {
            "running": InfraState.RUNNING,
            "starting": InfraState.PROVISIONING,
            "stopping": InfraState.TEARING_DOWN,
            "stopped": InfraState.DORMANT,
            "terminated": InfraState.DORMANT,
        }
        return mapping.get(state, InfraState.PROVISIONING)
