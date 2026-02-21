"""Koyeb infrastructure provider.

Creates a Koyeb service with GPU instances that natively scale to zero.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from harkd.transcription.infra.provider import (
    InfraProvider,
    InfraProviderConfig,
    InfraState,
    InfraStatus,
)

__all__ = ["KoyebInfraProvider"]

logger = logging.getLogger(__name__)

_KOYEB_API = "https://app.koyeb.com/v1"


class KoyebInfraProvider(InfraProvider):
    """Koyeb infrastructure provider.

    Strategy: create a Koyeb service on first use. Koyeb natively scales
    to 0 instances when idle (no compute cost). Wake-up on first request
    takes ~30-60s.
    """

    def __init__(
        self,
        api_token: str,
        region: str = "fra",
        instance_type: str = "gpu-nvidia-rtx-4000-sff-ada",
        app_name: str = "hark-worker",
        use_native_scale_to_zero: bool = True,
    ):
        self._api_token = api_token
        self._region = region
        self._instance_type = instance_type
        self._app_name = app_name
        self._use_native_scale_to_zero = use_native_scale_to_zero

    def provider_name(self) -> str:
        return "koyeb"

    async def provision(self, config: InfraProviderConfig) -> InfraStatus:
        async with self._client() as client:
            # Check if service already exists
            existing = await self._find_service(client)
            if existing:
                service_id = existing["id"]
                status = existing.get("status", "")

                if status == "PAUSED":
                    logger.info("[koyeb] Resuming paused service %s", service_id)
                    await client.post(f"{_KOYEB_API}/services/{service_id}/resume")
                    await self._wait_for_healthy(client, service_id)

                endpoint_url = self._extract_endpoint(existing)
                return InfraStatus(
                    state=InfraState.RUNNING,
                    provider="koyeb",
                    instance_id=service_id,
                    endpoint_url=endpoint_url,
                    created_at=datetime.now(UTC),
                )

            # Create new service
            logger.info("[koyeb] Creating new service %s", self._app_name)
            service_spec = self._build_service_spec(config)
            resp = await client.post(f"{_KOYEB_API}/services", json=service_spec)
            resp.raise_for_status()
            data = resp.json()
            service_id = data["service"]["id"]

            await self._wait_for_healthy(client, service_id)

            # Re-fetch to get the deployed URL
            resp = await client.get(f"{_KOYEB_API}/services/{service_id}")
            resp.raise_for_status()
            service = resp.json()["service"]
            endpoint_url = self._extract_endpoint(service)

            return InfraStatus(
                state=InfraState.RUNNING,
                provider="koyeb",
                instance_id=service_id,
                endpoint_url=endpoint_url,
                created_at=datetime.now(UTC),
            )

    async def teardown(self, instance_id: str) -> None:
        if self._use_native_scale_to_zero:
            logger.info("[koyeb] Teardown is no-op (native scale-to-zero): %s", instance_id)
            return

        async with self._client() as client:
            logger.info("[koyeb] Pausing service %s", instance_id)
            resp = await client.post(f"{_KOYEB_API}/services/{instance_id}/pause")
            resp.raise_for_status()

    async def destroy(self, instance_id: str) -> None:
        async with self._client() as client:
            logger.info("[koyeb] Deleting service %s", instance_id)
            resp = await client.delete(f"{_KOYEB_API}/services/{instance_id}")
            resp.raise_for_status()

    async def get_status(self, instance_id: str) -> InfraStatus:
        async with self._client() as client:
            resp = await client.get(f"{_KOYEB_API}/services/{instance_id}")
            resp.raise_for_status()
            service = resp.json()["service"]
            state = self._map_status(service.get("status", ""))
            return InfraStatus(
                state=state,
                provider="koyeb",
                instance_id=instance_id,
                endpoint_url=self._extract_endpoint(service),
            )

    async def discover_running(self) -> list[InfraStatus]:
        async with self._client() as client:
            services = await self._list_services(client)
            results = []
            for svc in services:
                status = svc.get("status", "")
                if status not in ("DELETED", "DELETING"):
                    results.append(
                        InfraStatus(
                            state=self._map_status(status),
                            provider="koyeb",
                            instance_id=svc["id"],
                            endpoint_url=self._extract_endpoint(svc),
                        )
                    )
            return results

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._api_token}"},
            timeout=30.0,
        )

    async def _find_service(self, client: httpx.AsyncClient) -> dict | None:
        services = await self._list_services(client)
        for svc in services:
            if svc.get("status") not in ("DELETED", "DELETING"):
                return svc
        return None

    async def _list_services(self, client: httpx.AsyncClient) -> list[dict]:
        resp = await client.get(
            f"{_KOYEB_API}/services", params={"name": self._app_name, "limit": "10"}
        )
        resp.raise_for_status()
        return resp.json().get("services", [])

    async def _wait_for_healthy(
        self, client: httpx.AsyncClient, service_id: str, timeout: int = 300
    ) -> None:
        import asyncio

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            resp = await client.get(f"{_KOYEB_API}/services/{service_id}")
            resp.raise_for_status()
            service = resp.json()["service"]
            status = service.get("status", "")
            if status == "HEALTHY":
                return
            if status in ("ERROR", "DELETED"):
                raise RuntimeError(f"Koyeb service entered {status} state")
            await asyncio.sleep(5)
        raise TimeoutError(f"Koyeb service {service_id} did not become healthy within {timeout}s")

    def _build_service_spec(self, config: InfraProviderConfig) -> dict:
        env_vars = [
            {"key": "HARKD_WORKER_API_KEY", "value": config.worker_api_key},
            {"key": "HARKD_WORKER_MODEL", "value": config.worker_model},
        ]
        if config.hf_token:
            env_vars.append({"key": "HARKD_HF_TOKEN", "value": config.hf_token})
        for key, value in config.extra_env.items():
            env_vars.append({"key": key, "value": value})

        return {
            "definition": {
                "name": self._app_name,
                "type": "WEB",
                "docker": {"image": config.docker_image},
                "instance_types": [{"type": self._instance_type}],
                "regions": [self._region],
                "env": env_vars,
                "scaling": {"min": 0, "max": 1},
                "ports": [{"port": 8000, "protocol": "http"}],
                "routes": [{"path": "/", "port": 8000}],
                "health_checks": [
                    {
                        "http": {"port": 8000, "path": "/health"},
                        "grace_period": 60,
                        "interval": 30,
                        "timeout": 10,
                    }
                ],
            }
        }

    @staticmethod
    def _extract_endpoint(service: dict) -> str | None:
        # Koyeb sets the public URL in active_deployment or computed_route
        for route in service.get("routes", []):
            if route.get("path") == "/":
                return f"https://{service.get('name', '')}.koyeb.app"
        # Fallback: use the service name
        name = service.get("name")
        return f"https://{name}.koyeb.app" if name else None

    @staticmethod
    def _map_status(status: str) -> InfraState:
        mapping = {
            "HEALTHY": InfraState.RUNNING,
            "STARTING": InfraState.PROVISIONING,
            "PAUSING": InfraState.TEARING_DOWN,
            "PAUSED": InfraState.DORMANT,
            "RESUMING": InfraState.PROVISIONING,
            "DELETED": InfraState.DORMANT,
            "DELETING": InfraState.TEARING_DOWN,
        }
        return mapping.get(status, InfraState.PROVISIONING)
