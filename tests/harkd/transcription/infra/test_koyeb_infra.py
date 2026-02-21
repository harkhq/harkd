"""Tests for KoyebInfraProvider."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from harkd.transcription.infra.koyeb_infra import KoyebInfraProvider
from harkd.transcription.infra.provider import InfraProviderConfig, InfraState


@pytest.fixture
def provider():
    return KoyebInfraProvider(
        api_token="test-token",
        region="fra",
        instance_type="gpu-test",
        app_name="hark-worker",
    )


@pytest.fixture
def infra_config():
    return InfraProviderConfig(
        docker_image="test:latest",
        worker_api_key="secret",
        worker_model="large-v3",
        hf_token="hf_test",
    )


class TestKoyebInfraProvider:
    def test_provider_name(self, provider):
        assert provider.provider_name() == "koyeb"

    @pytest.mark.asyncio
    async def test_provision_creates_new_service(self, provider, infra_config):
        mock_responses = [
            # GET /services?name=hark-worker — no existing
            httpx.Response(200, json={"services": []}),
            # GET /apps?name=hark-worker — no app yet
            httpx.Response(200, json={"apps": []}),
            # POST /apps — create app
            httpx.Response(
                200,
                json={"app": {"id": "app-abc", "name": "hark-worker"}},
            ),
            # POST /services — create
            httpx.Response(
                200,
                json={
                    "service": {
                        "id": "svc-abc",
                        "name": "hark-worker",
                        "status": "STARTING",
                    }
                },
            ),
            # GET /services/svc-abc — wait for healthy
            httpx.Response(
                200,
                json={
                    "service": {
                        "id": "svc-abc",
                        "name": "hark-worker",
                        "status": "HEALTHY",
                        "routes": [{"path": "/"}],
                    }
                },
            ),
            # GET /services/svc-abc — re-fetch for endpoint
            httpx.Response(
                200,
                json={
                    "service": {
                        "id": "svc-abc",
                        "name": "hark-worker",
                        "status": "HEALTHY",
                        "routes": [{"path": "/"}],
                    }
                },
            ),
        ]

        call_idx = 0

        async def mock_handler(request):
            nonlocal call_idx
            resp = mock_responses[call_idx]
            call_idx += 1
            return resp

        transport = httpx.MockTransport(mock_handler)
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            status = await provider.provision(infra_config)

        assert status.state == InfraState.RUNNING
        assert status.instance_id == "svc-abc"
        assert status.endpoint_url == "https://hark-worker.koyeb.app"

    @pytest.mark.asyncio
    async def test_provision_resumes_paused_service(self, provider, infra_config):
        mock_responses = [
            # GET /services?name=hark-worker — found paused
            httpx.Response(
                200,
                json={
                    "services": [
                        {
                            "id": "svc-paused",
                            "name": "hark-worker",
                            "status": "PAUSED",
                            "routes": [{"path": "/"}],
                        }
                    ]
                },
            ),
            # POST /services/svc-paused/resume
            httpx.Response(200, json={}),
            # GET /services/svc-paused — wait for healthy
            httpx.Response(
                200,
                json={
                    "service": {
                        "id": "svc-paused",
                        "name": "hark-worker",
                        "status": "HEALTHY",
                        "routes": [{"path": "/"}],
                    }
                },
            ),
        ]

        call_idx = 0

        async def mock_handler(request):
            nonlocal call_idx
            resp = mock_responses[call_idx]
            call_idx += 1
            return resp

        transport = httpx.MockTransport(mock_handler)
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            status = await provider.provision(infra_config)

        assert status.state == InfraState.RUNNING
        assert status.instance_id == "svc-paused"

    @pytest.mark.asyncio
    async def test_teardown_noop_with_scale_to_zero(self, provider):
        """Teardown is no-op when native scale-to-zero is enabled (default)."""
        # Should not make any HTTP calls
        await provider.teardown("svc-123")

    @pytest.mark.asyncio
    async def test_teardown_pauses_without_scale_to_zero(self):
        provider = KoyebInfraProvider(
            api_token="test-token",
            use_native_scale_to_zero=False,
        )

        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            await provider.teardown("svc-123")

    @pytest.mark.asyncio
    async def test_destroy_deletes_service(self, provider):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            await provider.destroy("svc-123")

    @pytest.mark.asyncio
    async def test_get_status(self, provider):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "service": {
                        "id": "svc-1",
                        "name": "hark-worker",
                        "status": "HEALTHY",
                        "routes": [{"path": "/"}],
                    }
                },
            )
        )
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            status = await provider.get_status("svc-1")

        assert status.state == InfraState.RUNNING
        assert status.endpoint_url == "https://hark-worker.koyeb.app"

    @pytest.mark.asyncio
    async def test_discover_running(self, provider):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "services": [
                        {
                            "id": "svc-1",
                            "name": "hark-worker",
                            "status": "HEALTHY",
                            "routes": [{"path": "/"}],
                        },
                        {
                            "id": "svc-2",
                            "name": "hark-worker",
                            "status": "DELETED",
                        },
                    ]
                },
            )
        )
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            results = await provider.discover_running()

        assert len(results) == 1
        assert results[0].instance_id == "svc-1"

    def test_build_service_spec(self, provider, infra_config):
        spec = provider._build_service_spec(infra_config, app_id="app-abc")
        assert spec["app_id"] == "app-abc"
        defn = spec["definition"]
        assert defn["docker"]["image"] == "test:latest"
        assert defn["instance_types"][0]["type"] == "gpu-test"
        assert defn["regions"] == ["fra"]
        assert defn["scalings"] == [{"min": 1, "max": 1}]
        env_keys = [e["key"] for e in defn["env"]]
        assert "HARKD_WORKER_API_KEY" in env_keys
        assert "HARKD_WORKER_MODEL" in env_keys
        assert "HARKD_HF_TOKEN" in env_keys

    @pytest.mark.asyncio
    async def test_ensure_app_exists_already_exists(self, provider):
        """When the app already exists, no POST /apps is made."""
        requests_made = []

        async def mock_handler(request):
            requests_made.append((request.method, str(request.url)))
            return httpx.Response(
                200,
                json={"apps": [{"id": "app-abc", "name": "hark-worker"}]},
            )

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            app_id = await provider._ensure_app_exists(client)

        assert app_id == "app-abc"
        assert len(requests_made) == 1
        assert requests_made[0][0] == "GET"

    def test_map_status(self):
        assert KoyebInfraProvider._map_status("HEALTHY") == InfraState.RUNNING
        assert KoyebInfraProvider._map_status("STARTING") == InfraState.PROVISIONING
        assert KoyebInfraProvider._map_status("PAUSED") == InfraState.DORMANT
        assert KoyebInfraProvider._map_status("DELETING") == InfraState.TEARING_DOWN
