"""Tests for DataCrunchInfraProvider."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from harkd.transcription.infra.datacrunch_infra import DataCrunchInfraProvider
from harkd.transcription.infra.provider import InfraProviderConfig, InfraState


@pytest.fixture
def provider():
    return DataCrunchInfraProvider(
        client_id="dc-client-id",
        client_secret="dc-client-secret",
        instance_type="1L40S.6V",
        location="FIN-01",
        ssh_key_ids=["key-1"],
    )


@pytest.fixture
def infra_config():
    return InfraProviderConfig(
        docker_image="test:latest",
        worker_api_key="secret",
        worker_model="large-v3",
    )


class TestDataCrunchInfraProvider:
    def test_provider_name(self, provider):
        assert provider.provider_name() == "datacrunch"

    @pytest.mark.asyncio
    async def test_provision_creates_instance(self, provider, infra_config):
        responses = [
            # POST /oauth2/token
            httpx.Response(200, json={"access_token": "bearer-tok"}),
            # POST /instances — create
            httpx.Response(200, json={"id": "inst-1", "status": "deploying", "ip": None}),
            # GET /instances/inst-1 — wait for running
            httpx.Response(200, json={"id": "inst-1", "status": "running", "ip": "10.0.0.1"}),
            # GET /instances/inst-1 — get details
            httpx.Response(200, json={"id": "inst-1", "status": "running", "ip": "10.0.0.1"}),
        ]

        call_idx = 0

        async def mock_handler(request):
            nonlocal call_idx
            resp = responses[call_idx]
            call_idx += 1
            return resp

        transport = httpx.MockTransport(mock_handler)

        with (
            patch.object(
                provider,
                "_client",
                return_value=httpx.AsyncClient(transport=transport),
            ),
            patch.object(provider, "_wait_for_health", return_value=None),
        ):
            status = await provider.provision(infra_config)

        assert status.state == InfraState.RUNNING
        assert status.instance_id == "inst-1"
        assert status.endpoint_url == "http://10.0.0.1:8000"

    @pytest.mark.asyncio
    async def test_teardown_calls_destroy(self, provider):
        """DataCrunch has no stop/start — teardown delegates to destroy."""
        provider._access_token = "existing-token"

        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            await provider.teardown("inst-1")

    @pytest.mark.asyncio
    async def test_destroy_deletes_instance(self, provider):
        provider._access_token = "existing-token"

        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            await provider.destroy("inst-1")

    @pytest.mark.asyncio
    async def test_get_status(self, provider):
        provider._access_token = "existing-token"

        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"id": "inst-1", "status": "running", "ip": "10.0.0.1"},
            )
        )
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            status = await provider.get_status("inst-1")

        assert status.state == InfraState.RUNNING
        assert status.endpoint_url == "http://10.0.0.1:8000"

    @pytest.mark.asyncio
    async def test_discover_running(self, provider):
        provider._access_token = "existing-token"

        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "instances": [
                        {
                            "id": "inst-1",
                            "hostname": "hark-worker",
                            "status": "running",
                            "ip": "10.0.0.1",
                            "description": "",
                        },
                        {
                            "id": "inst-2",
                            "hostname": "other",
                            "status": "running",
                            "ip": "10.0.0.2",
                            "description": "",
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
        assert results[0].instance_id == "inst-1"

    @pytest.mark.asyncio
    async def test_oauth2_token_flow(self, provider):
        """Test that provision fetches OAuth2 token first."""
        requests_log = []

        async def mock_handler(request):
            requests_log.append(str(request.url))
            if "oauth2/token" in str(request.url):
                return httpx.Response(200, json={"access_token": "new-token"})
            if request.method == "POST" and "instances" in str(request.url):
                return httpx.Response(
                    200, json={"id": "inst-1", "status": "running", "ip": "1.1.1.1"}
                )
            return httpx.Response(200, json={"id": "inst-1", "status": "running", "ip": "1.1.1.1"})

        transport = httpx.MockTransport(mock_handler)

        with (
            patch.object(
                provider,
                "_client",
                return_value=httpx.AsyncClient(transport=transport),
            ),
            patch.object(provider, "_wait_for_health", return_value=None),
        ):
            await provider.provision(InfraProviderConfig(docker_image="img", worker_api_key="k"))

        assert any("oauth2/token" in url for url in requests_log)

    def test_build_create_body(self, provider, infra_config):
        body = provider._build_create_body(infra_config)
        assert body["instance_type"] == "1L40S.6V"
        assert body["hostname"] == "hark-worker"
        assert body["location"] == "FIN-01"
        assert body["ssh_key_ids"] == ["key-1"]
        assert "startup_script" in body
        assert "HARKD_WORKER_API_KEY=secret" in body["startup_script"]

    def test_map_status(self):
        assert DataCrunchInfraProvider._map_status("running") == InfraState.RUNNING
        assert DataCrunchInfraProvider._map_status("deploying") == InfraState.PROVISIONING
        assert DataCrunchInfraProvider._map_status("terminated") == InfraState.DORMANT
