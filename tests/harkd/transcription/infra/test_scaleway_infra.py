"""Tests for ScalewayInfraProvider."""

from __future__ import annotations

import base64
from unittest.mock import patch

import httpx
import pytest

from harkd.transcription.infra.provider import InfraProviderConfig, InfraState
from harkd.transcription.infra.scaleway_infra import ScalewayInfraProvider


@pytest.fixture
def provider():
    return ScalewayInfraProvider(
        secret_key="scw-secret",
        organization_id="org-1",
        project_id="proj-1",
        zone="fr-par-2",
        instance_type="L4-1-24G",
    )


@pytest.fixture
def infra_config():
    return InfraProviderConfig(
        docker_image="test:latest",
        worker_api_key="secret",
        worker_model="large-v3",
        hf_token="hf_test",
    )


class TestScalewayInfraProvider:
    def test_provider_name(self, provider):
        assert provider.provider_name() == "scaleway"

    @pytest.mark.asyncio
    async def test_provision_creates_new_instance(self, provider, infra_config):
        responses = [
            # GET /servers?tags=hark-worker — no existing
            httpx.Response(200, json={"servers": []}),
            # POST /servers — create
            httpx.Response(
                200,
                json={
                    "server": {
                        "id": "srv-1",
                        "state": "stopped",
                        "public_ip": {"address": "1.2.3.4"},
                    }
                },
            ),
            # POST /servers/srv-1/action — poweron
            httpx.Response(200, json={}),
            # GET /servers/srv-1 — wait for running
            httpx.Response(
                200,
                json={
                    "server": {
                        "id": "srv-1",
                        "state": "running",
                        "public_ip": {"address": "1.2.3.4"},
                    }
                },
            ),
            # GET /servers/srv-1 — get details
            httpx.Response(
                200,
                json={
                    "server": {
                        "id": "srv-1",
                        "state": "running",
                        "public_ip": {"address": "1.2.3.4"},
                    }
                },
            ),
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
        assert status.instance_id == "srv-1"
        assert status.endpoint_url == "http://1.2.3.4:8000"

    @pytest.mark.asyncio
    async def test_provision_starts_stopped_instance(self, provider, infra_config):
        responses = [
            # GET /servers?tags=hark-worker — found stopped
            httpx.Response(
                200,
                json={
                    "servers": [
                        {
                            "id": "srv-stopped",
                            "state": "stopped",
                            "public_ip": {"address": "5.6.7.8"},
                        }
                    ]
                },
            ),
            # POST /servers/srv-stopped/action — poweron
            httpx.Response(200, json={}),
            # GET /servers/srv-stopped — wait for running
            httpx.Response(
                200,
                json={
                    "server": {
                        "id": "srv-stopped",
                        "state": "running",
                        "public_ip": {"address": "5.6.7.8"},
                    }
                },
            ),
            # GET /servers/srv-stopped — get details
            httpx.Response(
                200,
                json={
                    "server": {
                        "id": "srv-stopped",
                        "state": "running",
                        "public_ip": {"address": "5.6.7.8"},
                    }
                },
            ),
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

        assert status.instance_id == "srv-stopped"
        assert status.endpoint_url == "http://5.6.7.8:8000"

    @pytest.mark.asyncio
    async def test_teardown_powers_off(self, provider):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            await provider.teardown("srv-1")

    @pytest.mark.asyncio
    async def test_get_status(self, provider):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "server": {
                        "id": "srv-1",
                        "state": "running",
                        "public_ip": {"address": "1.2.3.4"},
                    }
                },
            )
        )
        with patch.object(
            provider,
            "_client",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            status = await provider.get_status("srv-1")

        assert status.state == InfraState.RUNNING
        assert status.endpoint_url == "http://1.2.3.4:8000"

    @pytest.mark.asyncio
    async def test_discover_running(self, provider):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "servers": [
                        {
                            "id": "srv-1",
                            "state": "running",
                            "public_ip": {"address": "1.1.1.1"},
                        },
                        {"id": "srv-2", "state": "stopped"},
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
        assert results[0].instance_id == "srv-1"

    def test_build_cloud_init(self, provider, infra_config):
        user_data_b64 = provider._build_cloud_init(infra_config)
        user_data = base64.b64decode(user_data_b64).decode()
        assert "#cloud-config" in user_data
        assert "docker pull test:latest" in user_data
        assert "HARKD_WORKER_API_KEY=secret" in user_data
        assert "HARKD_HF_TOKEN=hf_test" in user_data

    def test_extract_ip_public_ip_dict(self):
        server = {"public_ip": {"address": "1.2.3.4"}}
        assert ScalewayInfraProvider._extract_ip(server) == "1.2.3.4"

    def test_extract_ip_public_ips_list(self):
        server = {"public_ips": [{"address": "5.6.7.8"}]}
        assert ScalewayInfraProvider._extract_ip(server) == "5.6.7.8"

    def test_extract_ip_none(self):
        assert ScalewayInfraProvider._extract_ip({}) is None

    def test_map_state(self):
        assert ScalewayInfraProvider._map_state("running") == InfraState.RUNNING
        assert ScalewayInfraProvider._map_state("starting") == InfraState.PROVISIONING
        assert ScalewayInfraProvider._map_state("stopped") == InfraState.DORMANT
        assert ScalewayInfraProvider._map_state("stopping") == InfraState.TEARING_DOWN
