"""Tests for ManagedBackend — state machine, idle timer, provisioning, orphan recovery."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from harkd.exceptions import InfraProvisioningError
from harkd.transcription.backend import BackendHealth, TranscriptionBackend, TranscriptionRequest
from harkd.transcription.infra.managed import ManagedBackend
from harkd.transcription.infra.provider import (
    InfraProvider,
    InfraProviderConfig,
    InfraState,
    InfraStatus,
)

# --- Fixtures & helpers ---


class MockBackend(TranscriptionBackend):
    """Mock inner backend for testing."""

    def __init__(self):
        self.transcribe_result: dict[str, Any] = {"text": "hello", "segments": []}
        self.closed = False

    async def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]:
        return self.transcribe_result

    async def health_check(self) -> BackendHealth:
        return BackendHealth(healthy=True, details={"mock": True})

    async def close(self) -> None:
        self.closed = True


class MockInfraProvider(InfraProvider):
    """Mock infrastructure provider for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self.provision_result = InfraStatus(
            state=InfraState.RUNNING,
            provider=name,
            instance_id="inst-123",
            endpoint_url="http://mock:8000",
            created_at=datetime.now(UTC),
        )
        self.provision_called = 0
        self.teardown_called = 0
        self.destroy_called = 0
        self.get_status_result: InfraStatus | None = None

    async def provision(self, config: InfraProviderConfig) -> InfraStatus:
        self.provision_called += 1
        return self.provision_result

    async def teardown(self, instance_id: str) -> None:
        self.teardown_called += 1

    async def destroy(self, instance_id: str) -> None:
        self.destroy_called += 1

    async def get_status(self, instance_id: str) -> InfraStatus:
        if self.get_status_result:
            return self.get_status_result
        return self.provision_result

    async def discover_running(self) -> list[InfraStatus]:
        return []

    def provider_name(self) -> str:
        return self._name


@pytest.fixture
def infra_config():
    return InfraProviderConfig(
        docker_image="test:latest",
        worker_api_key="secret",
    )


@pytest.fixture
def mock_provider():
    return MockInfraProvider()


@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def backend_factory(mock_backend):
    def factory(endpoint_url: str) -> TranscriptionBackend:
        return mock_backend

    return factory


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "infra_state_mock.json"


@pytest.fixture
def managed(mock_provider, infra_config, backend_factory, state_file):
    return ManagedBackend(
        infra_provider=mock_provider,
        infra_config=infra_config,
        backend_factory=backend_factory,
        idle_timeout=0,  # Disable idle timer for most tests
        state_file=state_file,
    )


def _make_request() -> TranscriptionRequest:
    return TranscriptionRequest(
        audio_path=Path("/tmp/test.wav"),
        model_name="base",
        language=None,
        word_timestamps=False,
        diarize=False,
        hf_token=None,
        beam_size=3,
        batch_size=16,
        vad_onset=0.5,
        vad_offset=0.363,
        vad_method="pyannote",
        timeout=60,
    )


# --- Tests ---


class TestStateTransitions:
    """Test state machine transitions."""

    def test_initial_state_is_dormant(self, managed):
        assert managed.state == InfraState.DORMANT

    @pytest.mark.asyncio
    async def test_transcribe_provisions_and_runs(self, managed, mock_provider):
        result = await managed.transcribe(_make_request())
        assert result == {"text": "hello", "segments": []}
        assert managed.state == InfraState.RUNNING
        assert mock_provider.provision_called == 1

    @pytest.mark.asyncio
    async def test_second_transcribe_does_not_reprovision(self, managed, mock_provider):
        await managed.transcribe(_make_request())
        await managed.transcribe(_make_request())
        assert mock_provider.provision_called == 1

    @pytest.mark.asyncio
    async def test_close_tears_down(self, managed, mock_provider, mock_backend):
        await managed.transcribe(_make_request())
        assert managed.state == InfraState.RUNNING
        await managed.close()
        assert managed.state == InfraState.DORMANT
        assert mock_provider.teardown_called == 1
        assert mock_backend.closed

    @pytest.mark.asyncio
    async def test_close_when_dormant_is_noop(self, managed, mock_provider):
        await managed.close()
        assert mock_provider.teardown_called == 0


class TestProvisioning:
    """Test provisioning behavior."""

    @pytest.mark.asyncio
    async def test_provisioning_failure_returns_to_dormant(
        self, mock_provider, infra_config, backend_factory, state_file
    ):
        async def fail_provision(config):
            raise RuntimeError("Cloud API down")

        mock_provider.provision = fail_provision  # type: ignore[assignment]

        managed = ManagedBackend(
            infra_provider=mock_provider,
            infra_config=infra_config,
            backend_factory=backend_factory,
            state_file=state_file,
        )

        with pytest.raises(InfraProvisioningError, match="Cloud API down"):
            await managed.transcribe(_make_request())
        assert managed.state == InfraState.DORMANT

    @pytest.mark.asyncio
    async def test_provisioning_timeout(
        self, mock_provider, infra_config, backend_factory, state_file
    ):
        async def slow_provision(config):
            await asyncio.sleep(10)
            return mock_provider.provision_result

        mock_provider.provision = slow_provision  # type: ignore[assignment]

        managed = ManagedBackend(
            infra_provider=mock_provider,
            infra_config=infra_config,
            backend_factory=backend_factory,
            provisioning_timeout=1,
            state_file=state_file,
        )

        with pytest.raises(InfraProvisioningError, match="timed out"):
            await managed.transcribe(_make_request())
        assert managed.state == InfraState.DORMANT

    @pytest.mark.asyncio
    async def test_no_endpoint_url_raises(
        self, mock_provider, infra_config, backend_factory, state_file
    ):
        mock_provider.provision_result = InfraStatus(
            state=InfraState.RUNNING,
            provider="mock",
            instance_id="inst-1",
            endpoint_url=None,
        )

        managed = ManagedBackend(
            infra_provider=mock_provider,
            infra_config=infra_config,
            backend_factory=backend_factory,
            state_file=state_file,
        )

        with pytest.raises(InfraProvisioningError, match="no endpoint URL"):
            await managed.transcribe(_make_request())

    @pytest.mark.asyncio
    async def test_concurrent_transcribe_serializes_provisioning(
        self, mock_provider, infra_config, backend_factory, state_file
    ):
        provision_count = 0
        original_result = mock_provider.provision_result

        async def counting_provision(config):
            nonlocal provision_count
            provision_count += 1
            await asyncio.sleep(0.1)
            return original_result

        mock_provider.provision = counting_provision  # type: ignore[assignment]

        managed = ManagedBackend(
            infra_provider=mock_provider,
            infra_config=infra_config,
            backend_factory=backend_factory,
            state_file=state_file,
        )

        results = await asyncio.gather(
            managed.transcribe(_make_request()),
            managed.transcribe(_make_request()),
            managed.transcribe(_make_request()),
        )
        assert len(results) == 3
        assert provision_count == 1


class TestHealthCheck:
    """Test health check behavior."""

    @pytest.mark.asyncio
    async def test_dormant_is_healthy(self, managed):
        health = await managed.health_check()
        assert health.healthy is True
        assert health.details["state"] == "dormant"
        assert health.details["managed"] is True

    @pytest.mark.asyncio
    async def test_running_delegates_to_inner(self, managed):
        await managed.transcribe(_make_request())
        health = await managed.health_check()
        assert health.healthy is True
        assert health.details["state"] == "running"
        assert health.details["managed"] is True


class TestPreWarm:
    """Test pre-warm behavior."""

    @pytest.mark.asyncio
    async def test_pre_warm_starts_provisioning(self, managed, mock_provider):
        await managed.pre_warm()
        # Give the background task time to run
        await asyncio.sleep(0.1)
        assert mock_provider.provision_called == 1
        assert managed.state == InfraState.RUNNING

    @pytest.mark.asyncio
    async def test_pre_warm_when_running_is_noop(self, managed, mock_provider):
        await managed.transcribe(_make_request())
        assert mock_provider.provision_called == 1
        await managed.pre_warm()
        await asyncio.sleep(0.1)
        assert mock_provider.provision_called == 1


class TestIdleTimer:
    """Test idle timer teardown behavior."""

    @pytest.mark.asyncio
    async def test_idle_timeout_tears_down(
        self, mock_provider, infra_config, backend_factory, state_file
    ):
        managed = ManagedBackend(
            infra_provider=mock_provider,
            infra_config=infra_config,
            backend_factory=backend_factory,
            idle_timeout=1,
            state_file=state_file,
        )

        await managed.transcribe(_make_request())
        assert managed.state == InfraState.RUNNING

        # Wait for idle timer
        await asyncio.sleep(1.5)
        assert managed.state == InfraState.DORMANT
        assert mock_provider.teardown_called == 1

    @pytest.mark.asyncio
    async def test_transcribe_resets_idle_timer(
        self, mock_provider, infra_config, backend_factory, state_file
    ):
        managed = ManagedBackend(
            infra_provider=mock_provider,
            infra_config=infra_config,
            backend_factory=backend_factory,
            idle_timeout=2,
            state_file=state_file,
        )

        await managed.transcribe(_make_request())
        await asyncio.sleep(1)
        # This should reset the timer
        await managed.transcribe(_make_request())
        await asyncio.sleep(1.5)
        # Should still be running (timer was reset)
        assert managed.state == InfraState.RUNNING
        # Clean up
        await managed.close()


class TestStatePersistence:
    """Test state file persistence."""

    @pytest.mark.asyncio
    async def test_state_saved_on_provision(self, managed, state_file):
        await managed.transcribe(_make_request())
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["provider"] == "mock"
        assert data["instance_id"] == "inst-123"
        assert data["endpoint_url"] == "http://mock:8000"

    @pytest.mark.asyncio
    async def test_state_cleared_on_teardown(self, managed, state_file):
        await managed.transcribe(_make_request())
        assert state_file.exists()
        await managed.close()
        assert not state_file.exists()


class TestOrphanRecovery:
    """Test orphan recovery behavior."""

    @pytest.mark.asyncio
    async def test_no_state_file_is_noop(self, managed, state_file):
        assert not state_file.exists()
        await managed.recover_orphans()
        assert managed.state == InfraState.DORMANT

    @pytest.mark.asyncio
    async def test_adopts_running_orphan(
        self, mock_provider, infra_config, backend_factory, state_file
    ):
        # Write orphan state
        state_file.write_text(
            json.dumps(
                {
                    "provider": "mock",
                    "instance_id": "orphan-1",
                    "endpoint_url": "http://orphan:8000",
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
        )

        mock_provider.get_status_result = InfraStatus(
            state=InfraState.RUNNING,
            provider="mock",
            instance_id="orphan-1",
            endpoint_url="http://orphan:8000",
            created_at=datetime.now(UTC),
        )

        managed = ManagedBackend(
            infra_provider=mock_provider,
            infra_config=infra_config,
            backend_factory=backend_factory,
            state_file=state_file,
        )
        await managed.recover_orphans()

        assert managed.state == InfraState.RUNNING
        assert managed._instance_id == "orphan-1"
        # Clean up
        await managed.close()

    @pytest.mark.asyncio
    async def test_tears_down_non_running_orphan(
        self, mock_provider, infra_config, backend_factory, state_file
    ):
        state_file.write_text(
            json.dumps(
                {
                    "provider": "mock",
                    "instance_id": "dead-1",
                    "endpoint_url": "http://dead:8000",
                }
            )
        )

        mock_provider.get_status_result = InfraStatus(
            state=InfraState.DORMANT,
            provider="mock",
            instance_id="dead-1",
        )

        managed = ManagedBackend(
            infra_provider=mock_provider,
            infra_config=infra_config,
            backend_factory=backend_factory,
            state_file=state_file,
        )
        await managed.recover_orphans()

        assert managed.state == InfraState.DORMANT
        assert mock_provider.teardown_called == 1
        assert not state_file.exists()
