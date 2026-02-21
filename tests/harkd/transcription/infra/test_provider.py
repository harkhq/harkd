"""Tests for infrastructure provider data types."""

from datetime import UTC, datetime

from harkd.transcription.infra.provider import (
    InfraProviderConfig,
    InfraState,
    InfraStatus,
)


class TestInfraState:
    """Tests for InfraState enum."""

    def test_values(self):
        assert InfraState.DORMANT == "dormant"
        assert InfraState.PROVISIONING == "provisioning"
        assert InfraState.RUNNING == "running"
        assert InfraState.TEARING_DOWN == "tearing_down"

    def test_all_states_exist(self):
        assert len(InfraState) == 4


class TestInfraStatus:
    """Tests for InfraStatus dataclass."""

    def test_minimal(self):
        status = InfraStatus(state=InfraState.DORMANT, provider="koyeb")
        assert status.state == InfraState.DORMANT
        assert status.provider == "koyeb"
        assert status.instance_id is None
        assert status.endpoint_url is None
        assert status.created_at is None
        assert status.last_activity is None

    def test_full(self):
        now = datetime.now(UTC)
        status = InfraStatus(
            state=InfraState.RUNNING,
            provider="scaleway",
            instance_id="srv-123",
            endpoint_url="http://1.2.3.4:8000",
            created_at=now,
            last_activity=now,
        )
        assert status.instance_id == "srv-123"
        assert status.endpoint_url == "http://1.2.3.4:8000"
        assert status.created_at == now


class TestInfraProviderConfig:
    """Tests for InfraProviderConfig dataclass."""

    def test_minimal(self):
        config = InfraProviderConfig(
            docker_image="ghcr.io/test/worker:latest",
            worker_api_key="secret",
        )
        assert config.docker_image == "ghcr.io/test/worker:latest"
        assert config.worker_api_key == "secret"
        assert config.worker_model == "large-v3"
        assert config.hf_token is None
        assert config.extra_env == {}

    def test_full(self):
        config = InfraProviderConfig(
            docker_image="img:v1",
            worker_api_key="key",
            worker_model="small",
            hf_token="hf_test",
            extra_env={"FOO": "bar"},
        )
        assert config.worker_model == "small"
        assert config.hf_token == "hf_test"
        assert config.extra_env == {"FOO": "bar"}
