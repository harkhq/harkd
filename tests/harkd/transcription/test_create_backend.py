"""Tests for the create_backend factory function."""

import pytest

from harkd.config import (
    DataCrunchInfraSettings,
    HarkdSettings,
    InfraSettings,
    KoyebInfraSettings,
    KoyebProviderSettings,
    ScalewayInfraSettings,
    ScalewayProviderSettings,
    TranscriptionSettings,
    VerdaProviderSettings,
)
from harkd.transcription import (
    KoyebBackend,
    LocalBackend,
    RemoteBackend,
    ScalewayBackend,
    VerdaBackend,
    create_backend,
)
from harkd.transcription.infra.managed import ManagedBackend


class TestCreateBackend:
    """Tests for backend factory."""

    def test_default_creates_local(self):
        """Test default config creates LocalBackend."""
        config = HarkdSettings()
        backend = create_backend(config)
        assert isinstance(backend, LocalBackend)

    def test_explicit_local(self):
        """Test explicit local backend config."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(backend="local"),
        )
        backend = create_backend(config)
        assert isinstance(backend, LocalBackend)

    def test_remote_backend(self):
        """Test remote backend creation."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="remote",
                endpoint_url="http://192.168.178.20:8000",
                worker_api_key="wk-secret",
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, RemoteBackend)
        assert type(backend) is RemoteBackend  # exactly RemoteBackend, not a subclass

    def test_remote_backend_passes_config_through(self):
        """Test remote backend receives endpoint, key, timeout, retries from config."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="remote",
                endpoint_url="http://192.168.178.20:8000/",
                worker_api_key="wk-secret",
                remote_timeout=1800,
                max_retries=3,
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, RemoteBackend)
        assert backend._endpoint == "http://192.168.178.20:8000"  # trailing slash stripped
        assert backend._worker_api_key == "wk-secret"
        assert backend._max_retries == 3

    def test_remote_backend_without_worker_api_key(self):
        """Test remote backend works without worker_api_key (no auth header)."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="remote",
                endpoint_url="http://192.168.178.20:8000",
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, RemoteBackend)
        assert backend._worker_api_key is None
        assert "Authorization" not in backend._build_headers()

    def test_remote_missing_endpoint_url_raises(self):
        """Test remote backend without endpoint_url raises ValueError."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="remote",
            ),
        )
        with pytest.raises(ValueError, match="endpoint_url"):
            create_backend(config)

    def test_remote_with_managed_true_ignores_managed(self):
        """Test remote backend with managed=True still creates plain RemoteBackend."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="remote",
                managed=True,
                endpoint_url="http://192.168.178.20:8000",
                worker_api_key="wk-secret",
            ),
        )
        backend = create_backend(config)
        assert type(backend) is RemoteBackend

    def test_koyeb_backend(self):
        """Test Koyeb backend creation."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="koyeb",
                endpoint_url="https://hark.koyeb.app",
                worker_api_key="wk-secret",
                koyeb=KoyebProviderSettings(token="koyeb-token"),
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, KoyebBackend)

    def test_verda_backend(self):
        """Test Verda backend creation."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="verda",
                endpoint_url="https://containers.datacrunch.io/hark",
                verda=VerdaProviderSettings(api_key="dc_inf_test"),
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, VerdaBackend)

    def test_scaleway_backend(self):
        """Test Scaleway backend creation."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="scaleway",
                endpoint_url="https://gpu.scaleway.com",
                scaleway=ScalewayProviderSettings(api_key="scw-key"),
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, ScalewayBackend)

    def test_scaleway_without_provider_config(self):
        """Test Scaleway backend works without provider-specific config."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="scaleway",
                endpoint_url="https://self-hosted.example.com",
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, ScalewayBackend)

    def test_koyeb_missing_provider_config_raises(self):
        """Test Koyeb without provider config raises ValueError."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="koyeb",
                endpoint_url="https://hark.koyeb.app",
            ),
        )
        with pytest.raises(ValueError, match="koyeb"):
            create_backend(config)

    def test_verda_missing_provider_config_raises(self):
        """Test Verda without provider config raises ValueError."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="verda",
                endpoint_url="https://containers.datacrunch.io/hark",
            ),
        )
        with pytest.raises(ValueError, match="verda"):
            create_backend(config)

    def test_remote_missing_endpoint_raises(self):
        """Test remote backend without endpoint_url raises ValueError."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="koyeb",
                koyeb=KoyebProviderSettings(token="t"),
            ),
        )
        with pytest.raises(ValueError, match="endpoint_url"):
            create_backend(config)


class TestCreateManagedBackend:
    """Tests for managed backend creation."""

    def test_managed_koyeb(self):
        """Test managed Koyeb backend creates ManagedBackend."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="koyeb",
                managed=True,
                worker_api_key="wk-secret",
                infra=InfraSettings(docker_image="test:latest"),
                koyeb_infra=KoyebInfraSettings(api_token="koyeb-tok"),
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, ManagedBackend)
        assert backend._infra.provider_name() == "koyeb"

    def test_managed_scaleway(self):
        """Test managed Scaleway backend creates ManagedBackend."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="scaleway",
                managed=True,
                worker_api_key="wk-secret",
                infra=InfraSettings(docker_image="test:latest"),
                scaleway_infra=ScalewayInfraSettings(
                    secret_key="scw-key",
                    organization_id="org-1",
                    project_id="proj-1",
                ),
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, ManagedBackend)
        assert backend._infra.provider_name() == "scaleway"

    def test_managed_verda(self):
        """Test managed Verda backend creates ManagedBackend."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="verda",
                managed=True,
                worker_api_key="wk-secret",
                infra=InfraSettings(docker_image="test:latest"),
                verda=VerdaProviderSettings(api_key="verda-key"),
                datacrunch_infra=DataCrunchInfraSettings(
                    client_id="dc-id",
                    client_secret="dc-secret",
                ),
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, ManagedBackend)
        assert backend._infra.provider_name() == "datacrunch"

    def test_managed_koyeb_disables_idle_with_scale_to_zero(self):
        """Test managed Koyeb with native scale-to-zero disables idle timer."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="koyeb",
                managed=True,
                worker_api_key="wk",
                infra=InfraSettings(idle_timeout=300),
                koyeb_infra=KoyebInfraSettings(
                    api_token="tok",
                    use_native_scale_to_zero=True,
                ),
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, ManagedBackend)
        assert backend._idle_timeout == 0

    def test_managed_local_returns_local(self):
        """Test managed=True with local backend still creates LocalBackend."""
        config = HarkdSettings(
            transcription=TranscriptionSettings(
                backend="local",
                managed=True,
            ),
        )
        backend = create_backend(config)
        assert isinstance(backend, LocalBackend)
