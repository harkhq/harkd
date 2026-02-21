"""Tests for ScalewayBackend."""

from harkd.transcription.scaleway import ScalewayBackend


class TestScalewayBackend:
    """Tests for Scaleway-specific behavior."""

    def test_provider_headers_with_key(self):
        """Test Scaleway auth headers with API key."""
        backend = ScalewayBackend(
            endpoint_url="https://gpu.scaleway.com",
            api_key="scw-secret-key",
        )
        headers = backend._provider_headers()
        assert headers["X-Auth-Token"] == "scw-secret-key"

    def test_provider_headers_without_key(self):
        """Test Scaleway auth headers without API key."""
        backend = ScalewayBackend(
            endpoint_url="https://gpu.scaleway.com",
        )
        headers = backend._provider_headers()
        assert headers == {}

    def test_build_headers_combines_worker_and_provider(self):
        """Test combined headers include both worker and provider auth."""
        backend = ScalewayBackend(
            endpoint_url="https://gpu.scaleway.com",
            api_key="scw-secret",
            worker_api_key="worker-secret",
        )
        headers = backend._build_headers()
        assert headers["Authorization"] == "Bearer worker-secret"
        assert headers["X-Auth-Token"] == "scw-secret"

    def test_provider_name(self):
        """Test provider name is scaleway."""
        backend = ScalewayBackend(
            endpoint_url="https://gpu.scaleway.com",
        )
        assert backend._provider_name() == "scaleway"
