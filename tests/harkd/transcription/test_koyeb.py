"""Tests for KoyebBackend."""

from harkd.transcription.koyeb import KoyebBackend


class TestKoyebBackend:
    """Tests for Koyeb-specific behavior."""

    def test_provider_headers(self):
        """Test Koyeb auth headers."""
        backend = KoyebBackend(
            endpoint_url="https://hark.koyeb.app",
            token="koyeb-token-123",
        )
        headers = backend._provider_headers()
        assert headers["X-Koyeb-Token"] == "koyeb-token-123"

    def test_build_headers_combines_worker_and_provider(self):
        """Test combined headers include both worker and provider auth."""
        backend = KoyebBackend(
            endpoint_url="https://hark.koyeb.app",
            token="koyeb-token",
            worker_api_key="worker-secret",
        )
        headers = backend._build_headers()
        assert headers["Authorization"] == "Bearer worker-secret"
        assert headers["X-Koyeb-Token"] == "koyeb-token"

    def test_provider_name(self):
        """Test provider name is koyeb."""
        backend = KoyebBackend(
            endpoint_url="https://hark.koyeb.app",
            token="koyeb-token",
        )
        assert backend._provider_name() == "koyeb"
