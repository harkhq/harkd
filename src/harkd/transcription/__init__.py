"""Transcription backend abstraction layer.

Provides pluggable backends for transcription: local subprocess (default)
or remote GPU cloud providers (Koyeb, Verda, Scaleway).

When ``managed=True`` in config, backends are wrapped in a ManagedBackend
that auto-provisions and tears down cloud infrastructure on demand.
"""

from harkd.transcription.backend import BackendHealth, TranscriptionBackend, TranscriptionRequest
from harkd.transcription.koyeb import KoyebBackend
from harkd.transcription.local import LocalBackend
from harkd.transcription.remote import RemoteBackend
from harkd.transcription.scaleway import ScalewayBackend
from harkd.transcription.verda import VerdaBackend

__all__ = [
    "BackendHealth",
    "TranscriptionBackend",
    "TranscriptionRequest",
    "LocalBackend",
    "RemoteBackend",
    "KoyebBackend",
    "VerdaBackend",
    "ScalewayBackend",
    "create_backend",
]


def create_backend(config) -> TranscriptionBackend:
    """Create a transcription backend from daemon config.

    Args:
        config: HarkdSettings instance

    Returns:
        Configured TranscriptionBackend
    """
    from harkd.config import HarkdSettings

    assert isinstance(config, HarkdSettings)
    tc = config.transcription

    if tc.backend == "local":
        return LocalBackend()

    if tc.backend == "remote":
        if tc.endpoint_url is None:
            raise ValueError("endpoint_url is required for backend 'remote'")
        return RemoteBackend(
            endpoint_url=tc.endpoint_url,
            worker_api_key=tc.worker_api_key,
            timeout=tc.remote_timeout,
            max_retries=tc.max_retries,
        )

    if tc.managed:
        return _create_managed_backend(tc, config)

    if tc.endpoint_url is None:
        raise ValueError(f"endpoint_url is required for backend '{tc.backend}'")

    if tc.backend == "koyeb":
        if tc.koyeb is None:
            raise ValueError("transcription.koyeb config is required for koyeb backend")
        return KoyebBackend(
            endpoint_url=tc.endpoint_url,
            token=tc.koyeb.token,
            worker_api_key=tc.worker_api_key,
            timeout=tc.remote_timeout,
            max_retries=tc.max_retries,
        )

    if tc.backend == "verda":
        if tc.verda is None:
            raise ValueError("transcription.verda config is required for verda backend")
        return VerdaBackend(
            endpoint_url=tc.endpoint_url,
            provider=tc.verda,
            worker_api_key=tc.worker_api_key,
            file_storage=tc.file_storage,
            timeout=tc.remote_timeout,
            max_retries=tc.max_retries,
        )

    if tc.backend == "scaleway":
        api_key = tc.scaleway.api_key if tc.scaleway else None
        return ScalewayBackend(
            endpoint_url=tc.endpoint_url,
            api_key=api_key,
            worker_api_key=tc.worker_api_key,
            timeout=tc.remote_timeout,
            max_retries=tc.max_retries,
        )

    raise ValueError(f"Unknown transcription backend: {tc.backend}")


def _create_managed_backend(tc, config) -> TranscriptionBackend:
    """Create a ManagedBackend that auto-provisions cloud infrastructure."""
    from harkd.transcription.infra.managed import ManagedBackend
    from harkd.transcription.infra.provider import InfraProviderConfig

    assert tc.infra is not None

    infra_config = InfraProviderConfig(
        docker_image=tc.infra.docker_image,
        worker_api_key=tc.worker_api_key or "",
        worker_model=config.recording.model,
        hf_token=config.hf_token,
    )

    if tc.backend == "koyeb":
        from harkd.transcription.infra.koyeb_infra import KoyebInfraProvider

        assert tc.koyeb_infra is not None
        provider = KoyebInfraProvider(
            api_token=tc.koyeb_infra.api_token,
            region=tc.koyeb_infra.region,
            instance_type=tc.koyeb_infra.instance_type,
            app_name=tc.koyeb_infra.app_name,
            use_native_scale_to_zero=tc.koyeb_infra.use_native_scale_to_zero,
        )

        def koyeb_factory(endpoint_url: str) -> TranscriptionBackend:
            return KoyebBackend(
                endpoint_url=endpoint_url,
                token=tc.koyeb_infra.api_token,
                worker_api_key=tc.worker_api_key,
                timeout=tc.remote_timeout,
                max_retries=tc.max_retries,
            )

        idle_timeout = 0 if tc.koyeb_infra.use_native_scale_to_zero else tc.infra.idle_timeout

        return ManagedBackend(
            infra_provider=provider,
            infra_config=infra_config,
            backend_factory=koyeb_factory,
            idle_timeout=idle_timeout,
            max_runtime=tc.infra.max_runtime,
            provisioning_timeout=tc.infra.provisioning_timeout,
        )

    if tc.backend == "scaleway":
        from harkd.transcription.infra.scaleway_infra import ScalewayInfraProvider

        assert tc.scaleway_infra is not None
        provider = ScalewayInfraProvider(
            secret_key=tc.scaleway_infra.secret_key,
            organization_id=tc.scaleway_infra.organization_id,
            project_id=tc.scaleway_infra.project_id,
            zone=tc.scaleway_infra.zone,
            instance_type=tc.scaleway_infra.instance_type,
            image_id=tc.scaleway_infra.image_id,
        )

        def scaleway_factory(endpoint_url: str) -> TranscriptionBackend:
            api_key = tc.scaleway.api_key if tc.scaleway else None
            return ScalewayBackend(
                endpoint_url=endpoint_url,
                api_key=api_key,
                worker_api_key=tc.worker_api_key,
                timeout=tc.remote_timeout,
                max_retries=tc.max_retries,
            )

        return ManagedBackend(
            infra_provider=provider,
            infra_config=infra_config,
            backend_factory=scaleway_factory,
            idle_timeout=tc.infra.idle_timeout,
            max_runtime=tc.infra.max_runtime,
            provisioning_timeout=tc.infra.provisioning_timeout,
        )

    if tc.backend == "verda":
        from harkd.transcription.infra.datacrunch_infra import DataCrunchInfraProvider

        assert tc.datacrunch_infra is not None
        provider = DataCrunchInfraProvider(
            client_id=tc.datacrunch_infra.client_id,
            client_secret=tc.datacrunch_infra.client_secret,
            instance_type=tc.datacrunch_infra.instance_type,
            location=tc.datacrunch_infra.location,
            ssh_key_ids=tc.datacrunch_infra.ssh_key_ids,
            os_volume_id=tc.datacrunch_infra.os_volume_id,
        )

        def verda_factory(endpoint_url: str) -> TranscriptionBackend:
            assert tc.verda is not None
            return VerdaBackend(
                endpoint_url=endpoint_url,
                provider=tc.verda,
                worker_api_key=tc.worker_api_key,
                file_storage=tc.file_storage,
                timeout=tc.remote_timeout,
                max_retries=tc.max_retries,
            )

        return ManagedBackend(
            infra_provider=provider,
            infra_config=infra_config,
            backend_factory=verda_factory,
            idle_timeout=tc.infra.idle_timeout,
            max_runtime=tc.infra.max_runtime,
            provisioning_timeout=tc.infra.provisioning_timeout,
        )

    raise ValueError(f"Managed mode not supported for backend: {tc.backend}")
