"""Infrastructure lifecycle management for transcription backends.

Provides automatic provisioning, management, and teardown of cloud GPU
infrastructure so backends can spin up on demand and scale to zero when idle.
"""

from harkd.transcription.infra.managed import ManagedBackend
from harkd.transcription.infra.provider import (
    InfraProvider,
    InfraProviderConfig,
    InfraState,
    InfraStatus,
)

__all__ = [
    "InfraProvider",
    "InfraProviderConfig",
    "InfraState",
    "InfraStatus",
    "ManagedBackend",
]
