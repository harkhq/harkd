"""Uvicorn server startup for harkd daemon."""

import uvicorn

from harkd.config import get_settings

__all__ = ["run_server"]


def _validate_settings(settings) -> None:
    """Validate configuration before starting the server.

    Raises:
        SystemExit: If configuration is invalid
    """
    # Validate: diarization enabled requires hf_token
    if settings.recording.diarization and not settings.hf_token:
        raise SystemExit(
            "ERROR: Diarization is enabled (HARKD_RECORDING__DIARIZATION=true) "
            "but no HuggingFace token is configured.\n"
            "Set HARKD_HF_TOKEN or add hf_token to daemon.yaml.\n"
            "Get a token at https://huggingface.co/settings/tokens"
        )

    # Validate: LLM enabled requires api_key (except Ollama)
    if settings.llm.enabled and settings.llm.provider != "ollama" and not settings.llm.api_key:
        raise SystemExit(
            f"ERROR: LLM is enabled with provider '{settings.llm.provider}' "
            "but no API key is configured.\n"
            "Set HARKD_LLM__API_KEY or add llm.api_key to daemon.yaml."
        )


def run_server(
    host: str | None = None,
    port: int | None = None,
    reload: bool | None = None,
) -> None:
    """Run the harkd server.

    Args:
        host: Server host (defaults to settings)
        port: Server port (defaults to settings)
        reload: Auto-reload on code changes (defaults to settings)
    """
    settings = get_settings()

    # Validate configuration before starting
    _validate_settings(settings)

    # Use provided args or fall back to settings
    server_host = host or settings.server.host
    server_port = port or settings.server.port
    server_reload = reload if reload is not None else settings.server.reload

    # Run server — pass import string so uvicorn can reload the module
    uvicorn.run(
        "harkd.api.app:create_app",
        factory=True,
        host=server_host,
        port=server_port,
        reload=server_reload,
        log_level=settings.logging.level.lower(),
    )
