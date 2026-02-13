"""LangChain model factory for multiple LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from harkd.config import LLMSettings


def create_chat_model(config: LLMSettings) -> BaseChatModel:
    """Create a LangChain chat model from configuration.

    Uses lazy imports so missing provider packages only error when actually used.

    Args:
        config: LLM settings with provider, model, api_key, etc.

    Returns:
        A LangChain BaseChatModel instance

    Raises:
        ValueError: If the provider is unknown
        ImportError: If the provider's package is not installed
    """
    match config.provider:
        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model_name=config.model,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        case "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.api_key,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )
        case "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=config.model,
                base_url=config.base_url or "http://localhost:11434",
                temperature=config.temperature,
                num_predict=config.max_tokens,
            )
        case _:
            raise ValueError(f"Unknown LLM provider: {config.provider}")
