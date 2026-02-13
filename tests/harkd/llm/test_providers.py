"""Tests for LLM provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from harkd.config import LLMSettings
from harkd.llm.providers import create_chat_model


class TestCreateChatModel:
    """Tests for create_chat_model factory."""

    def test_openai_provider(self):
        """Test creating OpenAI model."""
        config = LLMSettings(
            provider="openai", model="gpt-4o-mini", api_key="sk-test"
        )

        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatOpenAI = mock_cls

        with patch.dict("sys.modules", {"langchain_openai": mock_module}):
            create_chat_model(config)

        mock_cls.assert_called_once_with(
            model="gpt-4o-mini",
            api_key="sk-test",
            temperature=0.3,
            max_tokens=4096,
        )

    def test_anthropic_provider(self):
        """Test creating Anthropic model."""
        config = LLMSettings(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key="sk-ant-test",
        )

        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatAnthropic = mock_cls

        with patch.dict(
            "sys.modules", {"langchain_anthropic": mock_module}
        ):
            create_chat_model(config)

        mock_cls.assert_called_once_with(
            model_name="claude-sonnet-4-5-20250929",
            api_key="sk-ant-test",
            temperature=0.3,
            max_tokens=4096,
        )

    def test_google_provider(self):
        """Test creating Google Generative AI model."""
        config = LLMSettings(
            provider="google", model="gemini-pro", api_key="google-test"
        )

        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatGoogleGenerativeAI = mock_cls

        with patch.dict(
            "sys.modules", {"langchain_google_genai": mock_module}
        ):
            create_chat_model(config)

        mock_cls.assert_called_once_with(
            model="gemini-pro",
            google_api_key="google-test",
            temperature=0.3,
            max_output_tokens=4096,
        )

    def test_ollama_provider(self):
        """Test creating Ollama model."""
        config = LLMSettings(provider="ollama", model="llama3.2")

        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatOllama = mock_cls

        with patch.dict("sys.modules", {"langchain_ollama": mock_module}):
            create_chat_model(config)

        mock_cls.assert_called_once_with(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.3,
            num_predict=4096,
        )

    def test_ollama_custom_base_url(self):
        """Test Ollama with custom base URL."""
        config = LLMSettings(
            provider="ollama",
            model="llama3.2",
            base_url="http://my-server:11434",
        )

        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatOllama = mock_cls

        with patch.dict("sys.modules", {"langchain_ollama": mock_module}):
            create_chat_model(config)

        mock_cls.assert_called_once_with(
            model="llama3.2",
            base_url="http://my-server:11434",
            temperature=0.3,
            num_predict=4096,
        )

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        config = LLMSettings(provider="openai")
        # Force an unknown provider by modifying the object
        object.__setattr__(config, "provider", "unknown_provider")

        with pytest.raises(
            ValueError, match="Unknown LLM provider: unknown_provider"
        ):
            create_chat_model(config)

    def test_missing_provider_package_raises_import_error(self):
        """Test ImportError when provider package is not installed."""
        config = LLMSettings(
            provider="openai", model="gpt-4o", api_key="sk-test"
        )

        # Remove langchain_openai from sys.modules so the lazy import fails
        with patch.dict(
            "sys.modules", {"langchain_openai": None}
        ):
            with pytest.raises(ImportError):
                create_chat_model(config)

    def test_custom_temperature_and_max_tokens_propagate(self):
        """Test that non-default temperature/max_tokens reach the provider."""
        config = LLMSettings(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            temperature=1.5,
            max_tokens=512,
        )

        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatOpenAI = mock_cls

        with patch.dict("sys.modules", {"langchain_openai": mock_module}):
            create_chat_model(config)

        mock_cls.assert_called_once_with(
            model="gpt-4o",
            api_key="sk-test",
            temperature=1.5,
            max_tokens=512,
        )
