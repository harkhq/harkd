"""Tests for LLM client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harkd.config import LLMSettings
from harkd.llm.client import LLMClient
from harkd.llm.types import MeetingMinutesResult


@pytest.fixture
def llm_config():
    """Create a test LLM config."""
    return LLMSettings(
        enabled=True,
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        enable_cache=False,
        enable_logging=False,
        enable_token_stats=False,
    )


@pytest.fixture
def mock_chat_model():
    """Create a mock LangChain chat model."""
    mock = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "Test response"
    mock_result.usage_metadata = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    mock.ainvoke = AsyncMock(return_value=mock_result)
    mock.bind_tools.return_value = mock
    return mock


class TestLLMClientInvoke:
    """Tests for LLMClient.invoke()."""

    @pytest.mark.asyncio
    async def test_invoke_basic(self, llm_config, mock_chat_model):
        """Test basic invoke call."""
        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_chat_model,
        ):
            client = LLMClient(llm_config)

        result = await client.invoke(
            [{"role": "user", "content": "Hello"}]
        )

        assert result.content == "Test response"
        assert result.model == "gpt-4o-mini"
        mock_chat_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_with_system_message(
        self, llm_config, mock_chat_model
    ):
        """Test invoke with system and user messages."""
        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_chat_model,
        ):
            client = LLMClient(llm_config)

        await client.invoke([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ])

        call_args = mock_chat_model.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_invoke_extracts_token_usage(
        self, llm_config, mock_chat_model
    ):
        """Test that token usage is extracted from response."""
        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_chat_model,
        ):
            client = LLMClient(llm_config)

        result = await client.invoke(
            [{"role": "user", "content": "Hello"}]
        )

        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30


class TestLLMClientInvokeWithTools:
    """Tests for LLMClient.invoke_with_tools()."""

    @pytest.mark.asyncio
    async def test_invoke_with_tools_binds_tools(
        self, llm_config, mock_chat_model
    ):
        """Test that tools are bound to the model."""
        # bind_tools returns a new mock that also supports ainvoke
        bound_mock = MagicMock()
        bound_result = MagicMock()
        bound_result.content = "Weather is sunny"
        bound_result.usage_metadata = None
        bound_mock.ainvoke = AsyncMock(return_value=bound_result)
        mock_chat_model.bind_tools.return_value = bound_mock

        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_chat_model,
        ):
            client = LLMClient(llm_config)

        tools = [{"name": "get_weather", "description": "Get weather"}]
        await client.invoke_with_tools(
            [{"role": "user", "content": "What's the weather?"}],
            tools=tools,
        )

        mock_chat_model.bind_tools.assert_called_once_with(tools)


class TestGenerateMeetingMinutes:
    """Tests for LLMClient.generate_meeting_minutes()."""

    @pytest.mark.asyncio
    async def test_generate_meeting_minutes_success(
        self, llm_config, mock_chat_model
    ):
        """Test successful meeting minutes generation."""
        minutes_json = json.dumps({
            "executive_summary": ["Key point 1", "Key point 2"],
            "meeting_notes": [
                {"topic": "Design", "content": "Discussed new design"},
            ],
            "tasks": [
                {
                    "task": "Update docs",
                    "assignee": "SPEAKER_01",
                    "due": None,
                },
            ],
            "decisions": ["Use React for frontend"],
        })

        mock_result = MagicMock()
        mock_result.content = minutes_json
        mock_result.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
        }
        mock_chat_model.ainvoke.return_value = mock_result

        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_chat_model,
        ):
            client = LLMClient(llm_config)

        result = await client.generate_meeting_minutes(
            transcript="Speaker 1 said hello. Speaker 2 agreed.",
            speakers=["SPEAKER_01", "SPEAKER_02"],
            language="en",
        )

        assert isinstance(result, MeetingMinutesResult)
        assert len(result.executive_summary) == 2
        assert result.executive_summary[0] == "Key point 1"
        assert len(result.meeting_notes) == 1
        assert result.meeting_notes[0]["topic"] == "Design"
        assert len(result.tasks) == 1
        assert result.tasks[0]["task"] == "Update docs"
        assert len(result.decisions) == 1

    @pytest.mark.asyncio
    async def test_generate_meeting_minutes_malformed_json(
        self, llm_config, mock_chat_model
    ):
        """Test graceful handling of malformed JSON from LLM."""
        mock_result = MagicMock()
        mock_result.content = "This is not valid JSON at all"
        mock_result.usage_metadata = None
        mock_chat_model.ainvoke.return_value = mock_result

        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_chat_model,
        ):
            client = LLMClient(llm_config)

        result = await client.generate_meeting_minutes(
            transcript="Hello world",
            speakers=[],
            language="en",
        )

        # Should fallback gracefully
        assert isinstance(result, MeetingMinutesResult)
        assert len(result.executive_summary) == 1
        assert "could not be parsed" in result.executive_summary[0]

    @pytest.mark.asyncio
    async def test_generate_meeting_minutes_json_in_code_block(
        self, llm_config, mock_chat_model
    ):
        """Test parsing JSON wrapped in markdown code blocks."""
        minutes_json = json.dumps({
            "executive_summary": ["Summary"],
            "meeting_notes": [],
            "tasks": [],
            "decisions": [],
        })
        wrapped = f"```json\n{minutes_json}\n```"

        mock_result = MagicMock()
        mock_result.content = wrapped
        mock_result.usage_metadata = None
        mock_chat_model.ainvoke.return_value = mock_result

        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_chat_model,
        ):
            client = LLMClient(llm_config)

        result = await client.generate_meeting_minutes(
            transcript="Test",
            speakers=[],
            language="en",
        )

        assert result.executive_summary == ["Summary"]


class TestParseMeetingMinutes:
    """Tests for _parse_meeting_minutes static method."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        content = json.dumps({
            "executive_summary": ["Point 1"],
            "meeting_notes": [{"topic": "A", "content": "B"}],
            "tasks": [],
            "decisions": ["Decision 1"],
        })

        result = LLMClient._parse_meeting_minutes(content)

        assert result.executive_summary == ["Point 1"]
        assert len(result.meeting_notes) == 1
        assert result.decisions == ["Decision 1"]

    def test_json_in_code_block(self):
        """Test parsing JSON in markdown code block."""
        inner = json.dumps({
            "executive_summary": ["S"],
            "meeting_notes": [],
            "tasks": [],
            "decisions": [],
        })
        content = f"```json\n{inner}\n```"

        result = LLMClient._parse_meeting_minutes(content)
        assert result.executive_summary == ["S"]

    def test_invalid_json_returns_fallback(self):
        """Test that invalid JSON returns fallback result."""
        result = LLMClient._parse_meeting_minutes("not json")

        assert len(result.executive_summary) == 1
        assert "could not be parsed" in result.executive_summary[0]

    def test_missing_fields_default_to_empty(self):
        """Test that missing fields get defaults."""
        content = json.dumps({"executive_summary": ["Only this"]})

        result = LLMClient._parse_meeting_minutes(content)
        assert result.executive_summary == ["Only this"]
        assert result.meeting_notes == []
        assert result.tasks == []
        assert result.decisions == []


class TestLLMClientWithMiddleware:
    """Tests for LLMClient with middleware enabled."""

    @pytest.mark.asyncio
    async def test_token_stats_tracked(self):
        """Test that token_stats accumulates when enable_token_stats=True."""
        config = LLMSettings(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            enable_cache=False,
            enable_logging=False,
            enable_token_stats=True,
        )

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "response"
        mock_result.usage_metadata = {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
        }
        mock_model.ainvoke = AsyncMock(return_value=mock_result)

        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_model,
        ):
            client = LLMClient(config)

        assert client.token_stats is not None
        assert client.token_stats.total_tokens == 0

        await client.invoke([{"role": "user", "content": "Hi"}])

        assert client.token_stats.prompt_tokens == 50
        assert client.token_stats.completion_tokens == 100
        assert client.token_stats.total_tokens == 150

    @pytest.mark.asyncio
    async def test_token_stats_none_when_disabled(self):
        """Test that token_stats is None when tracking disabled."""
        config = LLMSettings(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            enable_token_stats=False,
            enable_logging=False,
        )

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(
            return_value=MagicMock(content="r", usage_metadata=None)
        )

        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_model,
        ):
            client = LLMClient(config)

        assert client.token_stats is None

    @pytest.mark.asyncio
    async def test_cache_returns_cached_on_duplicate(self):
        """Test that cache middleware returns cached response on repeat call."""
        config = LLMSettings(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            enable_cache=True,
            enable_logging=False,
            enable_token_stats=False,
        )

        call_count = 0
        mock_model = MagicMock()

        async def mock_ainvoke(messages):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.content = "cached response"
            result.usage_metadata = None
            return result

        mock_model.ainvoke = mock_ainvoke

        with patch(
            "harkd.llm.client.create_chat_model",
            return_value=mock_model,
        ):
            client = LLMClient(config)

        msg = [{"role": "user", "content": "test"}]
        r1 = await client.invoke(msg)
        r2 = await client.invoke(msg)

        assert call_count == 1  # Second call served from cache
        assert r1.content == "cached response"
        assert r2.cached is True
