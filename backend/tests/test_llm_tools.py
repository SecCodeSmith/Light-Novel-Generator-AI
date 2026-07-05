"""Unit tests for llm.py — ChatResult, ToolCall, tool_calls parsing.

Tests cover the new dataclasses and parsing logic added for tool support.
"""
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock infrastructure modules before importing app.llm
sys.modules.setdefault("redis", MagicMock())
sys.modules.setdefault("neo4j", MagicMock())

from app.llm import (
    ChatResult,
    ToolCall,
    _parse_tool_calls,
    parse_json_reply,
    repair_truncated_json,
    _close_json,
)


class TestToolCall:
    """Tests for the ToolCall dataclass."""

    def test_creation(self):
        """ToolCall should store id, name, and arguments."""
        tc = ToolCall(id="call_123", name="web_search", arguments={"query": "test"})
        assert tc.id == "call_123"
        assert tc.name == "web_search"
        assert tc.arguments == {"query": "test"}

    def test_to_dict(self):
        """to_dict should return a plain dict representation."""
        tc = ToolCall(id="call_1", name="wikipedia_lookup",
                      arguments={"topic": "Rome", "language": "en"})
        d = tc.to_dict()
        assert d == {
            "id": "call_1",
            "name": "wikipedia_lookup",
            "arguments": {"topic": "Rome", "language": "en"},
        }


class TestChatResult:
    """Tests for the ChatResult dataclass."""

    def test_defaults(self):
        """ChatResult should have sensible defaults."""
        r = ChatResult()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.finish_reason == "stop"
        assert r.has_tool_calls is False

    def test_with_content(self):
        """ChatResult with content and no tool calls."""
        r = ChatResult(content="Hello world", finish_reason="stop")
        assert r.content == "Hello world"
        assert r.has_tool_calls is False

    def test_with_tool_calls(self):
        """ChatResult with tool calls should have has_tool_calls True."""
        r = ChatResult(
            content="",
            tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "test"})],
            finish_reason="tool_calls",
        )
        assert r.has_tool_calls is True
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "web_search"

    def test_with_both_content_and_tools(self):
        """ChatResult can have both content and tool calls."""
        r = ChatResult(
            content="I'll search for that.",
            tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "x"})],
            finish_reason="tool_calls",
        )
        assert r.content == "I'll search for that."
        assert r.has_tool_calls is True


class TestParseToolCalls:
    """Tests for _parse_tool_calls."""

    def test_parse_single_tool_call(self):
        """Parse a single tool call from the API response format."""
        raw = [{
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "medieval libraries"}'
            },
        }]
        calls = _parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].id == "call_abc123"
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"query": "medieval libraries"}

    def test_parse_multiple_tool_calls(self):
        """Parse multiple parallel tool calls."""
        raw = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "q1"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "wikipedia_lookup", "arguments": '{"topic": "Rome"}'},
            },
        ]
        calls = _parse_tool_calls(raw)
        assert len(calls) == 2
        assert calls[0].name == "web_search"
        assert calls[1].name == "wikipedia_lookup"
        assert calls[1].arguments == {"topic": "Rome"}

    def test_parse_invalid_json_arguments(self):
        """Invalid JSON arguments should be stored as _raw."""
        raw = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "web_search", "arguments": "not valid json{"},
        }]
        calls = _parse_tool_calls(raw)
        assert len(calls) == 1
        assert "_raw" in calls[0].arguments

    def test_parse_dict_arguments(self):
        """Arguments already as a dict (some providers) should work."""
        raw = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "web_search", "arguments": {"query": "test"}},
        }]
        calls = _parse_tool_calls(raw)
        assert calls[0].arguments == {"query": "test"}

    def test_parse_missing_id(self):
        """Missing tool call ID should generate a fallback."""
        raw = [{
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "test"}'},
        }]
        calls = _parse_tool_calls(raw)
        assert calls[0].id == "call_web_search"

    def test_parse_empty_list(self):
        """Empty tool calls list should return empty."""
        calls = _parse_tool_calls([])
        assert calls == []


class TestParseJsonReply:
    """Tests for parse_json_reply (existing function, verify still works)."""

    def test_plain_json(self):
        """Parse plain JSON."""
        result = parse_json_reply('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        """Parse JSON in markdown fences."""
        result = parse_json_reply('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_preamble(self):
        """Parse JSON preceded by text."""
        result = parse_json_reply('Here is the result:\n{"key": "value"}\n\nDone.')
        assert result == {"key": "value"}

    def test_think_blocks_stripped(self):
        """<think> blocks should be stripped before parsing."""
        result = parse_json_reply('<think>reasoning here</think>{"key": "value"}')
        assert result == {"key": "value"}


class TestRepairTruncatedJson:
    """Tests for repair_truncated_json."""

    def test_repair_truncated_object(self):
        """Repair a JSON object that was cut off."""
        result = repair_truncated_json('{"chapters": [{"number": 1}')
        assert result is not None
        assert "chapters" in result

    def test_repair_returns_none_for_non_json(self):
        """Return None when there's no JSON at all."""
        result = repair_truncated_json("This is just text with no JSON")
        assert result is None

    def test_close_json_string(self):
        """_close_json should close unclosed brackets/braces."""
        result = _close_json('{"a": [1, 2')
        assert result.endswith("]}")


class TestMockClient:
    """Tests for MockClient tool support."""

    def test_mock_returns_chat_result(self):
        """MockClient.chat() should return ChatResult."""
        from app.llm import MockClient

        client = MockClient()
        result = client.chat(model="mock-large",
                             messages=[{"role": "user", "content": "Hi"}])
        assert isinstance(result, ChatResult)
        assert result.content == "Mock reply."
        assert result.has_tool_calls is False

    def test_mock_agent_research_hint(self):
        """MockClient should simulate tool calls for agent_research hint."""
        from app.llm import MockClient

        client = MockClient()
        result = client.chat(
            model="mock-large",
            messages=[{"role": "user", "content": "Research for the story"}],
            tools=[{"type": "function", "function": {"name": "web_search"}}],
            mock_hint={"task": "agent_research", "research_query": "drowned library"},
        )
        assert isinstance(result, ChatResult)
        assert result.has_tool_calls is True
        assert result.tool_calls[0].name == "web_search"
        assert result.tool_calls[0].arguments["query"] == "drowned library"

    def test_mock_outline_returns_chat_result(self):
        """MockClient outline task should return ChatResult with JSON content."""
        from app.llm import MockClient

        client = MockClient()
        result = client.chat(
            model="mock-large",
            messages=[{"role": "user", "content": "Plan chapters"}],
            mock_hint={"task": "outline", "first": 1, "last": 3, "title": "Test Novel"},
        )
        assert isinstance(result, ChatResult)
        data = json.loads(result.content)
        assert "chapters" in data
        assert len(data["chapters"]) == 3

    def test_mock_list_models(self):
        """MockClient.list_models() should return mock model names."""
        from app.llm import MockClient

        client = MockClient()
        models = client.list_models()
        assert "mock-large" in models
        assert "mock-small" in models


class TestMakeClient:
    """Tests for make_client factory."""

    def test_mock_base_url(self):
        """'mock' base_url should return MockClient."""
        from app.llm import make_client, MockClient

        client = make_client({"base_url": "mock"})
        assert isinstance(client, MockClient)

    def test_empty_base_url_raises(self):
        """Empty base_url should raise LLMError."""
        from app.llm import make_client, LLMError

        with pytest.raises(LLMError, match="not configured"):
            make_client({"base_url": ""})

    def test_real_base_url(self):
        """A real URL should return OpenAICompatClient."""
        from app.llm import make_client, OpenAICompatClient

        client = make_client({"base_url": "https://api.openai.com/v1", "api_key": "sk-test"})
        assert isinstance(client, OpenAICompatClient)
