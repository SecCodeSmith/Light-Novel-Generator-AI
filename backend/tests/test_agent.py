"""Unit tests for the agent loop engine (agent.py).

Tests cover:
- Single-round (no tools) pass-through
- Multi-round tool-calling loop
- Max rounds enforcement
- JSON mode with agent_chat_json
- Graceful fallback when model doesn't use tools

Redis/Neo4j modules are mocked at sys.modules level to avoid connection requirements.
"""
import json
import sys
from unittest.mock import MagicMock, patch, call

import pytest

# Mock infrastructure modules before importing agent
_mock_redis = MagicMock()
_mock_neo4j = MagicMock()
sys.modules.setdefault("redis", _mock_redis)
sys.modules.setdefault("neo4j", _mock_neo4j)

from app.llm import ChatResult, ToolCall


class FakeChatClient:
    """A test double that returns pre-programmed ChatResult sequences."""

    def __init__(self, responses: list[ChatResult]):
        self.responses = list(responses)
        self.call_count = 0
        self.calls: list[dict] = []

    def chat(self, model, messages, temperature=0.8, max_tokens=4096,
             json_mode=False, tools=None, tool_choice=None, mock_hint=None):
        self.calls.append({
            "model": model, "messages": messages, "tools": tools,
            "json_mode": json_mode, "mock_hint": mock_hint,
        })
        if self.call_count < len(self.responses):
            result = self.responses[self.call_count]
            self.call_count += 1
            return result
        return ChatResult(content="Fallback response", finish_reason="stop")


class TestAgentChat:
    """Tests for the agent_chat function."""

    def test_single_round_no_tools(self):
        """When the model returns content with no tool calls, return immediately."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8

            from app.agent import agent_chat

            client = FakeChatClient([
                ChatResult(content="Hello, this is the answer.", finish_reason="stop"),
            ])

            result = agent_chat(client, "test-model",
                                [{"role": "user", "content": "Hi"}],
                                story_id="s1", role="test")

            assert result == "Hello, this is the answer."
            assert client.call_count == 1

    def test_tool_call_loop(self):
        """Model calls a tool, gets result, then produces final answer."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config, \
             patch("app.agent.dispatch_tool") as mock_dispatch:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8
            mock_dispatch.return_value = {"query": "medieval libraries",
                                          "results": [{"title": "Library", "url": "...", "snippet": "..."}]}

            from app.agent import agent_chat

            client = FakeChatClient([
                ChatResult(
                    content="",
                    tool_calls=[ToolCall(id="call_1", name="web_search",
                                         arguments={"query": "medieval libraries"})],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="Medieval libraries were...", finish_reason="stop"),
            ])

            result = agent_chat(client, "test-model",
                                [{"role": "user", "content": "Tell me about medieval libraries"}],
                                tools=[{"type": "function", "function": {"name": "web_search"}}],
                                story_id="s1", role="writer")

            assert result == "Medieval libraries were..."
            assert client.call_count == 2
            mock_dispatch.assert_called_once_with("web_search",
                                                  {"query": "medieval libraries"},
                                                  story_id="s1")

    def test_multiple_tool_calls_in_one_round(self):
        """Model can make multiple tool calls in a single round."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config, \
             patch("app.agent.dispatch_tool") as mock_dispatch:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8
            mock_dispatch.side_effect = [
                {"query": "gothic architecture", "results": []},
                {"topic": "Gargoyle", "found": True, "summary": "Stone creature..."},
            ]

            from app.agent import agent_chat

            client = FakeChatClient([
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_1", name="web_search",
                                 arguments={"query": "gothic architecture"}),
                        ToolCall(id="call_2", name="wikipedia_lookup",
                                 arguments={"topic": "Gargoyle"}),
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="Gothic buildings featured gargoyles...", finish_reason="stop"),
            ])

            result = agent_chat(client, "test-model",
                                [{"role": "user", "content": "Describe gothic buildings"}],
                                tools=[{}], story_id="s1", role="writer")

            assert "Gothic buildings featured gargoyles" in result
            assert mock_dispatch.call_count == 2

    def test_max_rounds_enforced(self):
        """Agent should stop after max_rounds even if model keeps calling tools."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config, \
             patch("app.agent.dispatch_tool") as mock_dispatch:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8
            mock_dispatch.return_value = {"results": []}

            from app.agent import agent_chat

            # Model always wants to call tools except when tools=None
            responses = []
            for i in range(3):
                responses.append(ChatResult(
                    content="",
                    tool_calls=[ToolCall(id=f"call_{i}", name="web_search",
                                         arguments={"query": f"search {i}"})],
                    finish_reason="tool_calls",
                ))
            # Final round (no tools) should produce content
            responses.append(ChatResult(content="Final answer", finish_reason="stop"))
            client = FakeChatClient(responses)

            result = agent_chat(client, "test-model",
                                [{"role": "user", "content": "Search"}],
                                tools=[{}], max_rounds=3, story_id="s1", role="test")

            assert client.call_count == 4  # 3 with tools + 1 without
            assert result == "Final answer"

    def test_no_tools_provided(self):
        """When no tools are provided, behaves like a regular chat call."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8

            from app.agent import agent_chat

            client = FakeChatClient([
                ChatResult(content="Just a normal response", finish_reason="stop"),
            ])

            result = agent_chat(client, "test-model",
                                [{"role": "user", "content": "Hi"}],
                                tools=None, story_id="s1", role="test")

            assert result == "Just a normal response"
            assert client.call_count == 1
            assert client.calls[0]["tools"] is None

    def test_empty_content_raises_error(self):
        """Empty content with no tool calls should raise LLMError."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8

            from app.agent import agent_chat
            from app.llm import LLMError

            client = FakeChatClient([
                ChatResult(content="", finish_reason="stop"),
            ])

            with pytest.raises(LLMError, match="empty content"):
                agent_chat(client, "test-model",
                           [{"role": "user", "content": "Hi"}],
                           story_id="s1", role="test")

    def test_tool_results_in_conversation(self):
        """Tool results should be appended to the conversation correctly."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config, \
             patch("app.agent.dispatch_tool") as mock_dispatch:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8
            mock_dispatch.return_value = {"results": [{"title": "Result"}]}

            from app.agent import agent_chat

            client = FakeChatClient([
                ChatResult(
                    content="",
                    tool_calls=[ToolCall(id="call_1", name="web_search",
                                         arguments={"query": "test"})],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="Answer based on search", finish_reason="stop"),
            ])

            result = agent_chat(client, "test-model",
                                [{"role": "user", "content": "Search something"}],
                                tools=[{}], story_id="s1", role="test")

            # Check the second call had the tool result in its messages
            second_call_messages = client.calls[1]["messages"]
            tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
            assert len(tool_msg) == 1
            assert tool_msg[0]["tool_call_id"] == "call_1"
            assert json.loads(tool_msg[0]["content"])["results"][0]["title"] == "Result"

    def test_logging_tool_calls(self):
        """Tool calls and results should be logged to the progress log."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config, \
             patch("app.agent.dispatch_tool") as mock_dispatch:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8
            mock_dispatch.return_value = {"results": []}

            from app.agent import agent_chat

            client = FakeChatClient([
                ChatResult(
                    content="",
                    tool_calls=[ToolCall(id="call_1", name="web_search",
                                         arguments={"query": "test query"})],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="Done", finish_reason="stop"),
            ])

            agent_chat(client, "test-model",
                       [{"role": "user", "content": "Search"}],
                       tools=[{}], story_id="s1", role="writer")

            # Check that cache.log was called with tool call info
            log_calls = [str(c) for c in mock_cache.log.call_args_list]
            tool_log = [c for c in log_calls if "web_search" in c]
            assert len(tool_log) >= 1


class TestAgentChatJson:
    """Tests for the agent_chat_json function."""

    def test_json_response(self):
        """agent_chat_json should parse JSON from the model's response."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8

            from app.agent import agent_chat_json

            client = FakeChatClient([
                ChatResult(
                    content='{"chapters": [{"number": 1, "title": "Chapter 1"}]}',
                    finish_reason="stop",
                ),
            ])

            result = agent_chat_json(client, "test-model",
                                     "You are an architect.", "Plan chapter 1.",
                                     temperature=0.8, story_id="s1", role="architect")

            assert "chapters" in result
            assert result["chapters"][0]["number"] == 1

    def test_json_with_tool_calls(self):
        """agent_chat_json should support tool calls before JSON response."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config, \
             patch("app.agent.dispatch_tool") as mock_dispatch:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8
            mock_dispatch.return_value = {"found": True, "summary": "Libraries in medieval times..."}

            from app.agent import agent_chat_json

            client = FakeChatClient([
                ChatResult(
                    content="",
                    tool_calls=[ToolCall(id="call_1", name="wikipedia_lookup",
                                         arguments={"topic": "Medieval libraries"})],
                    finish_reason="tool_calls",
                ),
                ChatResult(
                    content='{"premise": "A story about libraries", "characters": []}',
                    finish_reason="stop",
                ),
            ])

            result = agent_chat_json(client, "test-model",
                                     "You are an architect.", "Plan the story.",
                                     temperature=0.8, tools=[{}],
                                     story_id="s1", role="architect")

            assert result["premise"] == "A story about libraries"

    def test_json_repair_truncated(self):
        """agent_chat_json should repair truncated JSON responses."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8

            from app.agent import agent_chat_json

            client = FakeChatClient([
                ChatResult(
                    content='{"chapters": [{"number": 1, "title": "Ch1"}',
                    finish_reason="length",
                ),
            ])

            result = agent_chat_json(client, "test-model",
                                     "System.", "User.",
                                     temperature=0.8, story_id="s1", role="test")

            assert "chapters" in result

    def test_json_retry_on_invalid(self):
        """agent_chat_json should retry if the first response isn't valid JSON."""
        with patch("app.agent.cache") as mock_cache, \
             patch("app.agent.config") as mock_config:
            mock_cache.log = MagicMock()
            mock_cache.record_llm = MagicMock()
            mock_config.AGENT_MAX_TOOL_ROUNDS = 8

            from app.agent import agent_chat_json

            client = FakeChatClient([
                ChatResult(content="This is not JSON at all!", finish_reason="stop"),
                # Retry response
                ChatResult(content='{"result": "ok"}', finish_reason="stop"),
            ])

            result = agent_chat_json(client, "test-model",
                                     "System.", "User.",
                                     temperature=0.8, story_id="s1", role="test")

            assert result["result"] == "ok"
            assert client.call_count == 2


class TestToolDefsForRole:
    """Tests for _tool_defs_for_role."""

    def test_architect_gets_tools(self):
        """Architect role should get all research tools."""
        with patch("app.agent.cache"), patch("app.agent.config"):
            from app.agent import _tool_defs_for_role

            tools = _tool_defs_for_role("architect")
            names = {t["function"]["name"] for t in tools}
            assert "web_search" in names
            assert "wikipedia_lookup" in names
            assert "story_graph_query" in names

    def test_writer_gets_tools(self):
        """Writer role should get all research tools."""
        with patch("app.agent.cache"), patch("app.agent.config"):
            from app.agent import _tool_defs_for_role
            tools = _tool_defs_for_role("writer")
            assert len(tools) == 5

    def test_critic_gets_no_tools(self):
        """Critic role should get no tools."""
        with patch("app.agent.cache"), patch("app.agent.config"):
            from app.agent import _tool_defs_for_role
            tools = _tool_defs_for_role("critic")
            assert tools == []

    def test_extractor_gets_no_tools(self):
        """Extractor role should get no tools."""
        with patch("app.agent.cache"), patch("app.agent.config"):
            from app.agent import _tool_defs_for_role
            tools = _tool_defs_for_role("extractor")
            assert tools == []
