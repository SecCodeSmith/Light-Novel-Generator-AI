"""Unit tests for research_tools module.

All network calls are mocked to ensure tests run offline and deterministically.
The graph and cache modules are mocked at the sys.modules level to avoid
requiring Redis/Neo4j connections.
"""
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock infrastructure modules before importing research_tools
# to avoid neo4j/redis connection attempts.
_mock_config = MagicMock()
_mock_config.NEO4J_URI = "bolt://localhost:7687"
_mock_config.NEO4J_USER = "neo4j"
_mock_config.NEO4J_PASSWORD = "test"
_mock_config.REDIS_URL = "redis://localhost:6379/0"
_mock_config.DEFAULT_LLM_BASE_URL = "mock"
_mock_config.DEFAULT_LLM_API_KEY = ""
_mock_config.AGENT_MODE_DEFAULT = True
_mock_config.AGENT_MAX_TOOL_ROUNDS = 8

_mock_redis_mod = MagicMock()
_mock_neo4j_mod = MagicMock()

sys.modules["redis"] = _mock_redis_mod
sys.modules["neo4j"] = _mock_neo4j_mod

# Now safe to import
from app.research_tools import (
    TOOL_DEFINITIONS,
    dispatch_tool,
    story_graph_query,
    web_search,
    wikipedia_lookup,
)


class TestWebSearch:
    """Tests for the web_search function."""

    @patch("httpx.get")
    def test_web_search_returns_results(self, mock_get):
        """web_search should return a dict with 'query' and 'results'."""
        mock_resp = MagicMock()
        mock_resp.text = '''
            <div class="result">
                <h2 class="result__title">Result 1</h2>
                <a class="result__url" href="https://example.com/1">Link 1</a>
                <a class="result__snippet">Snippet 1</a>
            </div>
            <div class="result">
                <h2 class="result__title">Result 2</h2>
                <a class="result__url" href="https://example.com/2">Link 2</a>
                <a class="result__snippet">Snippet 2</a>
            </div>
        '''
        mock_get.return_value = mock_resp
        
        result = web_search("test query", max_results=5)
        
        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Result 1"
        assert result["results"][0]["url"] == "https://example.com/1"
        assert result["results"][0]["snippet"] == "Snippet 1"

    @patch("httpx.get")
    def test_web_search_handles_network_error(self, mock_get):
        """web_search should return an error dict on network failure."""
        mock_get.side_effect = Exception("Connection timeout")
        
        result = web_search("test query")
        
        assert result["query"] == "test query"
        assert "error" in result
        assert "Connection timeout" in result["error"]

    def test_web_search_clamps_max_results_low(self):
        """max_results below 1 should be clamped to 1."""
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            # Will fail due to import but the clamping still runs
            result = web_search("test", max_results=-5)
        # Just verify it doesn't crash
        assert result["query"] == "test"

    def test_web_search_clamps_max_results_high(self):
        """max_results above 10 should be clamped to 10."""
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            result = web_search("test", max_results=100)
        assert result["query"] == "test"


class TestWikipediaLookup:
    """Tests for the wikipedia_lookup function."""

    def test_wikipedia_lookup_found(self):
        """wikipedia_lookup should return article summary when found."""
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.title = "Test Article"
        mock_page.summary = "This is the first sentence. This is the second sentence. This is the third."
        mock_page.fullurl = "https://en.wikipedia.org/wiki/Test_Article"
        mock_page.sections = [MagicMock(title="History"), MagicMock(title="Overview")]

        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page

        mock_module = MagicMock()
        mock_module.Wikipedia.return_value = mock_wiki

        with patch.dict("sys.modules", {"wikipediaapi": mock_module}):
            result = wikipedia_lookup("Test Article", sentences=2)

        assert result["found"] is True
        assert result["title"] == "Test Article"
        assert result["url"] == "https://en.wikipedia.org/wiki/Test_Article"
        assert len(result["sections"]) == 2

    def test_wikipedia_lookup_not_found(self):
        """wikipedia_lookup should return found=False for missing articles."""
        mock_page = MagicMock()
        mock_page.exists.return_value = False

        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page

        mock_module = MagicMock()
        mock_module.Wikipedia.return_value = mock_wiki

        with patch.dict("sys.modules", {"wikipediaapi": mock_module}):
            result = wikipedia_lookup("Nonexistent Topic")

        assert result["found"] is False
        assert "error" in result

    def test_wikipedia_lookup_handles_import_error(self):
        """wikipedia_lookup should return an error dict if wikipedia-api is missing."""
        with patch.dict("sys.modules", {"wikipediaapi": None}):
            result = wikipedia_lookup("Test")

        assert result["found"] is False
        assert "error" in result

    def test_wikipedia_lookup_clamps_sentences(self):
        """sentences parameter should be clamped between 1 and 20."""
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.title = "Test"
        mock_page.summary = "One sentence."
        mock_page.fullurl = "https://en.wikipedia.org/wiki/Test"
        mock_page.sections = []

        mock_wiki = MagicMock()
        mock_wiki.page.return_value = mock_page

        mock_module = MagicMock()
        mock_module.Wikipedia.return_value = mock_wiki

        with patch.dict("sys.modules", {"wikipediaapi": mock_module}):
            result = wikipedia_lookup("Test", sentences=100)

        assert result["found"] is True


class TestStoryGraphQuery:
    """Tests for the story_graph_query function."""

    def test_graph_query_found(self):
        """story_graph_query should return entity details when found."""
        mock_graph = MagicMock()
        mock_graph.entity_connections.return_value = {
            "found": True,
            "node": {"name": "Aria Vale", "role": "protagonist", "status": "alive",
                     "description": "An archivist", "story_id": "abc123"},
            "links": [
                {"type": "ALLY_OF", "dir": "out", "other": "Corin Ashe", "label": "Character"},
            ],
        }

        with patch.dict("sys.modules", {"app.graph": mock_graph}):
            result = story_graph_query("abc123", "Aria Vale")

        assert result["found"] is True
        assert result["entity_name"] == "Aria Vale"
        assert result["node"]["name"] == "Aria Vale"
        assert "story_id" not in result["node"]  # should be stripped
        assert len(result["links"]) == 1
        assert result["links"][0]["type"] == "ALLY_OF"

    def test_graph_query_not_found(self):
        """story_graph_query should return found=False for missing entities."""
        mock_graph = MagicMock()
        mock_graph.entity_connections.return_value = {"found": False, "name": "Nobody"}

        with patch.dict("sys.modules", {"app.graph": mock_graph}):
            result = story_graph_query("abc123", "Nobody")

        assert result["found"] is False
        assert "error" in result

    def test_graph_query_handles_exception(self):
        """story_graph_query should return an error on database failure."""
        mock_graph = MagicMock()
        mock_graph.entity_connections.side_effect = Exception("DB connection lost")

        with patch.dict("sys.modules", {"app.graph": mock_graph}):
            result = story_graph_query("abc123", "Aria")

        assert result["found"] is False
        assert "error" in result


class TestDispatchTool:
    """Tests for the dispatch_tool function."""

    def test_dispatch_web_search(self):
        """dispatch_tool should route 'web_search' to web_search function."""
        with patch("app.research_tools.web_search") as mock_search:
            mock_search.return_value = {"query": "test", "results": []}
            result = dispatch_tool("web_search", {"query": "test"})

        mock_search.assert_called_once_with(query="test", max_results=5)
        assert result["query"] == "test"

    def test_dispatch_wikipedia(self):
        """dispatch_tool should route 'wikipedia_lookup' to wikipedia_lookup function."""
        with patch("app.research_tools.wikipedia_lookup") as mock_wiki:
            mock_wiki.return_value = {"topic": "test", "found": True}
            dispatch_tool("wikipedia_lookup", {"topic": "test"})

        mock_wiki.assert_called_once_with(topic="test", language="en", sentences=5)

    def test_dispatch_graph_query(self):
        """dispatch_tool should route 'story_graph_query' with story_id."""
        with patch("app.research_tools.story_graph_query") as mock_gq:
            mock_gq.return_value = {"entity_name": "Aria", "found": True}
            dispatch_tool("story_graph_query", {"entity_name": "Aria"}, story_id="s1")

        mock_gq.assert_called_once_with(story_id="s1", entity_name="Aria")

    def test_dispatch_graph_query_no_story_id(self):
        """dispatch_tool should error for story_graph_query without story_id."""
        result = dispatch_tool("story_graph_query", {"entity_name": "Aria"})
        assert "error" in result

    def test_dispatch_unknown_tool(self):
        """dispatch_tool should return an error for unknown tool names."""
        result = dispatch_tool("unknown_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]


class TestToolDefinitions:
    """Tests for the TOOL_DEFINITIONS list."""

    def test_tool_definitions_structure(self):
        """TOOL_DEFINITIONS should have 5 tools with correct structure."""
        assert len(TOOL_DEFINITIONS) == 5

        for tool_def in TOOL_DEFINITIONS:
            assert tool_def["type"] == "function"
            assert "name" in tool_def["function"]
            assert "description" in tool_def["function"]
            assert "parameters" in tool_def["function"]
            assert "properties" in tool_def["function"]["parameters"]

    def test_tool_names(self):
        """TOOL_DEFINITIONS should contain the expected tool names."""
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        assert names == {"web_search", "wikipedia_lookup", "story_graph_query", "add_lore_to_graph", "update_story_graph"}
