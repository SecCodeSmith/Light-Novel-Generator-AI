"""Research tools the LLM agent can call during story generation.

Each tool function accepts simple arguments and returns a JSON-serializable dict.
Network errors are caught and returned as error dicts so the agent loop never
crashes on a transient failure.

Tools:
  - web_search     — DuckDuckGo text search (MIT, no API key)
  - wikipedia_lookup — Wikipedia article summary via wikipedia-api (MIT)
  - story_graph_query — query the Neo4j story graph for entity connections
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- web search

def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web via DuckDuckGo HTML. Returns a list of results with title,
    url, and snippet. No API key required.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (1-10).

    Returns:
        A dict with 'results' list and 'query' echo.
    """
    max_results = max(1, min(int(max_results), 10))
    try:
        import httpx
        from bs4 import BeautifulSoup
        import urllib.parse
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        resp = httpx.get('https://html.duckduckgo.com/html/', params={'q': query}, headers=headers, timeout=10.0)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        for div in soup.find_all('div', class_='result'):
            if len(results) >= max_results:
                break
                
            a = div.find('a', class_='result__url')
            if not a:
                continue
                
            href = a.get('href', '')
            link = urllib.parse.unquote(href.split('uddg=')[1].split('&')[0]) if 'uddg=' in href else href
            
            title_tag = div.find('h2', class_='result__title')
            title = title_tag.text.strip() if title_tag else ''
            
            snippet_tag = div.find('a', class_='result__snippet')
            snippet = snippet_tag.text.strip() if snippet_tag else ''
            
            if link and title:
                results.append({
                    "title": title,
                    "url": link,
                    "snippet": snippet,
                })
                
        return {"query": query, "results": results}
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search failed for %r: %s", query, exc)
        return {"query": query, "results": [], "error": f"Search failed: {exc}"}


# --------------------------------------------------------- wikipedia lookup

def wikipedia_lookup(topic: str, language: str = "en", sentences: int = 5) -> dict[str, Any]:
    """Fetch a Wikipedia article summary.

    Args:
        topic: The article topic / title to look up.
        language: Wikipedia language code (default "en").
        sentences: Approximate number of sentences in the summary (1-20).

    Returns:
        A dict with 'title', 'summary', 'url', and 'sections' (top-level section titles).
    """
    sentences = max(1, min(int(sentences), 20))
    try:
        import wikipediaapi  # lazy import

        wiki = wikipediaapi.Wikipedia(
            user_agent="LightNovelGeneratorAI/1.0 (research-tool)",
            language=language,
        )
        page = wiki.page(topic)
        if not page.exists():
            return {"topic": topic, "found": False, "error": f"No Wikipedia article found for '{topic}'"}

        # Build a summary trimmed to roughly the requested sentence count
        full_summary = page.summary
        summary_sentences = full_summary.split(". ")
        trimmed = ". ".join(summary_sentences[:sentences])
        if not trimmed.endswith("."):
            trimmed += "."

        sections = [s.title for s in page.sections if s.title]

        return {
            "topic": topic,
            "found": True,
            "title": page.title,
            "summary": trimmed,
            "url": page.fullurl,
            "sections": sections[:15],  # cap to keep payload small
        }
    except ImportError:
        logger.warning("wikipedia-api is not installed — wikipedia_lookup unavailable")
        return {"topic": topic, "found": False, "error": "wikipedia-api package not installed"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("wikipedia_lookup failed for %r: %s", topic, exc)
        return {"topic": topic, "found": False, "error": f"Wikipedia lookup failed: {exc}"}


# -------------------------------------------------------- story graph query

def story_graph_query(story_id: str, entity_name: str) -> dict[str, Any]:
    """Query the story's Neo4j knowledge graph for an entity's connections.

    This lets the model look up character details, relationships, events,
    and locations from the story graph during writing.

    Args:
        story_id: The story UUID.
        entity_name: Character, location, or entity name to look up.

    Returns:
        A dict with 'found', 'node' details, and 'links' list.
    """
    try:
        from . import graph  # lazy to avoid circular import at module level

        result = graph.entity_connections(story_id, entity_name)
        if not result.get("found"):
            return {
                "entity_name": entity_name,
                "found": False,
                "error": f"'{entity_name}' not found in the story graph",
            }
        # Clean up the node for the agent — remove internal fields
        node = dict(result.get("node", {}))
        node.pop("story_id", None)
        links = result.get("links", [])
        return {
            "entity_name": entity_name,
            "found": True,
            "node": node,
            "links": [
                {"type": link["type"], "direction": link["dir"],
                 "other": link["other"], "label": link["label"]}
                for link in links if link.get("type")
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("story_graph_query failed for %r: %s", entity_name, exc)
        return {"entity_name": entity_name, "found": False, "error": f"Graph query failed: {exc}"}


# -------------------------------------------------- tool definitions for API

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for real-world information. Use this to research "
                "historical facts, cultural details, geography, science, or any "
                "real-world topic that could enrich the story's authenticity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10). Default 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_lookup",
            "description": (
                "Look up a topic on Wikipedia to get a factual summary. Use this "
                "for detailed information about historical events, places, people, "
                "scientific concepts, mythology, or cultural references."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The Wikipedia article title or topic to look up.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Wikipedia language code (e.g. 'en', 'ja', 'fr'). Default 'en'.",
                        "default": "en",
                    },
                    "sentences": {
                        "type": "integer",
                        "description": "Approximate number of summary sentences (1-20). Default 5.",
                        "default": 5,
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "story_graph_query",
            "description": (
                "Query the story's knowledge graph to look up a character, location, "
                "or entity. Returns their current status, description, relationships, "
                "and connected events. Use this to check facts about your story's "
                "own characters and world."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "The name of the character, location, or entity to look up.",
                    },
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_lore_to_graph",
            "description": (
                "Save researched lore, fan fiction rules, or world-building facts into the story's "
                "knowledge graph. This permanently establishes these rules as FACTS for the story."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The title or subject of the lore (e.g., 'Magic System', 'Elven Culture').",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed explanation of the rules, history, or facts.",
                    },
                },
                "required": ["topic", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_story_graph",
            "description": (
                "Explicitly add or update entities in the knowledge graph. "
                "Use this to proactively record characters, locations, or relationships."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add_character", "add_location", "add_relationship"],
                        "description": "The action to perform.",
                    },
                    "entity_data": {
                        "type": "object",
                        "description": "Data for the entity (e.g. {'name': 'John', 'role': 'protagonist', 'description': '...'}). For relationship: {'a': 'John', 'b': 'Mary', 'type': 'FRIENDS_WITH'}",
                    },
                },
                "required": ["action", "entity_data"],
            },
        },
    },
]

def update_story_graph(story_id: str, action: str, entity_data: dict) -> dict[str, Any]:
    try:
        from . import graph
        return graph.update_knowledge_graph(story_id, action, entity_data)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

def add_lore_to_graph(story_id: str, topic: str, description: str) -> dict[str, Any]:
    try:
        from . import graph
        graph.save_lore_node(story_id, topic, description)
        return {"success": True, "topic": topic, "message": "Lore saved to graph."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def dispatch_tool(tool_name: str, arguments: dict, story_id: str | None = None) -> dict[str, Any]:
    """Execute a tool call by name and return the result.

    Args:
        tool_name: One of 'web_search', 'wikipedia_lookup', 'story_graph_query', 'add_lore_to_graph'.
        arguments: The arguments dict parsed from the model's tool call.
        story_id: Required for story_graph_query and add_lore_to_graph.

    Returns:
        The tool result as a JSON-serializable dict.
    """
    if tool_name == "web_search":
        return web_search(
            query=arguments.get("query", ""),
            max_results=arguments.get("max_results", 5),
        )
    elif tool_name == "wikipedia_lookup":
        return wikipedia_lookup(
            topic=arguments.get("topic", ""),
            language=arguments.get("language", "en"),
            sentences=arguments.get("sentences", 5),
        )
    elif tool_name == "story_graph_query":
        if not story_id:
            return {"error": "story_id is required for story_graph_query"}
        return story_graph_query(
            story_id=story_id,
            entity_name=arguments.get("entity_name", ""),
        )
    elif tool_name == "add_lore_to_graph":
        if not story_id:
            return {"error": "story_id is required for add_lore_to_graph"}
        return add_lore_to_graph(
            story_id=story_id,
            topic=arguments.get("topic", ""),
            description=arguments.get("description", ""),
        )
    elif tool_name == "update_story_graph":
        if not story_id:
            return {"error": "story_id is required for update_story_graph"}
        return update_story_graph(
            story_id=story_id,
            action=arguments.get("action", ""),
            entity_data=arguments.get("entity_data", {}),
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}
