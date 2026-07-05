"""LLM access over any OpenAI-compatible API, plus an offline mock provider.

Set base_url to "mock" in the UI to run fully offline (demo / smoke tests).

Supports tool/function calling for agent mode: when tools are provided, the
response may include tool_calls instead of (or alongside) content.
"""
import json
import re
from dataclasses import dataclass, field

import httpx

MOCK_BASE_URL = "mock"


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    """A single tool call returned by the model."""
    id: str
    name: str
    arguments: dict
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ChatResult:
    """Structured result from an LLM chat call.

    Attributes:
        content: The text content of the response (may be empty if tool_calls present).
        tool_calls: List of tool calls the model wants to make (empty if none).
        finish_reason: The finish reason from the API ('stop', 'tool_calls', etc.).
    """
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


def parse_json_reply(text: str) -> dict:
    """Extract a JSON object from an LLM reply (tolerates fences, chatter, <think> blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def _close_json(s: str) -> str:
    """Append the closers a truncated JSON string is missing."""
    stack: list[str] = []
    in_str = esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    return s + ('"' if in_str else "") + "".join(
        "}" if c == "{" else "]" for c in reversed(stack))


def repair_truncated_json(text: str) -> dict | None:
    """Parse a reply whose JSON got cut off mid-generation (common with local
    models hitting their context/output limit). Drops the incomplete tail."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    pos = len(s)
    for _ in range(30):  # cut back brace by brace until it parses
        try:
            data = json.loads(_close_json(s[:pos]))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        pos = s.rfind("}", 0, pos - 1) + 1
        if pos <= 1:
            return None
    return None


def _parse_tool_calls(raw_tool_calls: list[dict]) -> list[ToolCall]:
    """Parse the tool_calls array from an OpenAI-compatible API response."""
    calls = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, ValueError):
            arguments = {"_raw": args_str}
        calls.append(ToolCall(
            id=tc.get("id", f"call_{name}"),
            name=name,
            arguments=arguments if isinstance(arguments, dict) else {"_raw": arguments},
            raw=tc,
        ))
    return calls


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> list[str]:
        try:
            resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=30)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot list models from {self.base_url}: {exc}") from exc
        data = resp.json().get("data", [])
        return sorted(m.get("id", "") for m in data if m.get("id"))

    def chat(self, model: str, messages: list[dict], temperature: float = 0.8,
             max_tokens: int = 4096, json_mode: bool = False,
             tools: list[dict] | None = None,
             tool_choice: str | dict | None = None,
             mock_hint: dict | None = None) -> ChatResult:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        # Providers differ: some reject response_format, o-series/gpt-5 style models
        # reject max_tokens (want max_completion_tokens) or a non-default temperature.
        # On a 400, adjust the payload based on the error text and retry.
        resp = None
        for _ in range(4):
            try:
                resp = httpx.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                  json=payload, timeout=600)
            except httpx.HTTPError as exc:
                raise LLMError(f"LLM API unreachable ({self.base_url}): {exc}") from exc
            if resp.status_code != 400:
                break
            err = resp.text
            if "max_completion_tokens" in err and "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
            elif "temperature" in err and "temperature" in payload:
                payload.pop("temperature")
            elif "response_format" in err and "response_format" in payload:
                payload.pop("response_format")
            elif "tools" in err and "tools" in payload:
                # Provider doesn't support tools — fall back to toolless mode
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
            else:
                break
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM API error {exc.response.status_code}: {exc.response.text[:400]}") from exc
        try:
            data = resp.json()
            choice = data["choices"][0]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Malformed LLM response: {resp.text[:400]}") from exc

        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason", "stop")

        # Parse tool calls if present
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = _parse_tool_calls(raw_tool_calls) if raw_tool_calls else []

        if not content.strip() and not tool_calls:
            reason = finish_reason
            raise LLMError(
                f"Model '{model}' returned empty content (finish_reason={reason}). "
                "Reasoning models can burn the whole token budget on thinking — "
                "pick a different model or a smaller chapter/outline batch.")
        return ChatResult(content=content, tool_calls=tool_calls, finish_reason=finish_reason)


class MockClient:
    """Deterministic offline provider so the whole pipeline can run without a real LLM."""

    CAST = [
        {"name": "Aria Vale", "role": "protagonist",
         "description": "A resourceful archivist who stumbles onto a forbidden truth.",
         "traits": ["curious", "stubborn", "loyal"]},
        {"name": "Corin Ashe", "role": "ally",
         "description": "A weary courier with a talent for being in the wrong place.",
         "traits": ["sardonic", "brave"]},
        {"name": "Magister Sorel", "role": "antagonist",
         "description": "Keeper of the sealed archive, willing to bury anyone who reads too much.",
         "traits": ["calculating", "patient"]},
    ]
    PLACES = [
        {"name": "The Sunken Archive", "description": "A library drowned a century ago, still dry inside."},
        {"name": "Lantern Row", "description": "A market street that never fully sleeps."},
    ]

    def list_models(self) -> list[str]:
        return ["mock-large", "mock-small"]

    def chat(self, model: str, messages: list[dict], temperature: float = 0.8,
             max_tokens: int = 4096, json_mode: bool = False,
             tools: list[dict] | None = None,
             tool_choice: str | dict | None = None,
             mock_hint: dict | None = None) -> ChatResult:
        hint = mock_hint or {}
        task = hint.get("task", "")

        # Simulate tool calls in agent mode for the "research" mock hint
        if task == "agent_research" and tools:
            return ChatResult(
                content="",
                tool_calls=[ToolCall(
                    id="mock_call_1",
                    name="web_search",
                    arguments={"query": hint.get("research_query", "drowned library mythology")},
                )],
                finish_reason="tool_calls",
            )

        if task == "outline":
            content = json.dumps(self._outline(hint))
        elif task == "write":
            content = self._chapter_prose(hint)
        elif task == "critic":
            content = json.dumps({"approved": True, "issues": [],
                                  "summary": "Draft is consistent with the recorded facts."})
        elif task == "extract":
            content = json.dumps(self._extraction(hint))
        elif task == "reconcile":
            content = json.dumps(self._reconcile(hint))
        else:
            content = "Mock reply."
        return ChatResult(content=content, tool_calls=[], finish_reason="stop")

    def _outline(self, hint: dict) -> dict:
        title = hint.get("title", "Untitled")
        if hint.get("analysis"):
            return {
                "premise": f"'{title}': a slow-burn mystery about a library that remembers what people paid to forget.",
                "style": "Atmospheric third-person past tense, measured pacing.",
                "characters": self.CAST,
                "locations": self.PLACES,
                "relationships": [
                    {"a": "Aria Vale", "b": "Corin Ashe", "type": "ALLY_OF"},
                    {"a": "Magister Sorel", "b": "Aria Vale", "type": "HUNTS"},
                ],
                "threads": [
                    {"description": "What was erased from the Sunken Archive, and why?",
                     "opened_chapter": 1},
                    {"description": "Who does Sorel answer to?", "opened_chapter": 2},
                ],
            }
        first, last = hint.get("first", 1), hint.get("last", 3)
        cast = [c["name"] for c in self.CAST]
        chapters = []
        for n in range(first, last + 1):
            chapters.append({
                "number": n,
                "title": f"Chapter {n}: The {['Discovery', 'Pursuit', 'Bargain', 'Descent', 'Reckoning'][n % 5]}",
                "summary": (f"Step {n} of the plot of '{title}'. Aria pushes deeper into the mystery "
                            f"while Sorel tightens his net; the cost of knowing grows."),
                "events": [
                    {"description": f"Aria uncovers clue #{n} about the sealed archive.",
                     "characters": [cast[0]], "location": self.PLACES[0]["name"]},
                    {"description": f"Sorel dispatches watchers after the intrusion in chapter {n}.",
                     "characters": [cast[2]], "location": self.PLACES[1]["name"]},
                ],
                "scenes": [
                    {"title": f"Stacks, level {n}", "summary": "Aria searches the drowned stacks.",
                     "location": self.PLACES[0]["name"], "characters": [cast[0], cast[1]]},
                ],
            })
        return {"chapters": chapters}

    def _chapter_prose(self, hint: dict) -> str:
        n = hint.get("number", 1)
        title = hint.get("chapter_title", f"Chapter {n}")
        events = hint.get("events", [])
        paras = [
            f"The morning of the {n}th day smelled of wet paper and old candle smoke. "
            f"Aria Vale counted her steps down into the Sunken Archive the way other people counted prayers.",
        ]
        for ev in events:
            paras.append(
                f"{ev} She wrote it down twice — once in the ledger she was allowed to keep, "
                f"and once in the one she was not. Corin watched the stairs and said nothing, "
                f"which from Corin was a whole speech."
            )
        paras.append(
            "Above them, in a room that had never been flooded and never would be, Magister Sorel "
            "turned a page and waited. Patience, he had found, drowned more people than water ever did."
        )
        paras.append(
            f"By the time the lanterns on Lantern Row guttered out, {title.lower()} had already cost "
            "someone more than they knew. Aria closed the ledger and did not sleep."
        )
        return "\n\n".join(paras)

    def _extraction(self, hint: dict) -> dict:
        n = hint.get("number", 1)
        events = hint.get("events", [])
        return {
            "chapter_summary": f"Aria advances the investigation (clue #{n}); Sorel responds by tightening surveillance.",
            "events": [{"description": e, "characters": ["Aria Vale"], "location": "The Sunken Archive"}
                       for e in events],
            "new_characters": [],
            "character_updates": [{"name": "Aria Vale",
                                   "change": f"Knows clue #{n}; increasingly watched."}],
            "relationships": [],
            "threads_opened": [],
            "threads_resolved": [],
        }

    def _reconcile(self, hint: dict) -> dict:
        connections = []
        for node in hint.get("unconnected", []):
            node_type, key = node.get("node_type"), node.get("key")
            if node_type == "Thread":
                connections.append({"node_type": "Thread", "key": key, "needed": True,
                                    "connect_to_type": "Character", "connect_to_key": "Aria Vale",
                                    "relationship": "INVOLVED_IN"})
            elif node_type == "Lore":
                connections.append({"node_type": "Lore", "key": key, "needed": True,
                                    "connect_to_type": "Location", "connect_to_key": self.PLACES[0]["name"],
                                    "relationship": "RELATES_TO"})
            else:
                connections.append({"node_type": node_type, "key": key, "needed": False})
        return {"connections": connections}


def make_client(cfg: dict):
    base_url = (cfg.get("base_url") or "").strip()
    if base_url.lower() == MOCK_BASE_URL:
        return MockClient()
    if not base_url:
        raise LLMError("LLM base URL is not configured. Open the Setup tab first.")
    return OpenAICompatClient(base_url, cfg.get("api_key", ""))
