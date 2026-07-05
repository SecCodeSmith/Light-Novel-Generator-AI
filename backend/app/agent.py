"""Agent loop engine for tool-calling LLM interactions.

When agent mode is enabled, replaces one-shot LLM calls with a multi-turn loop
that lets the model call research tools (web search, Wikipedia, graph queries)
before producing its final answer.

If the model never returns tool_calls (doesn't support function calling), the
loop completes in a single round — identical to the old one-shot behavior.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from . import cache, config
from .llm import ChatResult, LLMError, parse_json_reply, repair_truncated_json
from .research_tools import TOOL_DEFINITIONS, dispatch_tool

logger = logging.getLogger(__name__)


def _tool_defs_for_role(role: str) -> list[dict]:
    """Return the tool definitions appropriate for a pipeline role.

    - architect / writer: all tools (web_search, wikipedia_lookup, story_graph_query)
    - critic / extractor: no tools (they work from the text they're given)
    """
    if role in ("architect", "writer"):
        return list(TOOL_DEFINITIONS)
    return []


def agent_chat(
    client: Any,
    model: str,
    messages: list[dict],
    temperature: float = 0.8,
    max_tokens: int = 4096,
    json_mode: bool = False,
    tools: list[dict] | None = None,
    max_rounds: int | None = None,
    story_id: str | None = None,
    role: str = "llm",
    mock_hint: dict | None = None,
) -> str:
    """Run an agentic chat loop with tool calling.

    Keeps calling the model until it returns a text response (no more tool calls)
    or the max_rounds limit is hit.

    Args:
        client: An OpenAICompatClient or MockClient instance.
        model: The model ID to use.
        messages: The initial message list (system + user).
        temperature: Sampling temperature.
        max_tokens: Max tokens per model call.
        json_mode: If True, request JSON output format.
        tools: Tool definitions to pass to the model. None = no tools.
        max_rounds: Maximum number of tool-call rounds. Defaults to config value.
        story_id: Story ID for graph queries and logging.
        role: Pipeline role name for logging (e.g. 'architect', 'writer').
        mock_hint: Hint dict for the mock client.

    Returns:
        The final text content from the model.
    """
    if max_rounds is None:
        max_rounds = config.AGENT_MAX_TOOL_ROUNDS

    conversation = list(messages)
    total_tool_calls = 0

    for round_num in range(max_rounds + 1):
        result: ChatResult = client.chat(
            model=model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            tools=tools if round_num < max_rounds else None,  # last round: no tools, force answer
            mock_hint=mock_hint,
        )

        # Log the exchange
        prompt_summary = conversation[-1].get("content", "")[-2000:] if conversation else ""
        reply_summary = result.content[:2000] if result.content else f"[{len(result.tool_calls)} tool call(s)]"
        if story_id:
            cache.record_llm(story_id, f"{role} (round {round_num + 1})", prompt_summary, reply_summary)

        # If no tool calls, we're done
        if not result.has_tool_calls:
            if not result.content.strip():
                raise LLMError(
                    f"Model '{model}' returned empty content with no tool calls "
                    f"(finish_reason={result.finish_reason}). "
                    "Try a different model or disable agent mode."
                )
            return result.content

        # Process tool calls
        # Append the assistant message with tool calls (for the conversation history)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.content or None}
        assistant_msg["tool_calls"] = [
            tc.raw if tc.raw else {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in result.tool_calls
        ]
        conversation.append(assistant_msg)

        for tc in result.tool_calls:
            total_tool_calls += 1
            if story_id:
                cache.log(
                    story_id,
                    f"🔧 Agent ({role}): calling {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:200]})",
                )
            logger.info("Agent %s tool call: %s(%s)", role, tc.name, tc.arguments)

            # Execute the tool
            tool_result = dispatch_tool(tc.name, tc.arguments, story_id=story_id)

            if story_id:
                result_preview = json.dumps(tool_result, ensure_ascii=False)[:300]
                cache.log(story_id, f"   ↳ {tc.name} result: {result_preview}")

            # Append tool result to conversation
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        # Clear mock_hint after first round so mock doesn't loop on research
        if mock_hint:
            mock_hint = {k: v for k, v in mock_hint.items() if k != "task" or v != "agent_research"}

    # If we exhausted max_rounds, the last call was without tools, so we should
    # have gotten content. If not, raise an error.
    if story_id:
        cache.log(story_id, f"Agent ({role}): completed after {total_tool_calls} tool call(s) "
                            f"across {min(round_num + 1, max_rounds + 1)} round(s).")
    raise LLMError(
        f"Agent loop exhausted {max_rounds} rounds with {total_tool_calls} tool calls "
        f"without producing a final answer. The model may be stuck in a tool-calling loop."
    )


def agent_chat_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    max_rounds: int | None = None,
    story_id: str | None = None,
    role: str = "llm",
    mock_hint: dict | None = None,
) -> dict:
    """Agent chat that expects a JSON final response.

    Runs the agent loop, then parses the final text as JSON (with repair).
    Falls back to a retry prompt if JSON parsing fails.

    Returns:
        Parsed JSON dict from the model's final response.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    reply = agent_chat(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
        tools=tools,
        max_rounds=max_rounds,
        story_id=story_id,
        role=role,
        mock_hint=mock_hint,
    )

    # Try to parse JSON from the reply
    try:
        return parse_json_reply(reply)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to repair truncated JSON
    repaired = repair_truncated_json(reply)
    if repaired is not None:
        if story_id:
            cache.log(story_id, f"{role}: reply was cut off mid-JSON — repaired by dropping the incomplete tail.")
        return repaired

    # Retry: ask the model to fix its JSON (one-shot, no tools)
    retry_messages = messages + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
    ]
    retry_reply = client.chat(
        model=model,
        messages=retry_messages,
        temperature=0.2,
        max_tokens=max_tokens,
        json_mode=True,
        mock_hint=mock_hint,
    )
    if story_id:
        cache.record_llm(story_id, f"{role} (retry)", user, retry_reply.content)

    try:
        return parse_json_reply(retry_reply.content)
    except (json.JSONDecodeError, ValueError) as exc:
        repaired = repair_truncated_json(retry_reply.content)
        if repaired is not None:
            if story_id:
                cache.log(story_id, f"{role}: retry was also truncated — repaired.")
            return repaired
        raise LLMError(
            "Model did not return valid JSON after a retry — see the "
            "LLM traffic panel for the full reply. It began with: "
            f"{retry_reply.content[:300]!r}"
        ) from exc
