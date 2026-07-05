"""Story pipeline: architect (outline) -> writer -> critic -> reviser -> extractor.

The writer never sees full previous chapters: all continuity comes from the Neo4j
graph (character sheets, event timeline, open threads) plus the tail of the previous
chapter cached in Redis. After each accepted chapter, new facts are extracted and
merged back into the graph, and the cached contexts are invalidated via a version bump.

Agent mode: when enabled, the architect and writer LLM calls are routed through an
agentic loop that lets the model call research tools (web search, Wikipedia, graph
queries) before producing its answer. The critic and extractor remain one-shot.
"""
import json

from . import cache, config, graph
from .agent import _tool_defs_for_role, agent_chat, agent_chat_json
from .llm import ChatResult, LLMError, make_client, parse_json_reply, repair_truncated_json
from .research_tools import TOOL_DEFINITIONS

ARCHITECT_ANALYSIS_SYSTEM = """You are the story architect for a novel-length light novel.
Before any chapter is planned, fully read and analyze the author's ENTIRE pasted story
description end to end — do not skim or plan only from the opening. Reply with ONLY a
JSON object with keys: premise (string), style (string), characters (list of {name,
role, description, traits: [strings]}; role is one of protagonist/antagonist/ally/minor),
locations (list of {name, description}), relationships (list of {a, b, type} where type
is an UPPER_SNAKE verb like LOVES, HUNTS, SERVES), threads (list of {description,
opened_chapter}).
Extract EVERY character, location, relationship and plot thread implied ANYWHERE in the
description, including ones that only matter later — not just what appears in the
opening. If the pasted material contradicts itself (a character described two different
ways, conflicting timeline or world-rule statements, a name spelled differently, etc.),
resolve it yourself — prefer the most specific/most detailed statement — and record only
the resolved, consistent version; do not surface the contradiction as prose.
Do NOT include a chapters key: chapters are planned in a later step, once this base is
stored. No text outside the JSON object."""

ARCHITECT_ANALYSIS_SYSTEM_AGENT = ARCHITECT_ANALYSIS_SYSTEM + """

You have access to research tools. Use them to enrich your story planning:
- web_search: search the web for real-world facts, cultural details, historical context
- wikipedia_lookup: look up specific topics on Wikipedia for accurate reference material
- add_lore_to_graph: permanently save a researched fact/world rule so it is never
  researched again for this story
Before researching a topic, consider whether it's something you'd already reasonably
know or that's already implied by the pasted description — only call web_search /
wikipedia_lookup for things that truly need outside grounding. Every time research
turns up a fact worth keeping, call add_lore_to_graph immediately so later chapters
reuse it instead of re-searching the same topic.

If you need clarification from the author before outlining (e.g., asking about a specific
plot point, an unresolved contradiction, or a missing world rule), output a JSON with ONLY:
{"questions_for_user": ["Question 1", "Question 2"]}

Otherwise, after your research, output the standard analysis JSON object as described above."""

ARCHITECT_SYSTEM = """You are the story architect for a novel-length light novel.
The premise, cast, locations, relationships and open threads have already been
established in the story's knowledge graph. Reply with ONLY a JSON object with one key:
chapters — a list of {number, title, summary (2-3 sentences), events (2-4 of
{description, characters: [names], location}), scenes (1-3 of {title, summary,
location, characters: [names]})}.
Numbers must be exactly the requested range. Reuse existing character/location names
verbatim. No text outside the JSON object."""

ARCHITECT_SYSTEM_AGENT = ARCHITECT_SYSTEM + """

You have access to research tools. Use them to enrich your story planning:
- web_search: search the web for real-world facts, cultural details, historical context
- wikipedia_lookup: look up specific topics on Wikipedia for accurate reference material
- story_graph_query: check what's already known before researching — avoids redundant
  searches
- add_lore_to_graph: permanently save a researched fact/world rule so future chapters
  never need to research it again
Check story_graph_query / the FACTS you were given first; only research topics not
already covered. Save anything useful you find via add_lore_to_graph before finishing.

If you need clarification from the author before outlining (e.g., asking about a specific plot point or world rule), output a JSON with ONLY:
{"questions_for_user": ["Question 1", "Question 2"]}

Otherwise, after your research, output the standard outline JSON object as described above."""

WRITER_SYSTEM = """You are the novelist. Write the requested chapter as immersive prose.
Follow the plan and NEVER contradict the FACTS block (character status, past events,
relationships). Write in the story's language and style. Target about {words} words.
Output prose only: no headings, no chapter title, no notes, no markdown."""

WRITER_SYSTEM_AGENT = """You are the novelist. Write the requested chapter as immersive prose.
Follow the plan and NEVER contradict the FACTS block (character status, past events,
relationships). Write in the story's language and style. Target about {words} words.

You have access to research tools:
- web_search: search for real-world details to add authenticity (locations, customs, technology, etc.)
- wikipedia_lookup: look up specific topics for accurate descriptions
- story_graph_query: look up a character or entity in the story's knowledge graph to verify facts
- add_lore_to_graph: save a researched fact/world rule permanently so later chapters
  reuse it instead of re-researching the same topic

The FACTS block's lore_and_world_rules already holds everything researched so far —
check there (or via story_graph_query) before searching, and only research topics not
already covered. If you look something up and it's worth keeping, call add_lore_to_graph
with it right away. Use tools as needed to research real-world details that would
enrich the prose, but don't overdo it — focus on writing. After any research, output
prose only: no headings, no chapter title, no notes, no markdown."""

CRITIC_SYSTEM = """You are a strict continuity editor. You receive graph FACTS and a
chapter draft. Check: contradictions with established facts, dead/absent characters
acting, timeline violations, tone breaks, plot holes. Reply with ONLY JSON:
{"approved": true/false, "issues": ["specific issue", ...], "summary": "one sentence"}.
Approve unless there are real consistency problems."""

CRITIC_SYSTEM_AGENT = CRITIC_SYSTEM + """

You have access to one tool:
- story_graph_query: look up a character, location, or entity in the story's knowledge
  graph for its current status, relationships and connected events.
The FACTS block only lists the main cast and recent history — use story_graph_query to
check anything it doesn't cover (a minor character's last known status, an entity's
relationships) before deciding. You cannot edit the graph; you only verify it.

After checking anything you need, reply with ONLY the JSON verdict described above."""

EXTRACTOR_SYSTEM = """You extract story facts from a chapter so they can be stored in a
knowledge graph. Reply with ONLY JSON:
{"chapter_summary": "3-4 sentences",
 "events": [{"description", "characters": [names], "location"}],  // 3-6 key events that HAPPENED
 "new_characters": [{"name", "role", "description"}],
 "character_updates": [{"name", "change", "status"}],  // status only if it changed: alive/dead/missing
 "relationships": [{"a", "b", "type"}],  // new or changed, UPPER_SNAKE type
 "threads_opened": [{"description": "what the thread is about", "characters": ["Character Name 1"]}],
 "threads_resolved": ["thread id from the OPEN THREADS list"]}
Use existing character names verbatim. Only include what the chapter states."""

RECONCILE_SYSTEM = """You are the story architect reviewing the knowledge graph right
after outlining. You are given the story's premise, its chapter summaries, and a list
of graph nodes (lore entries, plot threads, characters, locations) that currently have
NO connection to anything else in the graph — usually lore researched along the way,
or threads/characters that were mentioned but never wired into a chapter.
For EACH node in unconnected_nodes, decide whether it is actually needed by this story.
Reply with ONLY JSON:
{"connections": [{"node_type": "Lore"|"Thread"|"Character"|"Location",
"key": "<the node's key, copied verbatim from the input>",
"needed": true or false,
"connect_to_type": "Character"|"Location"|"Thread"|"Chapter",  // required if needed
"connect_to_key": "<name, thread key, or chapter number of an EXISTING entity from "
"the premise or chapter summaries>",  // required if needed
"relationship": "UPPER_SNAKE type, e.g. RELATES_TO, SET_IN, INVOLVED_IN, ABOUT"  // required if needed
}, ...]}
One entry per input node. If a node isn't needed, set needed to false and omit the
connect_to_* fields — it is left alone, never deleted. Only connect to an entity that
already exists in the premise/chapter summaries; never invent a new name. No text
outside the JSON object."""


def _chat_json(client, model: str, system: str, user: str, temperature: float,
               max_tokens: int = 4096, mock_hint: dict | None = None,
               story_id: str | None = None, role: str = "llm") -> dict:
    """One-shot JSON chat call (no tools). Used by critic and extractor."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    result: ChatResult = client.chat(model=model, messages=messages, temperature=temperature,
                                     max_tokens=max_tokens, json_mode=True, mock_hint=mock_hint)
    reply = result.content
    if story_id:
        cache.record_llm(story_id, role, user, reply)
    try:
        return parse_json_reply(reply)
    except (json.JSONDecodeError, ValueError):
        pass
    repaired = repair_truncated_json(reply)
    if repaired is not None:
        if story_id:
            cache.log(story_id, f"{role}: reply was cut off mid-JSON (model output "
                                "limit) — repaired by dropping the incomplete tail.")
        return repaired
    retry = messages + [{"role": "assistant", "content": reply},
                        {"role": "user", "content": "That was not valid JSON. "
                         "Reply again with ONLY the JSON object."}]
    retry_result: ChatResult = client.chat(model=model, messages=retry, temperature=0.2,
                                           max_tokens=max_tokens, json_mode=True, mock_hint=mock_hint)
    retry_reply = retry_result.content
    if story_id:
        cache.record_llm(story_id, f"{role} (retry)", user, retry_reply)
    try:
        return parse_json_reply(retry_reply)
    except (json.JSONDecodeError, ValueError) as exc:
        repaired = repair_truncated_json(retry_reply)
        if repaired is not None:
            if story_id:
                cache.log(story_id, f"{role}: retry was also truncated — repaired.")
            return repaired
        raise LLMError("Model did not return valid JSON after a retry — see the "
                       "LLM traffic panel for the full reply. It began with: "
                       f"{retry_reply[:300]!r}") from exc


def _facts_block(ctx: dict) -> str:
    return json.dumps({
        "story": ctx["story"],
        "lore_and_world_rules": ctx.get("lore", []),
        "characters": ctx["characters"],
        "timeline_of_past_events": ctx["timeline"],
        "open_threads": ctx["open_threads"],
        "previous_chapter_summaries": ctx["previous_chapters"],
    }, ensure_ascii=False, indent=1)


def _reconcile_unconnected_nodes(story_id: str, client, model: str,
                                 chapter_summaries: list[str]) -> None:
    """Ask the architect to decide, for every graph node with zero edges (lore
    researched along the way, threads/characters that never made it into a chapter),
    whether it's needed — and if so, wire it into the graph. Left-alone nodes are
    never deleted."""
    unconnected = graph.find_unconnected_nodes(story_id)
    if not unconnected:
        return
    story = graph.get_story(story_id) or {}
    cache.log(story_id, f"Architect: found {len(unconnected)} unconnected node(s) in the "
                        "graph (lore/threads/etc.) — deciding whether to weave them in…")
    user = json.dumps({
        "premise": story.get("premise", ""),
        "chapter_summaries": chapter_summaries,
        "unconnected_nodes": unconnected,
    }, ensure_ascii=False, indent=1)
    decisions = _chat_json(client, model, RECONCILE_SYSTEM, user, temperature=0.2,
                           max_tokens=4096, story_id=story_id, role="architect (reconcile)",
                           mock_hint={"task": "reconcile", "unconnected": unconnected})
    connected, kept = 0, 0
    for item in decisions.get("connections", []):
        if not item.get("needed"):
            kept += 1
            continue
        result = graph.connect_unconnected_node(
            story_id, item.get("node_type", ""), item.get("key", ""),
            item.get("connect_to_type", ""), item.get("connect_to_key", ""),
            item.get("relationship", "RELATES_TO"),
        )
        if result.get("success"):
            connected += 1
            cache.log(story_id, f"Architect: connected {item.get('node_type')} "
                                f"'{item.get('key')}' → {item.get('connect_to_type')} "
                                f"'{item.get('connect_to_key')}' "
                                f"({item.get('relationship', 'RELATES_TO')}).")
        else:
            cache.log(story_id, f"Architect: could not connect {item.get('node_type')} "
                                f"'{item.get('key')}' — {result.get('error', 'unknown error')}.")
    if kept:
        cache.log(story_id, f"Architect: left {kept} node(s) unconnected — not needed for this story.")
    if connected:
        cache.log(story_id, f"Architect: wired {connected} previously-unconnected node(s) into the graph.")


def generate_outline(story_id: str) -> None:
    if not cache.acquire_lock(story_id):
        cache.log(story_id, "A job is already running for this story.")
        return
    cache.set_status(story_id, "outlining")
    try:
        story = graph.get_story(story_id)
        if not story:
            raise ValueError(f"Story {story_id} not found")
        cfg = cache.get_config()
        client = make_client(cfg)
        model = cfg["writer_model"] or "gpt-4o-mini"
        temperature = float(cfg.get("temperature", 0.8))
        total = int(story["num_chapters"])
        batch = config.OUTLINE_BATCH_SIZE
        
        # Load planned summaries from the graph instead of building them sequentially
        # so that when the outline loop resumes, it knows what was already planned.
        planned_chapters = graph.get_chapters(story_id)
        planned_summaries = [f"ch{c.get('number')}: {c.get('title', '')} — {c.get('summary', '')}" 
                             for c in planned_chapters]

        # Determine if agent mode is active
        use_agent = cache.is_agent_mode(cfg)
        tools = _tool_defs_for_role("architect") if use_agent else None

        if use_agent:
            cache.log(story_id, "🤖 Agent mode ON — architect can use research tools.")
        else:
            cache.log(story_id, "Agent mode OFF — using one-shot architect.")

        # Fully analyze the pasted story description ONCE, before any chapter is
        # planned, so the whole text (not just its opening) shapes the graph and
        # any internal contradictions in the pasted material are resolved up front.
        if not story.get("premise") and not planned_chapters:
            cache.log(story_id, f"Architect: analyzing the full story description "
                                f"using {model} @ {cfg['base_url']}…")
            analysis_system = ARCHITECT_ANALYSIS_SYSTEM_AGENT if use_agent else ARCHITECT_ANALYSIS_SYSTEM
            analysis_user = (f"Story title: {story['title']}\nLanguage: {story['language']}\n"
                             f"Total chapters: {total}\n\nStory description:\n{story['description']}")
            analysis_mock_hint = {"task": "outline", "analysis": True, "title": story["title"]}
            if use_agent:
                base_data = agent_chat_json(
                    client, model, analysis_system, analysis_user, temperature,
                    max_tokens=8192, tools=tools, story_id=story_id, role="architect (analysis)",
                    mock_hint=analysis_mock_hint,
                )
            else:
                base_data = _chat_json(client, model, analysis_system, analysis_user, temperature,
                                       max_tokens=8192, story_id=story_id, role="architect (analysis)",
                                       mock_hint=analysis_mock_hint)

            questions = base_data.get("questions_for_user")
            if questions and isinstance(questions, list):
                cache.log(story_id, f"Architect has {len(questions)} question(s) for the user.")
                cache.put_json(f"lng:story:{story_id}:questions", questions, ttl=86400 * 7)
                cache.set_status(story_id, "waiting_for_user")
                return  # Abort cleanly, wait for user to answer and re-trigger outline

            graph.save_outline_base(story_id, base_data)
            story = graph.get_story(story_id)  # premise/style now set
            cache.log(story_id, f"Architect: analysis complete — "
                                f"{len(base_data.get('characters', []))} character(s), "
                                f"{len(base_data.get('locations', []))} location(s), "
                                f"{len(base_data.get('threads', []))} thread(s) recorded in the graph.")

        system_prompt = ARCHITECT_SYSTEM_AGENT if use_agent else ARCHITECT_SYSTEM

        # Find the next unplanned chapter batch
        start_chapter = 1
        if planned_chapters:
            start_chapter = max(c["number"] for c in planned_chapters) + 1

        for first in range(start_chapter, total + 1, batch):
            last = min(first + batch - 1, total)
            cache.log(story_id, f"Architect: planning chapters {first}-{last} of {total} "
                                f"using {model} @ {cfg['base_url']}…")
            bible = graph.get_story_bible(story_id)
            cast_line = ", ".join(f"{c['name']} ({c['role']})" for c in bible["characters"]) or "(none yet)"
            locations_line = ", ".join(l["name"] for l in bible["locations"]) or "(none yet)"
            threads_line = "; ".join(t["description"] for t in bible["open_threads"]) or "(none yet)"
            user = (f"Story title: {story['title']}\nLanguage: {story['language']}\n"
                    f"Premise: {story.get('premise', '')}\n"
                    f"Established cast: {cast_line}\n"
                    f"Established locations: {locations_line}\n"
                    f"Open threads: {threads_line}\n"
                    "Existing chapter summaries:\n" + ("\n".join(planned_summaries) or "(none yet)") +
                    f"\n\nPlan chapters {first}-{last} of {total}."
                    + (f" The final chapter {total} must resolve the main plot." if last == total else "")
                    + " Include ONLY the 'chapters' key.")

            if use_agent:
                data = agent_chat_json(
                    client, model, system_prompt, user, temperature,
                    max_tokens=8192, tools=tools, story_id=story_id, role="architect",
                    mock_hint={"task": "outline", "first": first, "last": last,
                               "title": story["title"]},
                )
            else:
                data = _chat_json(client, model, system_prompt, user, temperature,
                                  max_tokens=8192, story_id=story_id, role="architect",
                                  mock_hint={"task": "outline", "first": first, "last": last,
                                             "title": story["title"]})

            # Check if architect needs user questions answered
            questions = data.get("questions_for_user")
            if questions and isinstance(questions, list):
                cache.log(story_id, f"Architect has {len(questions)} question(s) for the user.")
                cache.put_json(f"lng:story:{story_id}:questions", questions, ttl=86400 * 7)
                cache.set_status(story_id, "waiting_for_user")
                return  # Abort cleanly, wait for user to answer and re-trigger outline

            chapters = data.get("chapters", [])
            # A truncated reply may have lost some chapters — request just the gap.
            got = {int(c.get("number", 0)) for c in chapters}
            missing = sorted(set(range(first, last + 1)) - got)
            if missing:
                cache.log(story_id, f"Architect: chapters {missing} missing from the "
                                    "batch (truncated reply?) — requesting them again…")
                gap_user = (f"Story title: {story['title']}\n"
                            f"Premise: {story.get('premise', '')}\n"
                            "Continue the SAME story. Plan ONLY chapters "
                            f"{missing[0]}-{missing[-1]} of {total}. "
                            "Include ONLY the 'chapters' key.")
                if use_agent:
                    gap = agent_chat_json(
                        client, model, system_prompt, gap_user, temperature,
                        max_tokens=8192, tools=tools, story_id=story_id, role="architect",
                        mock_hint={"task": "outline", "first": missing[0],
                                   "last": missing[-1], "title": story["title"]},
                    )
                else:
                    gap = _chat_json(client, model, system_prompt, gap_user, temperature,
                                     max_tokens=8192, story_id=story_id, role="architect",
                                     mock_hint={"task": "outline", "first": missing[0],
                                                "last": missing[-1], "title": story["title"]})
                chapters += [c for c in gap.get("chapters", [])
                             if int(c.get("number", 0)) in missing]
            graph.save_outline_chapters(story_id, chapters)
            planned_summaries += [f"ch{c.get('number')}: {c.get('title', '')} — {c.get('summary', '')}"
                                  for c in chapters]

        _reconcile_unconnected_nodes(story_id, client, model, planned_summaries)

        cache.bump_graph_version(story_id)
        cache.set_status(story_id, "outlined")
        cache.log(story_id, f"Outline complete: {total} chapters planned, graph populated.")
    except (LLMError, Exception) as exc:  # noqa: BLE001 — surface everything to the UI log
        cache.set_status(story_id, "error")
        cache.log(story_id, f"ERROR during outline: {exc}")
    finally:
        cache.release_lock(story_id)


def _build_context_cached(story_id: str, number: int) -> dict:
    version = cache.graph_version(story_id)
    key = f"lng:story:{story_id}:ctx:{number}:v{version}"
    return cache.cached_json(key, config.CONTEXT_CACHE_TTL,
                             lambda: graph.build_context(story_id, number))


def write_one_chapter(story_id: str, number: int, client, cfg: dict) -> None:
    temperature = float(cfg.get("temperature", 0.8))
    writer_model = cfg["writer_model"] or "gpt-4o-mini"
    critic_model = cfg["critic_model"] or writer_model

    # Determine agent mode
    use_agent = cache.is_agent_mode(cfg)
    writer_tools = _tool_defs_for_role("writer") if use_agent else None
    
    target_words = int(cfg.get("chapter_target_words", config.CHAPTER_TARGET_WORDS))
    writer_system = (WRITER_SYSTEM_AGENT if use_agent else WRITER_SYSTEM).format(
        words=target_words
    )

    ctx = _build_context_cached(story_id, number)
    facts = _facts_block(ctx)
    chapter = ctx["chapter"]
    plan = json.dumps({"chapter": chapter, "planned_events": ctx["planned_events"],
                       "scenes": ctx["scenes"]}, ensure_ascii=False, indent=1)
    tail = cache.get_json(f"lng:story:{story_id}:tail") or ""
    mock_hint = {"task": "write", "number": number,
                 "chapter_title": chapter.get("title", f"Chapter {number}"),
                 "events": [e["description"] for e in ctx["planned_events"]]}

    mode_label = "🤖 Agent" if use_agent else "Writer"
    cache.log(story_id, f"{mode_label}: drafting chapter {number} '{chapter.get('title', '')}' "
                        f"using {writer_model}…")
    user = (f"FACTS (knowledge graph):\n{facts}\n\nCHAPTER PLAN:\n{plan}\n\n"
            + (f"END OF PREVIOUS CHAPTER (for prose continuity):\n…{tail}\n\n" if tail else "")
            + f"Write chapter {number} now.")

    if use_agent:
        draft = agent_chat(
            client=client, model=writer_model,
            messages=[{"role": "system", "content": writer_system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=8192,
            tools=writer_tools, story_id=story_id, role=f"writer ch{number}",
            mock_hint=mock_hint,
        )
        
        # Enforce minimum chapter size (target_words * 0.8)
        min_words = int(target_words * 0.8)
        words = len(draft.split())
        loops = 0
        messages_history = [{"role": "system", "content": writer_system}, {"role": "user", "content": user}, {"role": "assistant", "content": draft}]
        
        while words < min_words and loops < 3:
            loops += 1
            cache.log(story_id, f"Draft is {words} words (target ~{target_words}). Continuing chapter (part {loops+1})...")
            continue_prompt = (f"Your draft is {words} words. The target is {target_words} words. "
                               "Do NOT rewrite what you just wrote. Write the NEXT part of the chapter, "
                               "picking up exactly where you left off, to expand the story and hit the target length. "
                               "Output ONLY prose.")
            messages_history.append({"role": "user", "content": continue_prompt})
            
            part_draft = agent_chat(
                client=client, model=writer_model,
                messages=messages_history,
                temperature=temperature, max_tokens=8192,
                tools=writer_tools, story_id=story_id, role=f"writer ch{number} pt{loops+1}",
                mock_hint=mock_hint,
            )
            
            draft += "\n\n" + part_draft
            messages_history.append({"role": "assistant", "content": part_draft})
            words = len(draft.split())
            
    else:
        messages = [{"role": "system", "content": writer_system}, {"role": "user", "content": user}]
        result: ChatResult = client.chat(model=writer_model, messages=messages, temperature=temperature,
                                         max_tokens=8192, mock_hint=mock_hint)
        draft = result.content
        cache.record_llm(story_id, f"writer ch{number}", user, draft)

    cache.log(story_id, f"Critic: reviewing chapter {number}…")
    critic_user = f"FACTS:\n{facts}\n\nCHAPTER {number} DRAFT:\n{draft}"
    if use_agent:
        review = agent_chat_json(
            client, critic_model, CRITIC_SYSTEM_AGENT, critic_user, 0.2,
            max_tokens=4096, tools=_tool_defs_for_role("critic"), story_id=story_id,
            role=f"critic ch{number}", mock_hint={"task": "critic"},
        )
    else:
        review = _chat_json(client, critic_model, CRITIC_SYSTEM, critic_user,
                            temperature=0.2, story_id=story_id, role=f"critic ch{number}",
                            mock_hint={"task": "critic"})
    issues = review.get("issues") or []
    if not review.get("approved", True) and issues:
        cache.log(story_id, f"Critic found {len(issues)} issue(s); revising: "
                            + "; ".join(str(i) for i in issues[:3]))
        fix_user = (user + f"\n\nYOUR PREVIOUS DRAFT:\n{draft}\n\nA continuity editor "
                    "found these problems:\n- " + "\n- ".join(str(i) for i in issues)
                    + "\nRewrite the chapter fixing ALL of them. Output prose only.")

        if use_agent:
            draft = agent_chat(
                client=client, model=writer_model,
                messages=[{"role": "system", "content": writer_system},
                          {"role": "user", "content": fix_user}],
                temperature=temperature, max_tokens=8192,
                tools=writer_tools, story_id=story_id,
                role=f"writer ch{number} (revision)",
                mock_hint=mock_hint,
            )
        else:
            fix_result: ChatResult = client.chat(
                model=writer_model, temperature=temperature, max_tokens=8192,
                messages=[{"role": "system", "content": writer_system},
                          {"role": "user", "content": fix_user}],
                mock_hint=mock_hint,
            )
            draft = fix_result.content
            cache.record_llm(story_id, f"writer ch{number} (revision)", fix_user, draft)
    else:
        cache.log(story_id, f"Critic approved chapter {number}: {review.get('summary', 'ok')}")

    cache.log(story_id, f"Extractor: mining chapter {number} facts into the graph…")
    extraction = _chat_json(client, critic_model, EXTRACTOR_SYSTEM,
                            f"OPEN THREADS:\n{json.dumps(ctx['open_threads'])}\n\n"
                            f"KNOWN CHARACTERS: {[c['name'] for c in ctx['characters']]}\n\n"
                            f"CHAPTER {number} TEXT:\n{draft}",
                            temperature=0.2, story_id=story_id,
                            role=f"extractor ch{number}",
                            mock_hint={"task": "extract", "number": number,
                                       "events": mock_hint["events"]})

    graph.save_chapter_text(story_id, number, draft,
                            extraction.get("chapter_summary", ""))
    graph.apply_extraction(story_id, number, extraction)
    cache.bump_graph_version(story_id)
    cache.put_json(f"lng:story:{story_id}:tail", draft[-1500:], ttl=7 * 24 * 3600)
    words = len(draft.split())
    cache.log(story_id, f"Chapter {number} saved ({words} words); graph updated with "
                        f"{len(extraction.get('events', []))} events.")


def write_chapters(story_id: str, mode: str = "next") -> None:
    if not cache.acquire_lock(story_id):
        cache.log(story_id, "A writing job is already running for this story.")
        return
    try:
        cache.set_status(story_id, "writing")
        cfg = cache.get_config()
        client = make_client(cfg)
        pending = [c["number"] for c in graph.get_chapters(story_id)
                   if c.get("status") != "written"]
        if not pending:
            cache.log(story_id, "Nothing to write: all planned chapters are done.")
            cache.set_status(story_id, "complete")
            return
        targets = pending[:1] if mode == "next" else pending
        for number in targets:
            write_one_chapter(story_id, number, client, cfg)
        remaining = len(pending) - len(targets)
        cache.set_status(story_id, "complete" if remaining == 0 else "outlined")
        cache.log(story_id, "All chapters written — the novel is complete!"
                  if remaining == 0 else f"Paused. {remaining} chapter(s) still planned.")
    except (LLMError, Exception) as exc:  # noqa: BLE001
        cache.set_status(story_id, "error")
        cache.log(story_id, f"ERROR during writing: {exc}")
    finally:
        cache.release_lock(story_id)


def export_markdown(story_id: str) -> str:
    story = graph.get_story(story_id) or {}
    parts = [f"# {story.get('title', 'Untitled')}\n"]
    if story.get("premise"):
        parts.append(f"> {story['premise']}\n")
    for ch in graph.get_chapters(story_id):
        full = graph.get_chapter(story_id, ch["number"]) or {}
        if full.get("text"):
            parts.append(f"\n## {full.get('title', 'Chapter ' + str(ch['number']))}\n")
            parts.append(full["text"])
            parts.append("")
    return "\n".join(parts)
