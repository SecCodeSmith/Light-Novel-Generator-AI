# Light Novel Generator AI

A personal experiment in using LLMs to write a full-length novel: a **writer + critic
LLM pipeline**, a **Neo4j story knowledge graph** for long-range consistency, a **Redis
cache**, and a simple web UI built on the **Monaco editor** (the VS Code web editor
component).

## How it keeps a novel consistent

1. You write a story description → the **architect** LLM plans the outline in batches
   and the plot is stored as a graph in Neo4j: `Character`, `Event`, `Scene`,
   `Location`, `Chapter`, `Thread` nodes with relationships
   (`INVOLVES`, `APPEARS_IN`, `SET_IN`, `OCCURS_IN`, custom character relations…).
2. For each chapter the **writer** LLM gets its context **from the graph**, not from
   raw previous chapters: character sheets + statuses, the timeline of events that
   actually happened, open plot threads, and the cached tail of the previous chapter.
3. The **critic** LLM checks the draft against the same graph facts; if it rejects,
   the writer revises.
4. An **extractor** LLM mines the accepted chapter for new events, character changes
   (including deaths), relationships and threads, and merges them back into Neo4j.
5. Redis caches context packs (invalidated by a graph version bump), model lists,
   config, progress logs and write locks.
6. Consistency tools: graph visualization, Cypher checks (dead characters acting
   after death, stale open threads, isolated characters…), entity connection lookup.

## Run it docker Compose

```bash
docker compose up -d --build
```

Then open **http://localhost:8000**

- Neo4j browser: http://localhost:7474 (user `neo4j`, password `lightnovel123`)
- Redis: localhost:6379

## Using the app

1. **Setup tab** — pick a provider preset (OpenAI, OpenRouter, Groq, local Ollama /
   LM Studio, or the offline **Mock** demo), paste an API key, click *Load model
   list*, choose the **writer** and **critic** models, save.
2. **Story tab** — title + description + number of chapters → *Create story*.
   The outline and the graph are generated in the background.
3. **Write tab** — *Write next chapter* or *Write ALL remaining*. Watch the progress
   log; open chapters in the Monaco editor, edit and save. Export the novel as
   Markdown, all chapters as a **ZIP** of per-chapter Markdown files, or the full
   story as a **PDF** or **Word (.docx)** document.
4. **Graph tab** — visualize the story graph (filter node types via the legend
   checkboxes, hover a node for details, double-click a character/location for its
   connections), run consistency checks, look up any entity's connections.

> Tip: the **Mock** provider needs no key or internet and exercises the whole
> pipeline — useful for a smoke test.

## Architecture

```
browser (Monaco UI) ──► FastAPI app ──► OpenAI-compatible LLM API (writer/critic/extractor)
                            │ ├──► Neo4j  (story knowledge graph)
                            │ └──► Redis  (context cache, logs, locks, config)
                            └── serves static UI
```

| Service | Image | Port |
|---------|-------|------|
| app     | python:3.12-slim + FastAPI | 8000 |
| neo4j   | neo4j:5-community | 7474 / 7687 |
| redis   | redis:7-alpine | 6379 |
