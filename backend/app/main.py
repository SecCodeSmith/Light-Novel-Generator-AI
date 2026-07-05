"""Light Novel Generator AI — FastAPI backend.

Endpoints are synchronous on purpose (FastAPI runs them in a threadpool):
the Neo4j / Redis / HTTP clients used are all sync, and long jobs run as
background tasks guarded by a per-story Redis lock.
"""
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import cache, exports, graph, story
from .llm import LLMError, make_client
from .research_tools import web_search, wikipedia_lookup

app = FastAPI(title="Light Novel Generator AI")


@app.on_event("startup")
def startup() -> None:
    for attempt in range(30):
        try:
            graph.init_schema()
            return
        except Exception:  # noqa: BLE001 — neo4j may still be booting
            time.sleep(2)
    raise RuntimeError("Could not reach Neo4j to initialize the schema")


# ------------------------------------------------------------------- config

class ConfigIn(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    writer_model: str | None = None
    critic_model: str | None = None
    temperature: str | None = None
    agent_mode: str | None = None
    chapter_target_words: str | None = None


def _masked_config() -> dict:
    cfg = cache.get_config()
    key = cfg.get("api_key", "")
    cfg["api_key_set"] = bool(key)
    cfg["api_key"] = (key[:4] + "…" + key[-4:]) if len(key) > 12 else ("set" if key else "")
    return cfg


@app.get("/api/config")
def get_config():
    return _masked_config()


@app.post("/api/config")
def set_config(body: ConfigIn):
    values = body.model_dump(exclude_none=True)
    # Ignore the masked placeholder the UI sends back unchanged.
    if "…" in values.get("api_key", "") or values.get("api_key") == "set":
        values.pop("api_key")
    cache.set_config(values)
    return _masked_config()


@app.get("/api/models")
def list_models():
    cfg = cache.get_config()
    key = f"lng:models:{cfg['base_url']}"
    try:
        return {"models": cache.cached_json(key, 600, lambda: make_client(cfg).list_models())}
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ------------------------------------------------------------------ stories

class StoryIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=10)
    num_chapters: int = Field(default=10, ge=1, le=60)
    language: str = "English"


@app.get("/api/stories")
def list_stories():
    return {"stories": graph.list_stories()}


@app.post("/api/stories")
def create_story(body: StoryIn, tasks: BackgroundTasks):
    story_id = graph.create_story(body.title, body.description,
                                  body.num_chapters, body.language)
    cache.log(story_id, f"Story '{body.title}' created ({body.num_chapters} chapters planned).")
    tasks.add_task(story.generate_outline, story_id)
    return {"id": story_id}


@app.get("/api/stories/{story_id}")
def get_story(story_id: str):
    s = graph.get_story(story_id)
    if not s:
        raise HTTPException(status_code=404, detail="Story not found")
    return {"story": s, "chapters": graph.get_chapters(story_id),
            "status": cache.get_status(story_id), "busy": cache.is_locked(story_id)}


@app.delete("/api/stories/{story_id}")
def delete_story(story_id: str):
    graph.delete_story(story_id)
    return {"ok": True}


@app.post("/api/stories/{story_id}/outline")
def regenerate_outline(story_id: str, tasks: BackgroundTasks):
    if not graph.get_story(story_id):
        raise HTTPException(status_code=404, detail="Story not found")
    if cache.is_locked(story_id):
        raise HTTPException(status_code=409, detail="A job is already running")
    tasks.add_task(story.generate_outline, story_id)
    return {"started": True}


@app.get("/api/stories/{story_id}/questions")
def get_questions(story_id: str):
    if not graph.get_story(story_id):
        raise HTTPException(status_code=404, detail="Story not found")
    q = cache.get_json(f"lng:story:{story_id}:questions")
    return {"questions": q or []}


class AnswersIn(BaseModel):
    answers: list[str]


@app.post("/api/stories/{story_id}/outline/answer")
def answer_questions(story_id: str, body: AnswersIn, tasks: BackgroundTasks):
    s = graph.get_story(story_id)
    if not s:
        raise HTTPException(status_code=404, detail="Story not found")
    q = cache.get_json(f"lng:story:{story_id}:questions") or []
    if not q:
        raise HTTPException(status_code=400, detail="No questions pending")
    
    # Format Q&A and append to the story description in Neo4j
    qa_block = "\n\n--- Author's Notes / Clarifications ---\n"
    for question, answer in zip(q, body.answers):
        qa_block += f"Q: {question}\nA: {answer}\n\n"
    
    new_desc = s.get("description", "") + qa_block
    graph.run("MATCH (s:Story {id: $id}) SET s.description = $desc", 
              id=story_id, desc=new_desc)
    
    # Clear the questions cache and restart outlining
    cache.r().delete(f"lng:story:{story_id}:questions")
    cache.set_status(story_id, "outlining")
    tasks.add_task(story.generate_outline, story_id)
    return {"started": True}


@app.post("/api/stories/{story_id}/write")
def write(story_id: str, tasks: BackgroundTasks, mode: str = "next"):
    if not graph.get_story(story_id):
        raise HTTPException(status_code=404, detail="Story not found")
    if cache.is_locked(story_id):
        raise HTTPException(status_code=409, detail="A writing job is already running")
    if mode not in ("next", "all"):
        raise HTTPException(status_code=422, detail="mode must be 'next' or 'all'")
    tasks.add_task(story.write_chapters, story_id, mode)
    return {"started": True, "mode": mode}


@app.get("/api/stories/{story_id}/progress")
def progress(story_id: str, offset: int = 0):
    return {"status": cache.get_status(story_id), "busy": cache.is_locked(story_id),
            "log": cache.get_log(story_id, offset), "offset": offset}


@app.get("/api/stories/{story_id}/llm")
def llm_traffic(story_id: str):
    """Raw prompt/reply pairs of recent model calls (to follow the model's output)."""
    return {"exchanges": cache.get_llm_log(story_id)}


@app.get("/api/stories/{story_id}/chapters/{number}")
def get_chapter(story_id: str, number: int):
    ch = graph.get_chapter(story_id, number)
    if not ch:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return ch


class ChapterTextIn(BaseModel):
    text: str


@app.put("/api/stories/{story_id}/chapters/{number}")
def save_chapter(story_id: str, number: int, body: ChapterTextIn):
    ch = graph.get_chapter(story_id, number)
    if not ch:
        raise HTTPException(status_code=404, detail="Chapter not found")
    graph.save_chapter_text(story_id, number, body.text,
                            ch.get("written_summary", ch.get("summary", "")))
    cache.bump_graph_version(story_id)
    cache.log(story_id, f"Chapter {number} manually edited and saved.")
    return {"ok": True}


@app.get("/api/stories/{story_id}/export", response_class=PlainTextResponse)
def export(story_id: str):
    if not graph.get_story(story_id):
        raise HTTPException(status_code=404, detail="Story not found")
    return story.export_markdown(story_id)


def _export_response(story_id: str, builder, suffix: str, media_type: str) -> Response:
    s = graph.get_story(story_id)
    if not s:
        raise HTTPException(status_code=404, detail="Story not found")
    data = builder(story_id)
    fname = exports.safe_filename(s.get("title", "story"))
    return Response(data, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{fname}{suffix}"'})


@app.get("/api/stories/{story_id}/export/zip")
def export_zip(story_id: str):
    """All written chapters as individual Markdown files plus the outline."""
    return _export_response(story_id, exports.chapters_zip, "-chapters.zip", "application/zip")


@app.get("/api/stories/{story_id}/export/pdf")
def export_pdf(story_id: str):
    """The full story as a single PDF (title page + one chapter per section)."""
    return _export_response(story_id, exports.story_pdf, ".pdf", "application/pdf")


@app.get("/api/stories/{story_id}/export/docx")
def export_docx(story_id: str):
    """The full story as a Word document."""
    return _export_response(
        story_id, exports.story_docx, ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# ------------------------------------------------------- graph & consistency

@app.get("/api/stories/{story_id}/graph")
def story_graph(story_id: str):
    version = cache.graph_version(story_id)
    # viz2: payload shape changed (node 'detail' field) — old cached entries are stale.
    # Short TTL: the graph version only bumps when a job finishes, but the graph
    # grows DURING outlining/writing — a long TTL shows an empty/stale graph.
    return cache.cached_json(f"lng:story:{story_id}:viz2:v{version}", 30,
                             lambda: graph.graph_data(story_id))


@app.get("/api/stories/{story_id}/consistency")
def consistency(story_id: str):
    return {"checks": graph.consistency_checks(story_id)}


@app.get("/api/stories/{story_id}/entity")
def entity(story_id: str, name: str):
    return graph.entity_connections(story_id, name)


# ------------------------------------------------------------ research tools

class ResearchQuery(BaseModel):
    """Manual research query from the UI."""
    tool: str = Field(description="Tool to use: 'web_search' or 'wikipedia_lookup'")
    query: str = Field(min_length=1, max_length=500,
                       description="Search query or Wikipedia topic")
    language: str = "en"
    max_results: int = Field(default=5, ge=1, le=10)


@app.post("/api/research")
def research(body: ResearchQuery):
    """Let the user manually trigger a research query from the UI."""
    if body.tool == "web_search":
        return web_search(query=body.query, max_results=body.max_results)
    elif body.tool == "wikipedia_lookup":
        return wikipedia_lookup(topic=body.query, language=body.language)
    else:
        raise HTTPException(status_code=422,
                            detail=f"Unknown tool '{body.tool}'. Use 'web_search' or 'wikipedia_lookup'.")


# ------------------------------------------------------------------- health

@app.get("/api/health")
def health():
    out = {"app": "ok"}
    try:
        graph.run("RETURN 1")
        out["neo4j"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["neo4j"] = f"error: {exc}"
    try:
        cache.r().ping()
        out["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["redis"] = f"error: {exc}"
    return out


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
