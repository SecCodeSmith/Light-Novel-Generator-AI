#!/usr/bin/env python3
"""End-to-end smoke test against a running instance (uses the offline mock LLM).

    python3 scripts/smoke_test.py [base_url]      # default http://localhost:8000
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")


def call(method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def call_raw(path: str) -> tuple[bytes, str]:
    """GET a binary endpoint; returns (body, content_disposition)."""
    with urllib.request.urlopen(BASE + path, timeout=60) as resp:
        return resp.read(), resp.headers.get("Content-Disposition", "")


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def wait_status(story_id: str, want: str, timeout: int = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        p = call("GET", f"/api/stories/{story_id}/progress")
        if p["status"] == "error":
            for line in p["log"][-5:]:
                print("    log:", line["msg"])
            check(f"status reached '{want}'", False, "pipeline reported error")
        if p["status"] == want and not p["busy"]:
            return p
        time.sleep(2)
    check(f"status reached '{want}'", False, f"timed out after {timeout}s")


print(f"Smoke test against {BASE}")

h = call("GET", "/api/health")
check("health: app/neo4j/redis", h.get("neo4j") == "ok" and h.get("redis") == "ok", json.dumps(h))

# Remember the user's LLM config so we can put it back afterwards
# (api_key is returned masked and is left untouched in the store).
prev_cfg = call("GET", "/api/config")

call("POST", "/api/config", {"base_url": "mock",
                             "writer_model": "mock-large", "critic_model": "mock-small"})
models = call("GET", "/api/models")["models"]
check("mock model list", "mock-large" in models, str(models))

sid = call("POST", "/api/stories", {
    "title": "Smoke Test Novel",
    "description": "An archivist discovers a drowned library that erases memories.",
    "num_chapters": 3, "language": "English"})["id"]
check("story created", bool(sid), sid)

wait_status(sid, "outlined")
s = call("GET", f"/api/stories/{sid}")
check("outline: 3 chapters planned", len(s["chapters"]) == 3,
      f"{len(s['chapters'])} chapters, premise set: {bool(s['story'].get('premise'))}")

call("POST", f"/api/stories/{sid}/write?mode=all")
wait_status(sid, "complete")
s = call("GET", f"/api/stories/{sid}")
written = [c for c in s["chapters"] if c["status"] == "written"]
check("all 3 chapters written", len(written) == 3,
      f"word counts: {[c.get('word_count') for c in s['chapters']]}")

ch1 = call("GET", f"/api/stories/{sid}/chapters/1")
check("chapter 1 has prose", len(ch1.get("text", "")) > 200, f"{len(ch1.get('text', ''))} chars")

g = call("GET", f"/api/stories/{sid}/graph")
check("graph populated", len(g["nodes"]) >= 10 and len(g["edges"]) >= 10,
      f"{len(g['nodes'])} nodes, {len(g['edges'])} edges")

checks = call("GET", f"/api/stories/{sid}/consistency")["checks"]
check("consistency checks run", len(checks) >= 4,
      "; ".join(f"{c['check']}: {len(c['items'])}" for c in checks))

ent = call("GET", f"/api/stories/{sid}/entity?name=" + urllib.parse.quote("Aria Vale"))
check("entity lookup 'Aria Vale'", ent.get("found") is True,
      f"{len(ent.get('links', []))} connections")

export = call("GET", f"/api/stories/{sid}/export")
check("markdown export", isinstance(export, str) and export.startswith("#") and len(export) > 1000,
      f"{len(export)} chars")

zip_body, zip_disp = call_raw(f"/api/stories/{sid}/export/zip")
check("zip export (all chapters)", zip_body[:2] == b"PK" and "attachment" in zip_disp,
      f"{len(zip_body)} bytes, {zip_disp}")

pdf_body, pdf_disp = call_raw(f"/api/stories/{sid}/export/pdf")
check("pdf export (full story)", pdf_body[:5] == b"%PDF-" and "attachment" in pdf_disp,
      f"{len(pdf_body)} bytes, {pdf_disp}")

docx_body, docx_disp = call_raw(f"/api/stories/{sid}/export/docx")
check("docx export (full story)", docx_body[:2] == b"PK" and docx_disp.endswith('.docx"'),
      f"{len(docx_body)} bytes, {docx_disp}")

# Agent mode config toggle
agent_cfg = call("POST", "/api/config", {"agent_mode": "true"})
check("agent mode toggle on", agent_cfg.get("agent_mode") == "true")
agent_cfg = call("POST", "/api/config", {"agent_mode": "false"})
check("agent mode toggle off", agent_cfg.get("agent_mode") == "false")
agent_cfg = call("POST", "/api/config", {"agent_mode": "true"})  # restore

# Research API endpoint
research = call("POST", "/api/research", {"tool": "web_search", "query": "light novel genre"})
check("research API: web search", "query" in research and "results" in research,
      f"{len(research.get('results', []))} results")

research_wiki = call("POST", "/api/research", {"tool": "wikipedia_lookup",
                                                "query": "Light novel", "language": "en"})
check("research API: wikipedia lookup", "topic" in research_wiki,
      f"found={research_wiki.get('found', False)}")

ui = call("GET", "/")
check("web UI served", "Light Novel Generator" in str(ui) and "monaco" in str(ui).lower())

# A single $("missingId").onclick crashes the whole frontend script, so make
# sure every element id app.js references literally exists in the HTML.
app_js = call("GET", "/static/app.js")
wanted_ids = set(re.findall(r'\$\("([A-Za-z_]\w*)"\)', str(app_js)))
missing_ids = sorted(i for i in wanted_ids if f'id="{i}"' not in str(ui))
check("UI has every element app.js wires", not missing_ids, str(missing_ids) or "all present")

call("DELETE", f"/api/stories/{sid}")
check("cleanup: story deleted", call("GET", "/api/stories") is not None)

call("POST", "/api/config", {k: prev_cfg.get(k, "") for k in
                             ("base_url", "writer_model", "critic_model", "temperature")})
check("cleanup: LLM config restored", call("GET", "/api/config")["base_url"] == prev_cfg["base_url"])

print("\nALL CHECKS PASSED ✔")

