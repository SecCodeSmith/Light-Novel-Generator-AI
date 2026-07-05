"""Neo4j story graph: schema, outline persistence, writer context, extraction merge,
consistency checks and visualization data.

Node labels: Story, Chapter, Character, Location, Event, Scene, Thread.
Every non-Story node carries story_id. Events have kind 'planned' (from the outline)
or 'happened' (extracted from written prose) and a global `order` for the timeline.
"""
import uuid

from neo4j import GraphDatabase

from . import config

_driver = None


def driver():
    global _driver
    if _driver is None:
        # notifications off: queries probing not-yet-written property keys
        # (e.g. death_chapter before any character dies) are expected here.
        _driver = GraphDatabase.driver(config.NEO4J_URI,
                                       auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                                       notifications_min_severity="OFF")
    return _driver


def run(query: str, **params):
    with driver().session() as session:
        return list(session.run(query, **params))


def init_schema() -> None:
    statements = [
        "CREATE CONSTRAINT story_id IF NOT EXISTS FOR (s:Story) REQUIRE s.id IS UNIQUE",
        "CREATE INDEX chapter_story IF NOT EXISTS FOR (c:Chapter) ON (c.story_id, c.number)",
        "CREATE INDEX character_story IF NOT EXISTS FOR (c:Character) ON (c.story_id, c.name)",
        "CREATE INDEX location_story IF NOT EXISTS FOR (l:Location) ON (l.story_id, l.name)",
        "CREATE INDEX event_story IF NOT EXISTS FOR (e:Event) ON (e.story_id, e.order)",
        "CREATE INDEX thread_story IF NOT EXISTS FOR (t:Thread) ON (t.story_id, t.id)",
    ]
    for stmt in statements:
        run(stmt)


# ------------------------------------------------------------------ stories

def create_story(title: str, description: str, num_chapters: int, language: str) -> str:
    story_id = uuid.uuid4().hex[:12]
    run("""
        CREATE (s:Story {id: $id, title: $title, description: $description,
                         num_chapters: $num_chapters, language: $language,
                         premise: '', style: '', created_at: timestamp()})
        """, id=story_id, title=title, description=description,
        num_chapters=num_chapters, language=language)
    return story_id


def list_stories() -> list[dict]:
    rows = run("""
        MATCH (s:Story)
        OPTIONAL MATCH (all_c:Chapter {story_id: s.id})
        OPTIONAL MATCH (written_c:Chapter {story_id: s.id, status: 'written'})
        RETURN s, count(DISTINCT written_c) AS written, count(DISTINCT all_c) AS total_chapters
        ORDER BY s.created_at DESC
        """)
    return [{**dict(row["s"]), "written_chapters": row["written"], "total_chapters": row["total_chapters"]} for row in rows]


def get_story(story_id: str) -> dict | None:
    rows = run("MATCH (s:Story {id: $id}) RETURN s", id=story_id)
    return dict(rows[0]["s"]) if rows else None


def get_chapters(story_id: str) -> list[dict]:
    rows = run("""
        MATCH (c:Chapter {story_id: $id}) RETURN c ORDER BY c.number
        """, id=story_id)
    out = []
    for row in rows:
        ch = dict(row["c"])
        ch.pop("text", None)  # keep listings light
        out.append(ch)
    return out


def get_chapter(story_id: str, number: int) -> dict | None:
    rows = run("MATCH (c:Chapter {story_id: $id, number: $n}) RETURN c",
               id=story_id, n=number)
    return dict(rows[0]["c"]) if rows else None


def save_chapter_text(story_id: str, number: int, text: str, summary: str) -> None:
    run("""
        MATCH (c:Chapter {story_id: $id, number: $n})
        SET c.text = $text, c.status = 'written',
            c.word_count = size(split($text, ' ')),
            c.written_summary = $summary
        """, id=story_id, n=number, text=text, summary=summary)


def delete_story(story_id: str) -> None:
    run("MATCH (n {story_id: $id}) DETACH DELETE n", id=story_id)
    run("MATCH (s:Story {id: $id}) DETACH DELETE s", id=story_id)


# ------------------------------------------------------------------ outline

def save_outline_base(story_id: str, data: dict) -> None:
    """Persist premise, cast, locations, relationships and threads from batch 1."""
    run("MATCH (s:Story {id: $id}) SET s.premise = $premise, s.style = $style",
        id=story_id, premise=data.get("premise", ""), style=data.get("style", ""))
    for ch in data.get("characters", []):
        run("""
            MERGE (c:Character {story_id: $id, name: $name})
            SET c.role = $role, c.description = $desc,
                c.traits = $traits, c.status = coalesce(c.status, 'alive'),
                c.notes = coalesce(c.notes, [])
            """, id=story_id, name=ch["name"], role=ch.get("role", ""),
            desc=ch.get("description", ""), traits=ch.get("traits", []))
    for loc in data.get("locations", []):
        run("""
            MERGE (l:Location {story_id: $id, name: $name})
            SET l.description = $desc
            """, id=story_id, name=loc["name"], desc=loc.get("description", ""))
    for rel in data.get("relationships", []):
        add_relationship(story_id, rel.get("a", ""), rel.get("b", ""), rel.get("type", "RELATES_TO"))
    for thread in data.get("threads", []):
        run("""
            CREATE (t:Thread {story_id: $id, id: $tid, description: $desc,
                              status: 'open', opened_chapter: $opened})
            """, id=story_id, tid=uuid.uuid4().hex[:8],
            desc=thread.get("description", ""), opened=int(thread.get("opened_chapter", 1)))


def save_outline_chapters(story_id: str, chapters: list[dict]) -> None:
    for ch in chapters:
        n = int(ch["number"])
        # Idempotent re-outline: drop this chapter's previous planned events/scenes.
        run("""
            MATCH (x {story_id: $id, chapter: $n})
            WHERE (x:Scene) OR (x:Event AND x.kind = 'planned')
            DETACH DELETE x
            """, id=story_id, n=n)
        run("""
            MATCH (s:Story {id: $id})
            MERGE (c:Chapter {story_id: $id, number: $n})
            SET c.title = $title, c.summary = $summary,
                c.status = coalesce(c.status, 'planned')
            MERGE (c)-[:PART_OF]->(s)
            """, id=story_id, n=n, title=ch.get("title", f"Chapter {n}"),
            summary=ch.get("summary", ""))
        for idx, ev in enumerate(ch.get("events", [])):
            _add_event(story_id, n, idx, ev, kind="planned")
        for idx, sc in enumerate(ch.get("scenes", [])):
            sid = uuid.uuid4().hex[:8]
            run("""
                MATCH (c:Chapter {story_id: $id, number: $n})
                CREATE (sc:Scene {story_id: $id, id: $sid, title: $title,
                                  summary: $summary, chapter: $n, order: $order})
                CREATE (sc)-[:PART_OF]->(c)
                """, id=story_id, n=n, sid=sid,
                title=sc.get("title", ""), summary=sc.get("summary", ""), order=idx)
            if sc.get("location"):
                run("""
                    MATCH (sc:Scene {story_id: $id, id: $sid})
                    MERGE (l:Location {story_id: $id, name: $loc})
                    MERGE (sc)-[:SET_IN]->(l)
                    """, id=story_id, sid=sid, loc=sc["location"])
            for name in sc.get("characters", []):
                run("""
                    MATCH (sc:Scene {story_id: $id, id: $sid})
                    MERGE (p:Character {story_id: $id, name: $name})
                    ON CREATE SET p.status = 'alive', p.role = 'minor', p.description = ''
                    MERGE (p)-[:APPEARS_IN]->(sc)
                    """, id=story_id, sid=sid, name=name)


def _add_event(story_id: str, chapter: int, idx: int, ev: dict, kind: str) -> str:
    eid = uuid.uuid4().hex[:8]
    run("""
        MATCH (c:Chapter {story_id: $id, number: $n})
        CREATE (e:Event {story_id: $id, id: $eid, description: $desc,
                         chapter: $n, order: $order, kind: $kind})
        CREATE (e)-[:OCCURS_IN]->(c)
        """, id=story_id, n=chapter, eid=eid, desc=ev.get("description", ""),
        order=chapter * 1000 + idx, kind=kind)
    for name in ev.get("characters", []):
        run("""
            MATCH (e:Event {story_id: $id, id: $eid})
            MATCH (c:Chapter {story_id: $id, number: $n})
            MERGE (p:Character {story_id: $id, name: $name})
            ON CREATE SET p.status = 'alive', p.role = 'minor', p.description = ''
            MERGE (e)-[:INVOLVES]->(p)
            MERGE (p)-[:APPEARS_IN]->(c)
            """, id=story_id, eid=eid, name=name, n=chapter)
    if ev.get("location"):
        run("""
            MATCH (e:Event {story_id: $id, id: $eid})
            MERGE (l:Location {story_id: $id, name: $loc})
            MERGE (e)-[:AT]->(l)
            """, id=story_id, eid=eid, loc=ev["location"])
    return eid


def add_relationship(story_id: str, a: str, b: str, rel_type: str) -> None:
    if not a or not b or a == b:
        return
    safe_type = "".join(ch for ch in rel_type.upper().replace(" ", "_") if ch.isalnum() or ch == "_")
    safe_type = safe_type or "RELATES_TO"
    run(f"""
        MERGE (x:Character {{story_id: $id, name: $a}})
        ON CREATE SET x.status = 'alive', x.role = 'minor', x.description = ''
        MERGE (y:Character {{story_id: $id, name: $b}})
        ON CREATE SET y.status = 'alive', y.role = 'minor', y.description = ''
        MERGE (x)-[:`{safe_type}`]->(y)
        """, id=story_id, a=a, b=b)

def get_story_bible(story_id: str) -> dict:
    """Cast, locations and open threads recorded so far in the graph — used to give
    the architect the established names/facts when planning a new batch of chapters,
    instead of re-deriving them from the raw pasted description each time."""
    characters = run("""
        MATCH (c:Character {story_id: $id})
        RETURN c.name AS name, c.role AS role, c.description AS description
        ORDER BY c.name
        """, id=story_id)
    locations = run("""
        MATCH (l:Location {story_id: $id})
        RETURN l.name AS name, l.description AS description
        ORDER BY l.name
        """, id=story_id)
    threads = run("""
        MATCH (t:Thread {story_id: $id, status: 'open'})
        RETURN t.description AS description, t.opened_chapter AS opened_chapter
        ORDER BY t.opened_chapter
        """, id=story_id)
    return {
        "characters": [dict(row) for row in characters],
        "locations": [dict(row) for row in locations],
        "open_threads": [dict(row) for row in threads],
    }


def save_lore_node(story_id: str, topic: str, description: str) -> None:
    if not topic or not description:
        return
    run("""
        MERGE (l:Lore {story_id: $id, topic: $topic})
        SET l.description = $desc
        """, id=story_id, topic=topic, desc=description)

def update_knowledge_graph(story_id: str, action: str, entity_data: dict) -> dict:
    """Helper for the agent to explicitly add entities to the graph."""
    try:
        if action == "add_character":
            name = entity_data.get("name")
            if not name: return {"error": "Name is required for character."}
            run("""
                MERGE (c:Character {story_id: $id, name: $name})
                SET c.role = coalesce($role, c.role, 'minor'),
                    c.description = coalesce($desc, c.description, ''),
                    c.status = coalesce(c.status, 'alive')
                """, id=story_id, name=name, role=entity_data.get("role"),
                desc=entity_data.get("description"))
            return {"success": True, "message": f"Character '{name}' added/updated."}
            
        elif action == "add_location":
            name = entity_data.get("name")
            if not name: return {"error": "Name is required for location."}
            run("""
                MERGE (l:Location {story_id: $id, name: $name})
                SET l.description = coalesce($desc, l.description, '')
                """, id=story_id, name=name, desc=entity_data.get("description"))
            return {"success": True, "message": f"Location '{name}' added/updated."}
            
        elif action == "add_relationship":
            a = entity_data.get("a")
            b = entity_data.get("b")
            rel_type = entity_data.get("type", "RELATES_TO")
            add_relationship(story_id, a, b, rel_type)
            return {"success": True, "message": f"Relationship '{rel_type}' created between '{a}' and '{b}'."}
            
        return {"error": f"Unknown action: {action}"}
    except Exception as exc:
        return {"error": str(exc)}

# ----------------------------------------------------------- writer context

def build_context(story_id: str, number: int) -> dict:
    """Everything the writer needs, pulled from the graph instead of raw prose."""
    story = get_story(story_id) or {}
    chapter = get_chapter(story_id, number) or {}
    chapter.pop("text", None)

    planned = run("""
        MATCH (e:Event {story_id: $id, chapter: $n, kind: 'planned'})
        OPTIONAL MATCH (e)-[:INVOLVES]->(p:Character)
        OPTIONAL MATCH (e)-[:AT]->(l:Location)
        WITH e, l, collect(DISTINCT p.name) AS characters
        ORDER BY e.order
        RETURN e.description AS description, characters, l.name AS location
        """, id=story_id, n=number)
    scenes = run("""
        MATCH (sc:Scene {story_id: $id, chapter: $n})
        OPTIONAL MATCH (p:Character)-[:APPEARS_IN]->(sc)
        OPTIONAL MATCH (sc)-[:SET_IN]->(l:Location)
        WITH sc, l, collect(DISTINCT p.name) AS characters
        ORDER BY sc.order
        RETURN sc.title AS title, sc.summary AS summary, l.name AS location, characters
        """, id=story_id, n=number)

    names = set()
    for row in planned:
        names.update(n for n in row["characters"] if n)
    for row in scenes:
        names.update(n for n in row["characters"] if n)
    if not names:  # fall back to main cast
        rows = run("""
            MATCH (c:Character {story_id: $id}) WHERE c.role IN ['protagonist', 'antagonist', 'ally']
            RETURN c.name AS name
            """, id=story_id)
        names = {row["name"] for row in rows}

    characters = []
    for name in sorted(names):
        rows = run("""
            MATCH (c:Character {story_id: $id, name: $name})
            OPTIONAL MATCH (c)-[r]-(o:Character {story_id: $id})
            RETURN c, collect(DISTINCT type(r) + ' ' + o.name) AS rels
            """, id=story_id, name=name)
        if rows:
            c = dict(rows[0]["c"])
            characters.append({
                "name": c.get("name"), "role": c.get("role", ""),
                "description": c.get("description", ""),
                "status": c.get("status", "alive"),
                "traits": c.get("traits", []),
                "recent_notes": (c.get("notes") or [])[-5:],
                "relationships": [r for r in rows[0]["rels"]
                                  if r and not r.startswith(("APPEARS_IN", "INVOLVES"))],
            })

    timeline = run("""
        MATCH (e:Event {story_id: $id, kind: 'happened'}) WHERE e.chapter < $n
        RETURN e.chapter AS chapter, e.description AS description
        ORDER BY e.order DESC LIMIT $limit
        """, id=story_id, n=number, limit=config.TIMELINE_EVENT_LIMIT)
    timeline = [dict(row) for row in reversed(timeline)]

    threads = run("""
        MATCH (t:Thread {story_id: $id, status: 'open'})
        RETURN t.id AS id, t.description AS description, t.opened_chapter AS opened_chapter
        ORDER BY t.opened_chapter
        """, id=story_id)

    prev = run("""
        MATCH (c:Chapter {story_id: $id, status: 'written'}) WHERE c.number < $n
        RETURN c.number AS number, c.title AS title,
               coalesce(c.written_summary, c.summary) AS summary
        ORDER BY c.number DESC LIMIT 3
        """, id=story_id, n=number)

    lore = run("""
        MATCH (l:Lore {story_id: $id})
        RETURN l.topic AS topic, l.description AS description
        """, id=story_id)

    return {
        "story": {"title": story.get("title"), "premise": story.get("premise"),
                  "style": story.get("style"), "language": story.get("language", "English"),
                  "num_chapters": story.get("num_chapters")},
        "chapter": chapter,
        "planned_events": [dict(row) for row in planned],
        "scenes": [dict(row) for row in scenes],
        "characters": characters,
        "timeline": timeline,
        "open_threads": [dict(row) for row in threads],
        "previous_chapters": [dict(row) for row in reversed(prev)],
        "lore": [dict(row) for row in lore],
    }


# ------------------------------------------------------- extraction merging

def apply_extraction(story_id: str, number: int, data: dict) -> None:
    for idx, ev in enumerate(data.get("events", [])):
        _add_event(story_id, number, 500 + idx, ev, kind="happened")
    for ch in data.get("new_characters", []):
        run("""
            MERGE (c:Character {story_id: $id, name: $name})
            ON CREATE SET c.status = 'alive', c.notes = []
            SET c.role = coalesce($role, c.role, 'minor'),
                c.description = coalesce($desc, c.description, '')
            """, id=story_id, name=ch["name"], role=ch.get("role"),
            desc=ch.get("description"))
    for upd in data.get("character_updates", []):
        name, change = upd.get("name"), upd.get("change", "")
        if not name:
            continue
        run("""
            MERGE (c:Character {story_id: $id, name: $name})
            ON CREATE SET c.status = 'alive', c.role = 'minor', c.description = ''
            SET c.notes = coalesce(c.notes, []) + [$note]
            """, id=story_id, name=name, note=f"ch{number}: {change}")
        status = (upd.get("status") or "").lower()
        if status in ("alive", "dead", "missing", "unknown"):
            extra = ", c.death_chapter = $n" if status == "dead" else ""
            run(f"""
                MATCH (c:Character {{story_id: $id, name: $name}})
                SET c.status = $status{extra}
                """, id=story_id, name=name, status=status, n=number)
    for rel in data.get("relationships", []):
        add_relationship(story_id, rel.get("a", ""), rel.get("b", ""), rel.get("type", "RELATES_TO"))
    for thread in data.get("threads_opened", []):
        desc = thread.get("description") if isinstance(thread, dict) else str(thread)
        chars = thread.get("characters", []) if isinstance(thread, dict) else []
        if desc:
            tid = uuid.uuid4().hex[:8]
            run("""
                CREATE (t:Thread {story_id: $id, id: $tid, description: $desc,
                                  status: 'open', opened_chapter: $n})
                """, id=story_id, tid=tid, desc=desc, n=number)
            for char_name in chars:
                if char_name:
                    run("""
                        MATCH (t:Thread {story_id: $id, id: $tid})
                        MATCH (c:Character {story_id: $id, name: $char_name})
                        MERGE (c)-[:INVOLVED_IN]->(t)
                        """, id=story_id, tid=tid, char_name=char_name)
    for tid in data.get("threads_resolved", []):
        run("""
            MATCH (t:Thread {story_id: $id, id: $tid})
            SET t.status = 'resolved', t.resolved_chapter = $n
            """, id=story_id, tid=str(tid), n=number)


# ------------------------------------------------ consistency + visualization

def consistency_checks(story_id: str) -> list[dict]:
    checks = []

    rows = run("""
        MATCH (c:Character {story_id: $id, status: 'dead'})<-[:INVOLVES]-(e:Event {story_id: $id})
        WHERE c.death_chapter IS NOT NULL AND e.chapter > c.death_chapter
        RETURN c.name AS name, c.death_chapter AS died, collect(e.chapter) AS chapters
        """, id=story_id)
    checks.append({
        "check": "Dead characters acting after death", "severity": "error",
        "items": [f"{r['name']} died in ch. {r['died']} but appears in events of ch. {sorted(set(r['chapters']))}"
                  for r in rows],
    })

    rows = run("""
        MATCH (t:Thread {story_id: $id, status: 'open'})
        MATCH (c:Chapter {story_id: $id, status: 'written'})
        WITH t, max(c.number) AS latest
        WHERE latest - t.opened_chapter >= 5
        RETURN t.description AS desc, t.opened_chapter AS opened, latest
        """, id=story_id)
    checks.append({
        "check": "Plot threads open for 5+ chapters", "severity": "warning",
        "items": [f"'{r['desc']}' (opened ch. {r['opened']}, still open at ch. {r['latest']})" for r in rows],
    })

    rows = run("""
        MATCH (e:Event {story_id: $id}) WHERE NOT (e)-[:INVOLVES]->(:Character)
        RETURN e.chapter AS chapter, e.description AS desc LIMIT 20
        """, id=story_id)
    checks.append({
        "check": "Events with no characters attached", "severity": "info",
        "items": [f"ch. {r['chapter']}: {r['desc']}" for r in rows],
    })

    rows = run("""
        MATCH (c:Character {story_id: $id})
        WHERE NOT (c)-[]-(:Character) AND NOT (c)<-[:INVOLVES]-(:Event)
        RETURN c.name AS name LIMIT 20
        """, id=story_id)
    checks.append({
        "check": "Isolated characters (no relations, no events)", "severity": "info",
        "items": [r["name"] for r in rows],
    })

    rows = run("""
        MATCH (c:Chapter {story_id: $id}) WITH collect(c.number) AS nums, max(c.number) AS top
        WHERE top IS NOT NULL
        UNWIND range(1, top) AS n WITH n, nums WHERE NOT n IN nums
        RETURN collect(n) AS missing
        """, id=story_id)
    missing = rows[0]["missing"] if rows else []
    checks.append({
        "check": "Missing chapter numbers", "severity": "warning",
        "items": [f"Chapter {n} is missing from the outline" for n in missing],
    })
    return checks


def entity_connections(story_id: str, name: str) -> dict:
    rows = run("""
        MATCH (c {story_id: $id}) WHERE toLower(c.name) = toLower($name)
        OPTIONAL MATCH (c)-[r]-(o)
        RETURN c, collect({type: type(r), dir: CASE WHEN startNode(r) = c THEN 'out' ELSE 'in' END,
                           other: coalesce(o.name, o.title, o.description, ''),
                           label: labels(o)[0]}) AS links
        """, id=story_id, name=name)
    if not rows:
        return {"found": False, "name": name}
    node = dict(rows[0]["c"])
    node.pop("text", None)
    return {"found": True, "node": node, "links": [l for l in rows[0]["links"] if l["type"]]}


def graph_data(story_id: str, limit: int = 400) -> dict:
    node_rows = run("""
        MATCH (n {story_id: $id})
        RETURN elementId(n) AS id, labels(n)[0] AS label,
               coalesce(n.name, n.title, left(n.description, 40), toString(n.number)) AS caption,
               coalesce(n.description, n.summary, '') AS detail
        LIMIT $limit
        """, id=story_id, limit=limit)
    edge_rows = run("""
        MATCH (a {story_id: $id})-[r]->(b {story_id: $id})
        RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type
        LIMIT $limit
        """, id=story_id, limit=limit * 3)
    return {
        "nodes": [{"id": r["id"], "label": r["caption"] or r["label"], "group": r["label"],
                   "detail": (r["detail"] or "")[:400]}
                  for r in node_rows],
        "edges": [{"from": r["source"], "to": r["target"], "label": r["type"]}
                  for r in edge_rows],
    }
