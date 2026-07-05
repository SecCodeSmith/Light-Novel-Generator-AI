/* Light Novel Generator AI — frontend logic */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { storyId: null, chapter: null, polling: null, network: null };

async function api(path, opts = {}) {
  const res = await fetch("/api" + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res.text();
}

/* ------------------------------------------------------------------ tabs */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "graph" && state.storyId) refreshGraph();
  };
});

/* ------------------------------------------------------------ editor (Monaco with fallback) */
const editor = {
  monaco: null, fallback: null, _pending: null,
  init() {
    const boot = () => {
      try {
        require.config({ paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs" } });
        require(["vs/editor/editor.main"], () => {
          this.monaco = monaco.editor.create($("editor"), {
            value: "", language: "markdown", theme: "vs-dark",
            wordWrap: "on", minimap: { enabled: false }, fontSize: 14,
            // The editor is created while the Write tab is hidden (0×0 px);
            // without this it never lays itself out once the tab is shown.
            automaticLayout: true,
          });
          if (this._pending !== null) {
            this.monaco.setValue(this._pending);
            this._pending = null;
          }
        }, () => this.useFallback());
      } catch (_) { this.useFallback(); }
    };
    // The Monaco loader script is async (CDN): poll for it instead of requiring
    // it to be there already, and fall back to a textarea if it never arrives.
    const start = Date.now();
    const tryBoot = () => {
      if (typeof require !== "undefined") boot();
      else if (Date.now() - start > 5000) this.useFallback();
      else setTimeout(tryBoot, 250);
    };
    tryBoot();
    setTimeout(() => { if (!this.monaco && !this.fallback) this.useFallback(); }, 10000);
  },
  useFallback() {
    if (this.fallback) return;
    const ta = document.createElement("textarea");
    ta.className = "fallback";
    $("editor").appendChild(ta);
    this.fallback = ta;
    if (this._pending !== null) {
      this.fallback.value = this._pending;
      this._pending = null;
    }
  },
  set(text) {
    if (this.monaco) this.monaco.setValue(text);
    else if (this.fallback) this.fallback.value = text;
    else this._pending = text;
  },
  get() {
    if (this.monaco) return this.monaco.getValue();
    if (this.fallback) return this.fallback.value;
    return this._pending || "";
  }
};
editor.init();

/* ----------------------------------------------------------------- setup */
$("preset").onchange = () => { $("baseUrl").value = $("preset").value; };

async function loadConfig() {
  const cfg = await api("/config");
  $("baseUrl").value = cfg.base_url || "";
  $("apiKey").value = cfg.api_key || "";
  $("temperature").value = cfg.temperature || "0.8";
  $("chapterTargetWords").value = cfg.chapter_target_words || "1800";
  setSelect("writerModel", cfg.writer_model);
  setSelect("criticModel", cfg.critic_model);
  // Agent mode toggle
  const agentOn = (cfg.agent_mode || "true").toLowerCase() === "true";
  $("agentMode").checked = agentOn;
  updateAgentBadge(agentOn);
}

function setSelect(id, value) {
  const sel = $(id);
  if (value && ![...sel.options].some((o) => o.value === value)) {
    sel.add(new Option(value, value));
  }
  if (value) sel.value = value;
}

function updateAgentBadge(on) {
  const badge = $("agentBadge");
  badge.textContent = on ? "🤖 agent on" : "🤖 agent off";
  badge.classList.toggle("active", on);
}

$("agentMode").onchange = () => updateAgentBadge($("agentMode").checked);

$("saveConfig").onclick = async () => {
  try {
    await api("/config", { method: "POST", body: {
      base_url: $("baseUrl").value.trim(), api_key: $("apiKey").value.trim(),
      writer_model: $("writerModel").value, critic_model: $("criticModel").value,
      temperature: $("temperature").value,
      chapter_target_words: $("chapterTargetWords").value,
      agent_mode: $("agentMode").checked ? "true" : "false",
    }});
    $("configMsg").textContent = "✔ saved";
    setTimeout(() => { $("configMsg").textContent = ""; }, 2500);
  } catch (e) { $("configMsg").textContent = "✖ " + e.message; }
};

$("loadModels").onclick = async () => {
  $("modelsMsg").textContent = "loading…";
  try {
    await api("/config", { method: "POST", body: {
      base_url: $("baseUrl").value.trim(), api_key: $("apiKey").value.trim() } });
    const { models } = await api("/models");
    for (const id of ["writerModel", "criticModel"]) {
      const sel = $(id); const prev = sel.value;
      sel.innerHTML = "";
      models.forEach((m) => sel.add(new Option(m, m)));
      if (prev && models.includes(prev)) sel.value = prev;
    }
    $("modelsMsg").textContent = `✔ ${models.length} models — pick writer & critic, then Save`;
  } catch (e) { $("modelsMsg").textContent = "✖ " + e.message; }
};

/* ----------------------------------------------------------------- story */
async function refreshStoryList() {
  const { stories } = await api("/stories");
  const sel = $("storySelect");
  const prev = state.storyId || "";
  sel.innerHTML = '<option value="">— open a story —</option>';
  stories.forEach((s) => sel.add(new Option(
    `${s.title} (${s.written_chapters}/${s.total_chapters || s.num_chapters} ch)`, s.id)));
  sel.value = prev;
}

$("storySelect").onchange = () => {
  state.storyId = $("storySelect").value || null;
  state.chapter = null;
  state.graphData = null;
  editor.set("");
  if (state.storyId) { loadStory(); startPolling(); }
};

$("createStory").onclick = async () => {
  try {
    const { id } = await api("/stories", { method: "POST", body: {
      title: $("stTitle").value.trim() || "Untitled",
      description: $("stDescription").value.trim(),
      num_chapters: parseInt($("stChapters").value, 10) || 10,
      language: $("stLanguage").value.trim() || "English",
    }});
    state.storyId = id;
    await refreshStoryList();
    $("storySelect").value = id;
    startPolling();
    document.querySelector('[data-tab="write"]').click();
  } catch (e) { alert("Create failed: " + e.message); }
};

$("regenOutline").onclick = async () => {
  if (!state.storyId) { alert("Open a story first."); return; }
  try {
    await api(`/stories/${state.storyId}/outline`, { method: "POST" });
    startPolling();
  } catch (e) { alert("Regenerate failed: " + e.message); }
};

$("deleteStory").onclick = async () => {
  if (!state.storyId || !confirm("Delete this story and its whole graph?")) return;
  await api("/stories/" + state.storyId, { method: "DELETE" });
  state.storyId = null; state.chapter = null;
  editor.set(""); $("outlineBox").textContent = "No story loaded.";
  $("chapterList").innerHTML = ""; $("logBox").textContent = "";
  refreshStoryList();
};

async function loadStory() {
  if (!state.storyId) return;
  const data = await api("/stories/" + state.storyId);
  renderOutline(data);
  renderChapterList(data.chapters);
  setBadge(data.status, data.busy);
}

function renderOutline(data) {
  const s = data.story;
  let html = "";
  if (s.premise) html += `<p><b>Premise:</b> ${esc(s.premise)}</p>`;
  if (s.style) html += `<p><b>Style:</b> ${esc(s.style)}</p>`;
  html += data.chapters.map((c) =>
    `<div class="outline-ch"><b>${c.number}. ${esc(c.title || "")}</b>
     <span class="st ${c.status}">${c.status}</span><br>${esc(c.summary || "")}</div>`).join("");
  $("outlineBox").innerHTML = html || "Outline is being generated…";
}

function renderChapterList(chapters) {
  $("chapterList").innerHTML = chapters.map((c) =>
    `<div class="chapter-item ${c.number === state.chapter ? "sel" : ""}" data-n="${c.number}">
       ${c.number}. ${esc(c.title || "…")}
       <span class="st ${c.status}">${c.status}${c.word_count ? " · " + c.word_count + "w" : ""}</span>
     </div>`).join("");
  document.querySelectorAll(".chapter-item").forEach((el) => {
    el.onclick = () => openChapter(parseInt(el.dataset.n, 10));
  });
}

const PLACEHOLDER_PREFIX = "(not written yet — planned summary)";

async function openChapter(n) {
  state.chapter = n;
  try {
    const ch = await api(`/stories/${state.storyId}/chapters/${n}`);
    $("chapterLabel").textContent = `Chapter ${n}: ${ch.title || ""} [${ch.status}]`;
    const textContent = (ch.text !== null && ch.text !== undefined)
      ? ch.text : PLACEHOLDER_PREFIX + "\n\n" + (ch.summary || "");
    editor.set(textContent);
  } catch (e) {
    $("chapterLabel").textContent = `Chapter ${n}: ✖ could not load (${e.message})`;
  }
  document.querySelectorAll(".chapter-item").forEach((el) =>
    el.classList.toggle("sel", parseInt(el.dataset.n, 10) === n));
}

$("saveChapter").onclick = async () => {
  if (!state.storyId || state.chapter == null) return;
  await api(`/stories/${state.storyId}/chapters/${state.chapter}`,
    { method: "PUT", body: { text: editor.get() } });
  $("writeMsg").textContent = `✔ chapter ${state.chapter} saved`;
  setTimeout(() => { $("writeMsg").textContent = ""; }, 2500);
};

/* ----------------------------------------------------------------- write */
async function startWrite(mode) {
  if (!state.storyId) { alert("Create or open a story first."); return; }
  try {
    await api(`/stories/${state.storyId}/write?mode=${mode}`, { method: "POST" });
    startPolling();
  } catch (e) { $("writeMsg").textContent = "✖ " + e.message; }
}
$("writeNext").onclick = () => startWrite("next");
$("writeAll").onclick = () => startWrite("all");

$("refreshLLM").onclick = async () => {
  if (!state.storyId) { alert("Open a story first."); return; }
  const { exchanges } = await api(`/stories/${state.storyId}/llm`);
  if (!exchanges.length) {
    $("llmBox").innerHTML = '<p class="muted">No model calls recorded yet for this story.</p>';
    return;
  }
  $("llmBox").innerHTML = exchanges.slice().reverse().map((e) => `
    <details class="llm">
      <summary>[${new Date(e.t * 1000).toLocaleTimeString()}] <b>${esc(e.role)}</b>
        — reply ${e.reply.length} chars</summary>
      <h4>prompt (tail)</h4><pre>${esc(e.prompt)}</pre>
      <h4>raw reply</h4><pre>${esc(e.reply)}</pre>
    </details>`).join("");
};

$("exportBtn").onclick = () => {
  if (!state.storyId) return;
  window.open(`/api/stories/${state.storyId}/export`, "_blank");
};

function download(path) {
  if (!state.storyId) { alert("Open a story first."); return; }
  const a = document.createElement("a");
  a.href = `/api/stories/${state.storyId}${path}`;
  a.download = "";                     // server sets the filename
  document.body.appendChild(a);
  a.click();
  a.remove();
}
$("exportZip").onclick = () => download("/export/zip");
$("exportPdf").onclick = () => download("/export/pdf");
$("exportDocx").onclick = () => download("/export/docx");

function setBadge(status, busy) {
  const b = $("statusBadge");
  b.textContent = status + (busy ? " ⏳" : "");
  b.className = "badge " + (busy || status === "outlining" || status === "writing" ? "busy"
    : status === "error" ? "error" : status === "complete" ? "ok" : "");
}

function startPolling() {
  if (state.polling) clearInterval(state.polling);
  state.polling = setInterval(poll, 2000);
  poll();
}

let lastLogLen = 0;
async function poll() {
  if (!state.storyId) return;
  try {
    const p = await api(`/stories/${state.storyId}/progress`);
    setBadge(p.status, p.busy);
    const box = $("logBox");
    // Format log lines with tool call highlighting
    box.innerHTML = p.log.map((l) => {
      const time = new Date(l.t * 1000).toLocaleTimeString();
      const msg = esc(l.msg);
      if (l.msg.startsWith("🔧")) {
        return `<span class="tool-call">[${time}] ${msg}</span>`;
      } else if (l.msg.startsWith("   ↳")) {
        return `<span class="tool-result">[${time}] ${msg}</span>`;
      }
      return `[${time}] ${msg}`;
    }).join("\n");
    if (p.log.length !== lastLogLen) {
      box.scrollTop = box.scrollHeight;
      lastLogLen = p.log.length;
      loadStory();      // refresh outline / chapter statuses as work progresses
      refreshStoryList();
      // The open chapter may just have been written: reload it, but only while
      // the editor still shows the plan placeholder (never clobber user edits).
      if (state.chapter != null && editor.get().startsWith(PLACEHOLDER_PREFIX)) {
        openChapter(state.chapter);
      }
    }
    if (!p.busy && !["outlining", "writing"].includes(p.status)) {
      if (p.status === "waiting_for_user") {
        clearInterval(state.polling); state.polling = null;
        showQuestionsPanel();
      } else {
        clearInterval(state.polling); state.polling = null;
      }
    }
  } catch (_) { /* transient */ }
}

async function showQuestionsPanel() {
  const { questions } = await api(`/stories/${state.storyId}/questions`);
  if (!questions || !questions.length) return;
  const list = $("questionsList");
  list.innerHTML = questions.map((q, i) => `
    <div style="margin-top: 1rem;">
      <label><b>${esc(q)}</b>
        <textarea id="qAns_${i}" rows="3" style="width: 100%; margin-top: 0.5rem;" placeholder="Your answer..."></textarea>
      </label>
    </div>
  `).join("");
  $("questionsPanel").style.display = "block";
  
  $("submitAnswers").onclick = async () => {
    $("submitAnswers").disabled = true;
    $("submitAnswers").textContent = "Submitting...";
    try {
      const answers = questions.map((_, i) => $(`qAns_${i}`).value.trim() || "No answer provided.");
      await api(`/stories/${state.storyId}/outline/answer`, {
        method: "POST", body: { answers }
      });
      $("questionsPanel").style.display = "none";
      startPolling();
    } catch (e) {
      alert("Failed to submit answers: " + e.message);
    } finally {
      $("submitAnswers").disabled = false;
      $("submitAnswers").textContent = "Submit Answers & Resume";
    }
  };
}

/* ----------------------------------------------------------------- graph */
const GROUP_COLORS = {
  Character: "#6fc3df", Event: "#dcdcaa", Scene: "#c586c0",
  Location: "#4ec9b0", Chapter: "#569cd6", Thread: "#f48771", Lore: "#ce9178",
};
state.graphData = null;

function graphMessage(html) {
  if (state.network) { state.network.destroy(); state.network = null; }
  $("graphCanvas").innerHTML = `<p class="muted" style="padding:20px">${html}</p>`;
}

async function refreshGraph() {
  if (!state.storyId) { graphMessage("Open a story first."); return; }
  graphMessage("Loading graph…");
  try {
    state.graphData = await api(`/stories/${state.storyId}/graph`);
  } catch (e) {
    graphMessage("✖ Could not load the graph: " + esc(e.message));
    return;
  }
  drawGraph();
}

function nodeTooltip(text) {
  const div = document.createElement("div");
  div.textContent = text;                       // textContent: LLM output stays inert
  div.style.maxWidth = "360px";
  div.style.whiteSpace = "pre-wrap";
  return div;
}

function drawGraph() {
  const data = state.graphData;
  if (!data) return;
  if (typeof vis === "undefined") {
    graphMessage("vis-network CDN unavailable — graph view disabled (data: "
      + data.nodes.length + " nodes, " + data.edges.length + " edges)");
    return;
  }
  const groups = new Set([...document.querySelectorAll(".gfilter:checked")].map((c) => c.value));
  const showEdgeLabels = $("edgeLabels").checked;
  const visible = data.nodes.filter((n) => groups.has(n.group) || !(n.group in GROUP_COLORS));
  if (!visible.length) {
    graphMessage(data.nodes.length
      ? "All node types are filtered out — tick some boxes below."
      : "The graph is empty — generate an outline first.");
    $("graphMeta").textContent = "";
    return;
  }
  const ids = new Set(visible.map((n) => n.id));
  const nodes = visible.map((n) => ({
    id: n.id, label: n.label, group: n.group,
    title: n.detail ? nodeTooltip(n.detail) : undefined,
    color: { background: "#2d2d30", border: GROUP_COLORS[n.group] || "#888",
             highlight: { background: "#094771", border: GROUP_COLORS[n.group] || "#888" } },
    font: { color: GROUP_COLORS[n.group] || "#d4d4d4", size: 16 }, shape: "box",
  }));
  const edges = data.edges
    .filter((e) => ids.has(e.from) && ids.has(e.to))
    .map((e) => ({
      from: e.from, to: e.to, label: showEdgeLabels ? e.label : undefined,
      arrows: "to", color: { color: "#555" },
      font: { color: "#858585", size: 10, strokeWidth: 0 },
    }));
  if (state.network) { state.network.destroy(); state.network = null; }
  $("graphCanvas").innerHTML = "";
  state.network = new vis.Network($("graphCanvas"),
    { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) },
    { physics: { solver: "forceAtlas2Based", stabilization: { iterations: 150 } },
      interaction: { hover: true, tooltipDelay: 150 } });
  $("graphMeta").textContent = `${nodes.length} nodes · ${edges.length} edges`
    + " — scroll to zoom, drag to pan, hover for details, double-click a node for connections";
  state.network.once("stabilizationIterationsDone", () => {
    if (!state.network) return;
    state.network.setOptions({ physics: false });  // stop the endless wobble
    state.network.fit();
    // Fit on a big graph zooms out until labels are unreadable; clamp it and
    // let the user pan instead.
    if (state.network.getScale() < 0.6) state.network.moveTo({ scale: 0.6 });
  });
  state.network.on("doubleClick", (params) => {
    if (!params.nodes.length) return;
    const node = data.nodes.find((n) => n.id === params.nodes[0]);
    if (node && (node.group === "Character" || node.group === "Location")) {
      $("entityName").value = node.label;
      $("lookupEntity").click();
    }
  });
}

$("refreshGraph").onclick = refreshGraph;
document.querySelectorAll(".gfilter").forEach((el) => { el.onchange = drawGraph; });
$("edgeLabels").onchange = drawGraph;

$("runChecks").onclick = async () => {
  if (!state.storyId) return;
  const { checks } = await api(`/stories/${state.storyId}/consistency`);
  $("checksBox").innerHTML = checks.map((c) => `
    <div class="check ${c.items.length ? c.severity : ""}">
      <b>${esc(c.check)}</b> — ${c.items.length ? c.items.length + " finding(s)" : "OK ✔"}
      ${c.items.length ? "<ul>" + c.items.map((i) => `<li>${esc(i)}</li>`).join("") + "</ul>" : ""}
    </div>`).join("");
};

$("lookupEntity").onclick = async () => {
  if (!state.storyId) return;
  const name = $("entityName").value.trim();
  if (!name) return;
  const data = await api(`/stories/${state.storyId}/entity?name=${encodeURIComponent(name)}`);
  if (!data.found) { $("entityBox").innerHTML = `<p class="muted">'${esc(name)}' not found in the graph.</p>`; return; }
  const n = data.node;
  $("entityBox").innerHTML = `
    <div class="check"><b>${esc(n.name || n.title || "")}</b>
      ${n.status ? `<span class="chip ${n.status === "dead" ? "dead" : ""}">${n.status}</span>` : ""}
      <p>${esc(n.description || "")}</p>
      ${(n.notes || []).slice(-5).map((x) => `<div class="muted">• ${esc(x)}</div>`).join("")}
      <ul>${data.links.map((l) => `<li>${l.dir === "out" ? "→" : "←"} <b>${esc(l.type)}</b>
        ${esc(l.other)} <span class="muted">(${l.label})</span></li>`).join("")}</ul>
    </div>`;
};

/* ------------------------------------------------------------------ misc */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function init() {
  try {
    const h = await api("/health");
    const ok = h.neo4j === "ok" && h.redis === "ok";
    $("healthBadge").textContent = ok ? "db ✔" : "db ✖";
    $("healthBadge").className = "badge " + (ok ? "ok" : "error");
    $("healthBadge").title = `neo4j: ${h.neo4j} | redis: ${h.redis}`;
  } catch (_) { $("healthBadge").textContent = "api ✖"; }
  await loadConfig().catch(() => {});
  await refreshStoryList().catch(() => {});
  // Deep link: #story=<id>&tab=<setup|story|write|graph>&ch=<n> (also as ?query)
  const params = new URLSearchParams(location.hash.slice(1) || location.search);
  const sid = params.get("story");
  if (sid) {
    state.storyId = sid;
    $("storySelect").value = sid;
    loadStory().catch(() => {});
    startPolling();
    const ch = parseInt(params.get("ch"), 10);
    if (ch) openChapter(ch);
  }
  const btn = params.get("tab") &&
    document.querySelector(`.tab[data-tab="${params.get("tab")}"]`);
  if (btn) btn.click();
}
init();
