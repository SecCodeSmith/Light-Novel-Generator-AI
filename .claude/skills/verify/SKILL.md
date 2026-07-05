---
name: verify
description: Build, run and end-to-end verify the Light Novel Generator (Docker on WSL)
---

# Verify: Light Novel Generator AI

## Build & launch (Docker lives inside WSL, not on Windows)

```powershell
wsl --cd /mnt/c/Users/Kuba/Desktop/Light_novels_Generator_AI -- docker compose up -d --build
```

Gotchas learned the hard way:
- **WSL idle-termination kills the containers.** Windows terminates the idle WSL VM
  shortly after the last `wsl` command exits, taking dockerd down with it. Keep a
  holder alive while testing: `wsl -- sh -c "sleep 1800"` in the background.
  Services use `restart: unless-stopped`, so they come back when the VM reboots.
- **After editing backend code**, `docker compose up -d --build app` sometimes builds
  the new image but keeps the old container. Check with
  `docker inspect lng_app --format '{{.Image}}'` vs `docker images`; use
  `docker compose up -d --force-recreate app` if they differ.
- Host port 6379 is taken by another project (`zabd-redis`); lng redis intentionally
  has no host port mapping.
- **Never `--force-recreate app` while a story job is running.** Outline/write jobs
  are in-process background tasks; a restart kills them and leaves a stale Redis
  lock (`lng:story:<id>:lock`, 1h TTL) so the story sticks at "busy". Check
  `redis-cli --scan --pattern 'lng:story:*:lock'` (inside lng_redis) first; after an
  accidental kill, DEL the lock and `POST /api/stories/<id>/outline` — the outline
  loop resumes from the first unplanned chapter.
- **The smoke test swaps the LLM config to mock and restores it at the end** — do
  not run it while the user has a generation in flight.

## Drive it

Full pipeline smoke test (uses the offline mock LLM, no API key needed):

```powershell
wsl --cd /mnt/c/Users/Kuba/Desktop/Light_novels_Generator_AI -- python3 scripts/smoke_test.py
```

Covers: health, mock model list, story creation, outline (3 chapters), writing all
chapters (writer→critic→extractor), graph population, consistency checks, entity
lookup, markdown export, UI serving, deletion. Expect `ALL CHECKS PASSED ✔`.

UI evidence: headless Edge screenshots. Hard-won specifics:

- The UI supports deep links for this: `http://localhost:8000/?story=<id>&tab=<setup|story|write|graph>&ch=<n>`
  (query form, not `#hash` — Windows CLI launches drop URL fragments).
- Use `--headless=new` plus a throwaway `--user-data-dir` (the default profile
  drags in sync/Tracking Prevention noise; TP also blocks the Monaco CDN, which
  exercises the textarea fallback).
- `--virtual-time-budget=30000` for screenshots; plain `--timeout` often fires
  before fetches finish, and `--dump-dom` captures at load — too early for any
  fetch-driven DOM. Full recipe:
  `msedge --headless=new --disable-gpu --user-data-dir=<tmp> --no-first-run
   --window-size=1400,1000 --virtual-time-budget=30000 --screenshot=<png>
   "http://localhost:8000/?story=<id>&tab=graph"`
- Frontend regressions can be invisible to the API smoke test: a single
  `$("missingId").onclick` crashes ALL of app.js (this happened — the research
  panel HTML was removed while its JS wiring stayed). The smoke test now checks
  every `$("id")` in app.js exists in index.html; keep that check passing.

Worth probing: second `POST /write` while busy → 409; bad base_url then
`GET /api/models` → 502 with clean message; description <10 chars → 422.
