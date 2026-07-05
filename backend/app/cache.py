"""Redis helpers: config storage, progress logs, context cache, write locks."""
import json
import time

import redis

from . import config

_pool = redis.ConnectionPool.from_url(config.REDIS_URL, decode_responses=True)


def r() -> redis.Redis:
    return redis.Redis(connection_pool=_pool)


# ---------------------------------------------------------------- LLM config

CONFIG_KEY = "lng:config"

DEFAULT_CONFIG = {
    "base_url": config.DEFAULT_LLM_BASE_URL,
    "api_key": config.DEFAULT_LLM_API_KEY,
    "writer_model": "",
    "critic_model": "",
    "temperature": "0.8",
    "agent_mode": "true" if config.AGENT_MODE_DEFAULT else "false",
    "chapter_target_words": str(config.CHAPTER_TARGET_WORDS),
}


def get_config() -> dict:
    stored = r().hgetall(CONFIG_KEY)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(stored)
    return cfg


def set_config(values: dict) -> dict:
    clean = {k: str(v) for k, v in values.items() if k in DEFAULT_CONFIG and v is not None}
    if clean:
        r().hset(CONFIG_KEY, mapping=clean)
    return get_config()


def is_agent_mode(cfg: dict | None = None) -> bool:
    """Return True if agent mode is currently enabled."""
    if cfg is None:
        cfg = get_config()
    return cfg.get("agent_mode", "true").lower() in ("true", "1", "yes")


# ------------------------------------------------------------- progress log

def log(story_id: str, message: str) -> None:
    entry = json.dumps({"t": round(time.time(), 1), "msg": message})
    key = f"lng:story:{story_id}:log"
    pipe = r().pipeline()
    pipe.rpush(key, entry)
    pipe.ltrim(key, -400, -1)
    pipe.execute()


def get_log(story_id: str, offset: int = 0) -> list[dict]:
    raw = r().lrange(f"lng:story:{story_id}:log", offset, -1)
    return [json.loads(x) for x in raw]


def set_status(story_id: str, status: str) -> None:
    r().set(f"lng:story:{story_id}:status", status)


def get_status(story_id: str) -> str:
    return r().get(f"lng:story:{story_id}:status") or "idle"


# ------------------------------------------------------- raw LLM traffic log

def record_llm(story_id: str, role: str, prompt: str, reply: str) -> None:
    """Keep the last few raw prompt/reply pairs so users can follow the model."""
    entry = json.dumps({"t": round(time.time(), 1), "role": role,
                        "prompt": prompt[-6000:], "reply": reply[:20000]})
    key = f"lng:story:{story_id}:llm"
    pipe = r().pipeline()
    pipe.rpush(key, entry)
    pipe.ltrim(key, -12, -1)
    pipe.expire(key, 7 * 24 * 3600)
    pipe.execute()


def get_llm_log(story_id: str) -> list[dict]:
    return [json.loads(x) for x in r().lrange(f"lng:story:{story_id}:llm", 0, -1)]


# ------------------------------------------------------------ context cache

def graph_version(story_id: str) -> str:
    return r().get(f"lng:story:{story_id}:gv") or "0"


def bump_graph_version(story_id: str) -> None:
    r().incr(f"lng:story:{story_id}:gv")


def cached_json(key: str, ttl: int, builder):
    """Return cached JSON value or build, store and return it."""
    hit = r().get(key)
    if hit is not None:
        return json.loads(hit)
    value = builder()
    r().set(key, json.dumps(value), ex=ttl)
    return value


def put_json(key: str, value, ttl: int) -> None:
    r().set(key, json.dumps(value), ex=ttl)


def get_json(key: str):
    hit = r().get(key)
    return json.loads(hit) if hit is not None else None


# ------------------------------------------------------------------- locks

def acquire_lock(story_id: str, ttl: int = 3600) -> bool:
    return bool(r().set(f"lng:story:{story_id}:lock", "1", nx=True, ex=ttl))


def release_lock(story_id: str) -> None:
    r().delete(f"lng:story:{story_id}:lock")


def is_locked(story_id: str) -> bool:
    return r().exists(f"lng:story:{story_id}:lock") == 1
