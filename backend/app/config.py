import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lightnovel123")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Defaults pre-filled into the UI config the first time the app starts.
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")

OUTLINE_BATCH_SIZE = 5          # chapters planned per architect call (small: local models truncate)
TIMELINE_EVENT_LIMIT = 30       # past events fed to the writer
CHAPTER_TARGET_WORDS = 1800
CONTEXT_CACHE_TTL = 3600
MODELS_CACHE_TTL = 600

# Agent mode settings
AGENT_MODE_DEFAULT = True       # whether agent mode is enabled by default
AGENT_MAX_TOOL_ROUNDS = 8      # max tool-call rounds per agent invocation
