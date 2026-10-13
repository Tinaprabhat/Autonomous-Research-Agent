"""
config.py — single place where environment configuration is read.

Everything is env-driven so local dev (embedded/Docker Qdrant) and prod
(Qdrant Cloud) use identical code with different values.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _env(*names: str, default: str | None = None) -> str | None:
    """First non-empty env var among `names` (supports legacy spellings)."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY       = _env("GROQ_API_KEY")
GROQ_MODEL         = _env("GROQ_MODEL", default="openai/gpt-oss-120b")
GROQ_REWRITE_MODEL = _env("GROQ_REWRITE_MODEL", default="openai/gpt-oss-20b")
# Extra Groq models tried (in order) when the primary hits a rate limit. Each free-tier model has its
# OWN daily token budget (200k/day), so a chain multiplies the daily headroom.
GROQ_FALLBACK_MODELS = [m.strip() for m in (_env("GROQ_FALLBACK_MODELS",
                        default="openai/gpt-oss-20b,qwen/qwen3.8-27b") or "").split(",") if m.strip()]
OLLAMA_MODEL       = _env("OLLAMA_MODEL", default="qwen2.5:1.5b")

# ── Qdrant ───────────────────────────────────────────────────────────────────
# URL set  -> remote (Docker or Qdrant Cloud). Unset -> embedded local storage.
# "Quadrant_*" are the spellings already present in the project's .env.
QDRANT_URL        = _env("QDRANT_URL", "Quadrant_EndPoint", "Quadrant_Endpoint")
QDRANT_API_KEY    = _env("QDRANT_API_KEY", "Quadrant_API_KEY")
QDRANT_PATH       = _env("QDRANT_PATH", default="data/qdrant_local")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", default="papers")
# Force embedded local mode even if a URL is configured (safe dev switch).
QDRANT_FORCE_LOCAL = (_env("QDRANT_FORCE_LOCAL", default="0") or "0") == "1"

EMBED_DIM = 384  # all-MiniLM-L6-v2

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR      = _env("RAW_DIR", default="data/raw_papers")
JOBS_DB      = _env("JOBS_DB", default="data/jobs.db")
FEEDBACK_LOG = _env("FEEDBACK_LOG", default="data/feedback.jsonl")
QUERY_LOG    = _env("QUERY_LOG", default="data/query_log.jsonl")

# ── Abuse limits ─────────────────────────────────────────────────────────────
MAX_QUERY_CHARS    = int(_env("MAX_QUERY_CHARS", default="500"))
RATE_LIMIT_PER_MIN = int(_env("RATE_LIMIT_PER_MIN", default="20"))

# -- Auth + per-user limits (backend/app/auth.py) ----------------------------
# API_KEYS="name:key[:per_minute[:per_day]],..." - empty means auth is OFF (dev/tests).
API_KEYS                = _env("API_KEYS", default="")
USER_RATE_LIMIT_PER_MIN = int(_env("USER_RATE_LIMIT_PER_MIN", default="20"))
USER_DAILY_QUOTA        = int(_env("USER_DAILY_QUOTA", default="200"))
AUTH_FAIL_PER_MIN       = int(_env("AUTH_FAIL_PER_MIN", default="10"))   # bad-key attempts per IP per minute

# ── Verification / safety models ─────────────────────────────────────────────
NLI_MODEL       = _env("NLI_MODEL", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
JAILBREAK_MODEL = _env("JAILBREAK_MODEL", default="protectai/deberta-v3-base-prompt-injection-v2")
