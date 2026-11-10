import os
import hashlib
import secrets
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Known placeholder values that must never be used as a real signing key.
_PLACEHOLDER_SECRETS = {
    "",
    "CHANGE_THIS_SECRET_KEY_IN_ENV",
    "REPLACE_WITH_A_STRONG_RANDOM_SECRET",
}


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _resolve_session_secret() -> str:
    """Return the configured session secret, or a temporary random one.

    A predictable signing key lets anyone forge session cookies and bypass
    authentication, so the known placeholder values are rejected. When no
    secure value is set, a random per-process secret is generated and a
    warning is emitted; sessions then do not persist across restarts or
    multiple workers, which is acceptable for local use but must be fixed for
    production by setting SESSION_SECRET in the environment.
    """
    value = os.getenv("SESSION_SECRET", "")
    if value and value not in _PLACEHOLDER_SECRETS:
        return value
    warnings.warn(
        "SESSION_SECRET is not set to a secure value; generating a temporary "
        "random secret for this process. Set SESSION_SECRET in your .env "
        "(python3 -c \"import secrets; print(secrets.token_hex(32))\") for "
        "production so sessions remain valid across restarts.",
        RuntimeWarning,
        stacklevel=2,
    )
    return secrets.token_hex(32)


class Settings:
    # ── Server ──────────────────────────────────────────────────
    HOST: str            = os.getenv("HOST", "0.0.0.0")
    PORT: int            = int(os.getenv("PORT", 8000))
    SESSION_SECRET: str  = _resolve_session_secret()

    # ── Storage ──────────────────────────────────────────────────
    SESSION_DIR: str     = os.getenv("SESSION_DIR", str(BASE_DIR / "data" / "sessions"))
    ATLAS_DIR: str       = os.getenv("ATLAS_DIR",   str(BASE_DIR / "atlas_data"))

    # ── Model paths ──────────────────────────────────────────────
    TRIBE_MODEL_PATH: str = os.getenv(
        "TRIBE_MODEL_PATH",
        str(BASE_DIR / "models" / "tribe_v2")
    )

    # ── Ollama / Llama 3.2 ───────────────────────────────────────
    OLLAMA_HOST: str     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    LLM_MODEL: str       = os.getenv("LLM_MODEL",  "llama3.2")

    # ── Auth users  {username: sha256(password)} ─────────────────
    # Add more users by appending to SPIKD_USERS env var as JSON
    # e.g. SPIKD_USERS='{"admin":"password123","analyst":"pass456"}'
    @property
    def USERS(self) -> dict:
        import json
        raw = os.getenv("SPIKD_USERS", "")
        if raw:
            try:
                plain = json.loads(raw)
                return {u: _hash(p) for u, p in plain.items()}
            except Exception:
                pass
        # Defaults — override via env in production
        return {
            os.getenv("ADMIN_USER", "admin"): _hash(
                os.getenv("ADMIN_PASS", "spikd2024")
            )
        }

    # ── VRAM budget (MB) — for warnings ──────────────────────────
    VRAM_BUDGET_MB: int  = int(os.getenv("VRAM_BUDGET_MB", 20000))


settings = Settings()

# Warn once if the built-in default credentials are still in use.
if not os.getenv("SPIKD_USERS") and os.getenv("ADMIN_PASS", "spikd2024") == "spikd2024":
    warnings.warn(
        "Using the default admin credentials (admin / spikd2024). Set "
        "SPIKD_USERS or ADMIN_USER/ADMIN_PASS in your .env before exposing "
        "this service.",
        RuntimeWarning,
        stacklevel=2,
    )

# Ensure directories exist
Path(settings.SESSION_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.ATLAS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.TRIBE_MODEL_PATH).mkdir(parents=True, exist_ok=True)
