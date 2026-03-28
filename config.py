import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


class Settings:
    # ── Server ──────────────────────────────────────────────────
    HOST: str            = os.getenv("HOST", "0.0.0.0")
    PORT: int            = int(os.getenv("PORT", 8000))
    SESSION_SECRET: str  = os.getenv("SESSION_SECRET", "CHANGE_THIS_SECRET_KEY_IN_ENV")

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

# Ensure directories exist
Path(settings.SESSION_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.ATLAS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.TRIBE_MODEL_PATH).mkdir(parents=True, exist_ok=True)
