"""
Central configuration.

All settings come from environment variables (optionally loaded from
``backend/.env``). Nothing here is required for the core offline analytics:
every value has a safe default. Only the optional LLM features need a key.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

BACKEND_DIR = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(BACKEND_DIR / ".env", override=False)


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


PLACEHOLDER_KEYS = {"", "your_key_here", "sk-ant-your-key-here", "changeme"}


@dataclass
class Settings:
    app_name: str = "DataForge AI"
    version: str = "5.0.0"

    data_dir: Path = field(default_factory=lambda: Path(_env("DATAFORGE_DATA_DIR", str(BACKEND_DIR / "data"))))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- Upload / dataset limits -------------------------------------------------
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 100))
    max_rows: int = field(default_factory=lambda: _env_int("MAX_ROWS", 2_000_000))
    max_columns: int = field(default_factory=lambda: _env_int("MAX_COLUMNS", 1_000))
    max_cells: int = field(default_factory=lambda: _env_int("MAX_CELLS", 60_000_000))
    max_json_body_kb: int = field(default_factory=lambda: _env_int("MAX_JSON_BODY_KB", 2_048))

    # --- Retention ---------------------------------------------------------------
    workspace_ttl_days: float = field(default_factory=lambda: _env_float("WORKSPACE_TTL_DAYS", 14))
    max_cached_workspaces: int = field(default_factory=lambda: _env_int("MAX_CACHED_WORKSPACES", 8))
    workspace_listing: bool = field(default_factory=lambda: _env_bool("WORKSPACE_LISTING", True))

    # --- ML resource limits --------------------------------------------------------
    training_max_rows: int = field(default_factory=lambda: _env_int("TRAINING_MAX_ROWS", 200_000))
    training_time_budget_s: int = field(default_factory=lambda: _env_int("TRAINING_TIME_BUDGET_S", 120))
    max_concurrent_jobs: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_JOBS", 2))
    max_queued_jobs: int = field(default_factory=lambda: _env_int("MAX_QUEUED_JOBS", 8))
    ml_n_jobs: int = field(default_factory=lambda: _env_int("ML_N_JOBS", 2))
    batch_predict_max_rows: int = field(default_factory=lambda: _env_int("BATCH_PREDICT_MAX_ROWS", 500_000))

    # --- Analysis limits ---------------------------------------------------------------
    profile_sample_rows: int = field(default_factory=lambda: _env_int("PROFILE_SAMPLE_ROWS", 500_000))
    chart_max_points: int = field(default_factory=lambda: _env_int("CHART_MAX_POINTS", 2_000))

    # --- Security ------------------------------------------------------------------
    cors_origins: List[str] = field(default_factory=lambda: [
        o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()
    ])
    csv_formula_escape: bool = field(default_factory=lambda: _env_bool("CSV_FORMULA_ESCAPE", True))
    secret_key: str = field(default_factory=lambda: _env("DATAFORGE_SECRET_KEY", ""))

    # --- Optional LLM -----------------------------------------------------------------
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    llm_enabled_flag: bool = field(default_factory=lambda: _env_bool("LLM_ENABLED", True))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 30))
    anthropic_base_url: str = field(default_factory=lambda: _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))

    # --- Frontend static serving (single-process deployment) ---------------------------
    frontend_dist: Path = field(default_factory=lambda: Path(_env("FRONTEND_DIST", str(BACKEND_DIR.parent / "frontend" / "dist"))))

    @property
    def llm_enabled(self) -> bool:
        return self.llm_enabled_flag and self.anthropic_api_key not in PLACEHOLDER_KEYS

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        (self.data_dir / "workspaces").mkdir(parents=True, exist_ok=True)

    def get_secret(self) -> bytes:
        """Secret used to sign persisted model artifacts (HMAC-SHA256).

        If DATAFORGE_SECRET_KEY is not provided, a random key is generated once and
        stored in the data directory with owner-only permissions.
        """
        if self.secret_key:
            return self.secret_key.encode()
        self.ensure_dirs()
        path = self.data_dir / ".secret_key"
        if path.exists():
            return path.read_bytes().strip()
        key = secrets.token_hex(32).encode()
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
        return key

    def public_dict(self) -> dict:
        """Non-secret settings safe to expose to the frontend."""
        return {
            "version": self.version,
            "max_upload_mb": self.max_upload_mb,
            "max_rows": self.max_rows,
            "max_columns": self.max_columns,
            "workspace_ttl_days": self.workspace_ttl_days,
            "training_max_rows": self.training_max_rows,
            "training_time_budget_s": self.training_time_budget_s,
            "llm_enabled": self.llm_enabled,
            "llm_model": self.anthropic_model if self.llm_enabled else None,
            "workspace_listing": self.workspace_listing,
        }


settings = Settings()


def reload_settings() -> Settings:
    """Re-read the environment (used by tests)."""
    global settings
    new = Settings()
    settings.__dict__.update(new.__dict__)
    return settings
