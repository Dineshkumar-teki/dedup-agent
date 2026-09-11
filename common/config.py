"""Centralized application configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)


def _lookup(name: str, default: str | None = None) -> str | None:
    """Read a config value from env vars, then Streamlit secrets."""
    env_value = os.environ.get(name)
    if env_value is not None:
        return env_value

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _lookup(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _lookup(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _lookup(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    api_provider: str
    chroma_persist_dir: Path
    llm_model: str
    embedding_model: str
    llm_temperature: float
    duplicate_distance_threshold: float
    enrich_concurrency: int
    max_upload_rows: int
    max_upload_bytes: int
    api_retry_attempts: int
    api_retry_min_wait: float
    api_retry_max_wait: float
    app_username: str | None
    app_password: str | None
    log_level: str

    @property
    def auth_enabled(self) -> bool:
        return bool(self.app_username and self.app_password)

    def require_api_key(self) -> None:
        if not self.openrouter_api_key:
            raise RuntimeError(
                "API key is not set. Add OPENAI_API_KEY or OPENROUTER_API_KEY to .env, "
                "Streamlit secrets, or export it before running."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)
    openrouter_key = _lookup("OPENROUTER_API_KEY", "") or ""
    openai_key = _lookup("OPENAI_API_KEY", "") or ""
    provider = (_lookup("API_PROVIDER") or "").strip().lower()

    if provider == "openai":
        api_key = openai_key or openrouter_key
    elif provider == "openrouter":
        api_key = openrouter_key or openai_key
    elif openai_key:
        api_key = openai_key
        provider = "openai"
    elif openrouter_key:
        api_key = openrouter_key
        provider = "openrouter"
    else:
        api_key = ""
        provider = "openrouter"

    persist_dir = _lookup("CHROMA_PERSIST_DIR", "chroma_db")
    path = Path(persist_dir or "chroma_db")
    if not path.is_absolute():
        path = ROOT_DIR / path

    default_llm_model = "gpt-4o-mini" if provider == "openai" else "google/gemini-3-flash-preview"
    default_embedding_model = "text-embedding-3-large" if provider == "openai" else "openai/text-embedding-3-large"

    return Settings(
        openrouter_api_key=api_key,
        api_provider=provider,
        chroma_persist_dir=path,
        llm_model=_lookup("LLM_MODEL", default_llm_model) or default_llm_model,
        embedding_model=_lookup("EMBEDDING_MODEL", default_embedding_model) or default_embedding_model,
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        duplicate_distance_threshold=_env_float("DUPLICATE_DISTANCE_THRESHOLD", 0.25),
        enrich_concurrency=_env_int("ENRICH_CONCURRENCY", 10),
        max_upload_rows=_env_int("MAX_UPLOAD_ROWS", 500),
        max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024),
        api_retry_attempts=_env_int("API_RETRY_ATTEMPTS", 3),
        api_retry_min_wait=_env_float("API_RETRY_MIN_WAIT", 1.0),
        api_retry_max_wait=_env_float("API_RETRY_MAX_WAIT", 30.0),
        app_username=_lookup("APP_USERNAME"),
        app_password=_lookup("APP_PASSWORD"),
        log_level=_lookup("LOG_LEVEL", "INFO") or "INFO",
    )
