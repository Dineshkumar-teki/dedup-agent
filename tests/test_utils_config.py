"""Tests for column matching, config, and parallel enrichment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.utils import find_column, normalize


def test_normalize_strips_separators() -> None:
    assert normalize("Question_ID") == "questionid"
    assert normalize(" raw-question ") == "rawquestion"


def test_find_column_matches_aliases() -> None:
    headers = {"question id": "Question ID", "text": "text"}
    assert find_column(headers, ["qid", "question_id", "id"]) == "Question ID"
    assert find_column(headers, ["question", "raw_question", "text"]) == "text"
    assert find_column(headers, ["missing"]) is None


def test_settings_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("APP_USERNAME", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    from common.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.auth_enabled is False
    with pytest.raises(RuntimeError, match="API key is not set"):
        settings.require_api_key()


def test_settings_auth_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("APP_USERNAME", "admin")
    monkeypatch.setenv("APP_PASSWORD", "secret")
    from common.config import get_settings

    get_settings.cache_clear()
    assert get_settings().auth_enabled is True


def test_settings_detects_openai_key_and_defaults_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-key")
    monkeypatch.delenv("API_PROVIDER", raising=False)
    from common.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.api_provider == "openai"
    assert settings.openrouter_api_key == "sk-proj-test-key"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.embedding_model == "text-embedding-3-large"


def test_enrich_rows_parallel_collects_success_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from common import enrichment

    def fake_enrich(qid: str, raw_question: str, subject: str = "sql"):
        if qid == "Q2":
            raise RuntimeError("boom")
        return SimpleNamespace(qid=qid, enriched_text=f"intent {qid}")

    monkeypatch.setattr("common.enrichment.enrich_question", fake_enrich)
    progress: list[tuple[int, int]] = []

    clean, errors = enrichment.enrich_rows_parallel(
        [("Q1", "one"), ("Q2", "two"), ("Q3", "three")],
        "sql",
        max_workers=2,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert clean == [("Q1", "one", "intent Q1"), ("Q3", "three", "intent Q3")]
    assert len(errors) == 1
    assert errors[0]["qid"] == "Q2"
    assert "boom" in errors[0]["violations"]
    assert progress[-1] == (3, 3)


def test_enrich_rows_parallel_empty() -> None:
    from common.enrichment import enrich_rows_parallel

    assert enrich_rows_parallel([], "sql") == ([], [])
