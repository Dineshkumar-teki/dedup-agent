"""Tests for Chroma deduplication logic with mocked embeddings."""

from __future__ import annotations

import pytest

from agent2.storage import ChromaRAGStore


class _FakeEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]


@pytest.fixture
def store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> ChromaRAGStore:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from common.config import get_settings

    get_settings.cache_clear()

    rag = ChromaRAGStore.for_subject("sql", persist_dir=str(tmp_path / "chroma"))
    rag.duplicate_distance_threshold = 0.25

    vectors = {
        "intent a": [1.0, 0.0, 0.0],
        "intent b": [0.0, 1.0, 0.0],
        "intent a similar": [0.99, 0.01, 0.0],
        "intent borderline": [0.74, 0.6726, 0.0],
        "intent far": [0.0, 0.0, 1.0],
    }
    rag.embedding = _FakeEmbeddings(vectors)  # type: ignore[assignment]
    return rag


def test_store_new_question(store: ChromaRAGStore) -> None:
    result = store.add_or_flag_duplicate("Q1", "raw", "intent a")
    assert result.status == "stored"
    assert store.count_questions() == 1


def test_detect_duplicate(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.add_or_flag_duplicate("Q2", "raw two", "intent a similar")
    assert result.status == "duplicate"
    assert result.original_qid == "Q1"
    assert store.count_questions() == 1


def test_detect_exact_duplicate(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.add_or_flag_duplicate("Q2", "raw two", "intent a")
    assert result.status == "duplicate"
    assert result.message == "Exact duplicate found."


def test_already_stored_same_qid_and_text(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw", "intent a")
    result = store.add_or_flag_duplicate("Q1", "raw", "intent a")
    assert result.status == "already_stored"


def test_allow_duplicate_stores_similar_question(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.add_or_flag_duplicate(
        "Q2",
        "raw two",
        "intent a similar",
        allow_duplicate=True,
    )
    assert result.status == "stored"
    assert result.message == "Duplicate question stored by user request."
    assert store.count_questions() == 2


def test_check_duplicate_returns_new(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.check_duplicate("Q2", "intent b")
    assert result.status == "new"
    assert store.count_questions() == 1


def test_check_duplicate_finds_similar(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.check_duplicate("Q2", "intent a similar")
    assert result.status == "duplicate"
    assert result.original_qid == "Q1"


def test_check_already_exists(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.check_duplicate("Q1", "intent a")
    assert result.status == "already_exists"


def test_check_qid_conflict(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.check_duplicate("Q1", "intent b")
    assert result.status == "qid_conflict"


def test_qid_conflict(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.add_or_flag_duplicate("Q1", "raw changed", "intent b")
    assert result.status == "qid_conflict"


def test_borderline_stored(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    result = store.add_or_flag_duplicate("Q2", "raw two", "intent borderline")
    assert result.status == "borderline_stored"
    assert store.count_questions() == 2


def test_batch_store(store: ChromaRAGStore) -> None:
    rows = [
        ("Q1", "raw one", "intent a"),
        ("Q2", "raw two", "intent b"),
    ]
    results = store.batch_store(rows)
    assert len(results) == 2
    assert all(r.status == "stored" for r in results)


def test_batch_store_empty(store: ChromaRAGStore) -> None:
    assert store.batch_store([]) == []


def test_batch_store_flags_duplicates_and_conflicts(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    results = store.batch_store(
        [
            ("Q1", "raw one", "intent a"),
            ("Q1", "raw changed", "intent b"),
            ("Q2", "raw two", "intent a similar"),
            ("Q3", "raw three", "intent far"),
        ]
    )
    assert [r.status for r in results] == [
        "already_stored",
        "qid_conflict",
        "duplicate",
        "stored",
    ]


def test_check_duplicates_batch_does_not_write(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw one", "intent a")
    results = store.check_duplicates_batch(
        [
            ("Q1", "raw one", "intent a"),
            ("Q2", "raw two", "intent a similar"),
            ("Q3", "raw three", "intent far"),
            ("Q1", "raw changed", "intent b"),
        ]
    )
    assert [r.status for r in results] == [
        "already_exists",
        "duplicate",
        "new",
        "qid_conflict",
    ]
    assert store.count_questions() == 1


def test_check_duplicates_batch_empty(store: ChromaRAGStore) -> None:
    assert store.check_duplicates_batch([]) == []


def test_for_subject_slugifies_collection_name(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from common.config import get_settings

    get_settings.cache_clear()

    rag = ChromaRAGStore.for_subject("React Native", persist_dir=str(tmp_path / "chroma"))
    assert rag.collection.name == "react_native"


@pytest.mark.parametrize("subject", ["", "!!!"])
def test_for_subject_rejects_invalid_name(subject: str) -> None:
    with pytest.raises(ValueError):
        ChromaRAGStore.for_subject(subject)


def test_list_subjects_includes_created_collection(store: ChromaRAGStore) -> None:
    store.add_or_flag_duplicate("Q1", "raw", "intent a")
    assert "sql" in store.list_subjects()
