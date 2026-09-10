"""Tests for the RAG file pipeline with mocked enrichment."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent2.rag import process_file
from agent2.storage import StoreResult


def test_rag_process_file_stores_clean_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text("qid,question\nQ1,What is a join?\n", encoding="utf-8")

    monkeypatch.setattr(
        "agent2.rag.enrich_rows_parallel",
        lambda rows, subject: (
            [("Q1", "What is a join?", "Combine rows from two tables using matching keys.")],
            [],
        ),
    )

    stored: list[tuple] = []

    class _FakeStore:
        def batch_store(self, rows):
            stored.extend(rows)
            return [StoreResult(rows[0][0], "stored", original_qid=rows[0][0])]

    results = process_file(str(csv_path), _FakeStore(), subject="sql")
    assert results[0].status == "stored"
    assert stored[0][0] == "Q1"


def test_rag_process_file_raises_on_enrichment_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text("qid,question\nQ1,What is a join?\n", encoding="utf-8")
    monkeypatch.setattr(
        "agent2.rag.enrich_rows_parallel",
        lambda rows, subject: ([], [{"qid": "Q1", "violations": "boom"}]),
    )

    with pytest.raises(ValueError, match="Enrichment failed"):
        process_file(str(csv_path), SimpleNamespace(), subject="sql")


def test_rag_process_file_raises_on_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text("qid,question\nQ1,What is a join?\n", encoding="utf-8")
    monkeypatch.setattr(
        "agent2.rag.enrich_rows_parallel",
        lambda rows, subject: ([("Q1", "What is a join?", "Q1 SELECT * FROM t")], []),
    )

    with pytest.raises(ValueError, match="Validation failed"):
        process_file(str(csv_path), SimpleNamespace(), subject="sql")


def test_rag_process_file_rejects_empty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("qid,question\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No valid rows"):
        process_file(str(csv_path), SimpleNamespace(), subject="sql")
