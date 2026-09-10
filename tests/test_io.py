"""Tests for CSV/Excel loading and upload limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.io import load_questions_from_csv, load_questions_from_file


@pytest.fixture(autouse=True)
def _set_test_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_ROWS", "3")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    from common.config import get_settings

    get_settings.cache_clear()


def test_load_questions_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "qid,question\nQ1,What is a join?\nQ2,Explain GROUP BY\n",
        encoding="utf-8",
    )

    rows = load_questions_from_csv(str(csv_path))
    assert rows == [("Q1", "What is a join?"), ("Q2", "Explain GROUP BY")]


def test_load_questions_rejects_too_many_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "large.csv"
    csv_path.write_text(
        "qid,question\n" + "\n".join(f"Q{i},Question {i}" for i in range(5)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exceeding the maximum"):
        load_questions_from_file(str(csv_path))


def test_load_questions_rejects_large_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "big.csv"
    csv_path.write_text("qid,question\nQ1,Hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="maximum upload size"):
        load_questions_from_file(str(csv_path), file_size=2048)


def test_load_questions_accepts_column_aliases(tmp_path: Path) -> None:
    csv_path = tmp_path / "aliased.csv"
    csv_path.write_text(
        "Question ID,raw_question\nQ1,What is a join?\n",
        encoding="utf-8",
    )

    rows = load_questions_from_csv(str(csv_path))
    assert rows == [("Q1", "What is a join?")]


def test_load_questions_skips_blank_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "sparse.csv"
    csv_path.write_text(
        "qid,question\nQ1,What is a join?\n,,\nQ2,\n,Only text\nQ3,Explain GROUP BY\n",
        encoding="utf-8",
    )

    rows = load_questions_from_csv(str(csv_path))
    assert rows == [("Q1", "What is a join?"), ("Q3", "Explain GROUP BY")]


def test_load_questions_requires_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="must include headers"):
        load_questions_from_csv(str(csv_path))


def test_load_questions_missing_qid_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "no_qid.csv"
    csv_path.write_text("question\nWhat is a join?\n", encoding="utf-8")

    with pytest.raises(ValueError, match="QID column not found"):
        load_questions_from_csv(str(csv_path))


def test_load_questions_missing_question_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "no_question.csv"
    csv_path.write_text("qid\nQ1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Question column not found"):
        load_questions_from_csv(str(csv_path))


def test_load_questions_rejects_unsupported_type(tmp_path: Path) -> None:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("qid,question\nQ1,Hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported input file type"):
        load_questions_from_file(str(txt_path))


def test_load_questions_from_excel(tmp_path: Path) -> None:
    from openpyxl import Workbook

    xlsx_path = tmp_path / "questions.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["id", "text"])
    sheet.append(["Q1", "What is a join?"])
    sheet.append([None, None])
    sheet.append(["Q2", "Explain GROUP BY"])
    workbook.save(xlsx_path)

    rows = load_questions_from_file(str(xlsx_path))
    assert rows == [("Q1", "What is a join?"), ("Q2", "Explain GROUP BY")]


def test_write_enriched_questions(tmp_path: Path) -> None:
    from common.io import write_enriched_questions

    output_path = tmp_path / "out.csv"
    write_enriched_questions([("Q1", "intent a"), ("Q2", "intent b")], str(output_path))
    assert output_path.read_text(encoding="utf-8") == "qid,enriched_text\nQ1,intent a\nQ2,intent b\n"
