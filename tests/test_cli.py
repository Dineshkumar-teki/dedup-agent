"""Tests for CLI argument parsing and usage errors."""

from __future__ import annotations

import pytest

from common.cli import parse_args


def test_parse_enricher_file_args() -> None:
    args = parse_args(["-i", "questions.csv", "--subject", "python"])
    assert args.input_file == "questions.csv"
    assert args.agent == "enricher"
    assert args.store is False
    assert args.subject == "python"


def test_parse_rag_store_args() -> None:
    args = parse_args(
        ["-i", "questions.xlsx", "--agent", "rag", "--store", "--persist-dir", "/tmp/chroma"]
    )
    assert args.agent == "rag"
    assert args.store is True
    assert args.persist_dir == "/tmp/chroma"


def test_parse_single_question_args() -> None:
    args = parse_args(["--qid", "Q1", "--question", "What is a JOIN?"])
    assert args.qid == "Q1"
    assert args.question == "What is a JOIN?"
    assert args.input_file is None


def test_main_requires_input_or_question(capsys: pytest.CaptureFixture[str]) -> None:
    from common.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Provide either --input-file" in captured.err


def test_main_rag_without_store_raises() -> None:
    from common.cli import main

    with pytest.raises(ValueError, match="requires --store"):
        main(["-i", "questions.csv", "--agent", "rag"])
