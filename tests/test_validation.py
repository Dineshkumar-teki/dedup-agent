"""Tests for enriched text validation."""

from common.validation import validate_enriched_text


def test_validate_flags_low_confidence() -> None:
    violations = validate_enriched_text("Q1", "This is [LOW_CONFIDENCE] output.", "python")
    assert "LOW_CONFIDENCE flag present" in violations


def test_validate_flags_qid_leak() -> None:
    violations = validate_enriched_text("Q1", "Question Q1 asks about joins.", "sql")
    assert "qid leaked into enriched_text" in violations


def test_validate_accepts_clean_text() -> None:
    violations = validate_enriched_text(
        "Q1",
        "Determine how to combine rows from two tables using matching keys.",
        "sql",
    )
    assert violations == []


def test_validate_flags_sql_syntax() -> None:
    violations = validate_enriched_text("Q1", "Write SELECT name FROM users.", "sql")
    assert "code syntax detected in enriched_text" in violations


def test_validate_flags_code_syntax_for_python() -> None:
    violations = validate_enriched_text("Q1", "Use def helper to compute totals.", "python")
    assert "code syntax detected in enriched_text" in violations


def test_validate_flags_overlong_text() -> None:
    text = " ".join(["word"] * 81)
    violations = validate_enriched_text("Q1", text, "dsa")
    assert any(item.startswith("enriched_text too long") for item in violations)


def test_validate_collects_multiple_violations() -> None:
    violations = validate_enriched_text(
        "Q9",
        "Q9 [LOW_CONFIDENCE] SELECT * FROM users",
        "sql",
    )
    assert "LOW_CONFIDENCE flag present" in violations
    assert "qid leaked into enriched_text" in violations
    assert "code syntax detected in enriched_text" in violations
