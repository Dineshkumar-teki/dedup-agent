"""Validation helpers for enriched question text."""

from __future__ import annotations

SQL_SYNTAX_TOKENS = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY"]
CODE_SYNTAX_TOKENS = ["def ", "const ", "let ", "var ", "function ", "class ", "=>"]


def validate_enriched_text(qid: str, enriched_text: str, subject: str) -> list[str]:
    violations: list[str] = []

    if "[LOW_CONFIDENCE]" in enriched_text:
        violations.append("LOW_CONFIDENCE flag present")

    if qid in enriched_text:
        violations.append("qid leaked into enriched_text")

    word_count = len(enriched_text.split())
    if word_count > 80:
        violations.append(f"enriched_text too long ({word_count} words)")

    tokens = SQL_SYNTAX_TOKENS if subject == "sql" else CODE_SYNTAX_TOKENS
    if any(token in enriched_text for token in tokens):
        violations.append("code syntax detected in enriched_text")

    return violations
