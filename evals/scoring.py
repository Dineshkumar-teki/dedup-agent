"""Metric helpers and per-case scoring for evals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SQL_QUERY_TOKENS = ["SELECT", "FROM", "WHERE", "GROUP BY"]
CODE_BLOCK_TOKENS = ["def ", "const ", "let ", "var ", "=>", "```"]


def prompt_rule_violations(qid: str, enriched_text: str, subject: str) -> list[str]:
    """Check enrichment against prompt rules, allowing operation names used in prose."""
    violations: list[str] = []
    if qid and qid in enriched_text:
        violations.append("qid leaked into enriched_text")
    word_count = len(enriched_text.split())
    if word_count > 80:
        violations.append(f"enriched_text too long ({word_count} words)")
    if subject == "sql":
        tokens = SQL_QUERY_TOKENS
    else:
        tokens = CODE_BLOCK_TOKENS
    if any(token in enriched_text for token in tokens):
        violations.append("code syntax detected in enriched_text")
    return violations

DUPLICATE_STATUSES = {"duplicate", "already_exists"}


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    score: float
    expected: str | None = None
    predicted: str | None = None
    detail: str = ""


@dataclass
class SuiteResult:
    name: str
    score: float
    results: list[CaseResult] = field(default_factory=list)
    extra: dict[str, float] = field(default_factory=dict)


def contains_phrase(text: str, phrase: str) -> bool:
    """Match SQL tokens / identifiers case-sensitively; other phrases ignore case."""
    if phrase.isupper() or any(char.isupper() for char in phrase):
        return phrase in text
    return phrase.lower() in text.lower()


def score_enrichment_text(case: dict[str, Any], enriched_text: str) -> CaseResult:
    qid = str(case["qid"])
    subject = str(case["subject"])
    expect_low = bool(case.get("expect_low_confidence", False))
    violations = prompt_rule_violations(qid, enriched_text, subject)

    if expect_low:
        rule_ok = "[LOW_CONFIDENCE]" in enriched_text
        rule_note = "low-confidence flag present" if rule_ok else "missing [LOW_CONFIDENCE]"
    else:
        rule_ok = not violations
        rule_note = "rules ok" if rule_ok else ", ".join(violations)

    must_include = list(case.get("must_include", []))
    include_hits = sum(1 for phrase in must_include if contains_phrase(enriched_text, phrase))
    include_score = include_hits / len(must_include) if must_include else 1.0

    must_not = list(case.get("must_not_include", []))
    forbidden_hits = [phrase for phrase in must_not if contains_phrase(enriched_text, phrase)]
    forbid_ok = not forbidden_hits

    starts = list(case.get("must_start_with", []))
    start_ok = True
    if starts:
        start_ok = any(enriched_text.startswith(prefix) for prefix in starts)

    score = round(
        100.0
        * (
            0.40 * (1.0 if rule_ok else 0.0)
            + 0.30 * include_score
            + 0.20 * (1.0 if forbid_ok else 0.0)
            + 0.10 * (1.0 if start_ok else 0.0)
        ),
        1,
    )
    passed = score >= 80.0
    detail_parts = [rule_note]
    if must_include:
        detail_parts.append(f"keywords {include_hits}/{len(must_include)}")
    if forbidden_hits:
        detail_parts.append(f"forbidden: {', '.join(forbidden_hits)}")
    if starts and not start_ok:
        detail_parts.append("missing operation opener")
    return CaseResult(
        case_id=str(case["id"]),
        passed=passed,
        score=score,
        expected="valid enrichment",
        predicted=enriched_text,
        detail="; ".join(detail_parts),
    )


def classification_metrics(results: list[CaseResult]) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for item in results:
        expected_dup = item.expected == "duplicate"
        predicted_dup = item.predicted == "duplicate"
        if expected_dup and predicted_dup:
            tp += 1
        elif not expected_dup and predicted_dup:
            fp += 1
        elif not expected_dup and not predicted_dup:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(results) if results else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def weighted_overall(scores: dict[str, float]) -> float:
    """Combine suite scores into a single 0-100 eval score."""
    weights = {
        "enrichment": 0.25,
        "validation": 0.15,
        "dedup_text": 0.60,
    }
    total_weight = 0.0
    total = 0.0
    for name, weight in weights.items():
        if name in scores:
            total += scores[name] * weight
            total_weight += weight
    if total_weight == 0:
        return 0.0
    return round(max(0.0, min(100.0, total / total_weight)), 1)
