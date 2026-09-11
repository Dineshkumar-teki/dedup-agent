"""Regression tests for the offline eval scorecard."""

from __future__ import annotations

from evals.runner import MIN_OVERALL_SCORE, run_evals
from evals.scoring import score_enrichment_text, weighted_overall


def test_offline_eval_meets_score_floor() -> None:
    report = run_evals(live=False)
    assert report["suites"]["validation"] == 100.0
    assert report["suites"]["enrichment"] >= 90.0
    assert report["overall"] >= MIN_OVERALL_SCORE


def test_enrichment_scorer_penalizes_qid_leak() -> None:
    case = {
        "id": "leak",
        "qid": "Q1",
        "subject": "sql",
        "must_include": ["LEFT JOIN"],
        "must_not_include": ["Q1"],
        "must_start_with": ["LEFT JOIN"],
        "expect_low_confidence": False,
    }
    result = score_enrichment_text(case, "LEFT JOIN of person Q1 and address.")
    assert result.passed is False
    assert result.score < 80


def test_overall_weights() -> None:
    assert weighted_overall({"enrichment": 100, "validation": 100, "dedup_text": 100}) == 100
    assert weighted_overall({"enrichment": 0, "validation": 0, "dedup_text": 100}) == 60
