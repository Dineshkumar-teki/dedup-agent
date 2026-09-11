"""Run offline (and optional live) evals and print an eval score."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent2.storage import ChromaRAGStore
from common.validation import validate_enriched_text
from evals.embeddings import IntentBagEmbeddings
from evals.scoring import (
    DUPLICATE_STATUSES,
    CaseResult,
    SuiteResult,
    classification_metrics,
    score_enrichment_text,
    weighted_overall,
)

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
MIN_OVERALL_SCORE = 85.0


def load_dataset() -> dict[str, Any]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _fresh_store(tmp_dir: Path, subject: str, texts: list[str]) -> ChromaRAGStore:
    os.environ.setdefault("OPENROUTER_API_KEY", "eval-offline-key")
    from common.config import get_settings

    get_settings.cache_clear()
    store = ChromaRAGStore.for_subject(subject, persist_dir=str(tmp_dir / subject))
    store.duplicate_distance_threshold = 0.25
    store.embedding = IntentBagEmbeddings(texts)  # type: ignore[assignment]
    return store


def eval_enrichment(dataset: dict[str, Any]) -> SuiteResult:
    results = [
        score_enrichment_text(case, str(case["gold_enriched"]))
        for case in dataset["enrichment_cases"]
    ]
    score = round(sum(item.score for item in results) / len(results), 1)
    return SuiteResult(name="enrichment", score=score, results=results)


def eval_validation(dataset: dict[str, Any]) -> SuiteResult:
    results: list[CaseResult] = []
    for case in dataset["validation_cases"]:
        violations = validate_enriched_text(case["qid"], case["enriched"], case["subject"])
        flagged = bool(violations)
        expected = bool(case["should_flag"])
        passed = flagged is expected
        results.append(
            CaseResult(
                case_id=case["id"],
                passed=passed,
                score=100.0 if passed else 0.0,
                expected="flag" if expected else "allow",
                predicted="flag" if flagged else "allow",
                detail=", ".join(violations) if violations else "clean",
            )
        )
    score = round(sum(item.score for item in results) / len(results), 1)
    return SuiteResult(name="validation", score=score, results=results)


def eval_dedup_text(dataset: dict[str, Any]) -> SuiteResult:
    pairs = dataset["dedup_pairs"]
    all_texts = [pair["anchor"]["enriched"] for pair in pairs] + [
        pair["candidate"]["enriched"] for pair in pairs
    ]
    results: list[CaseResult] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for pair in pairs:
            subject = pair["subject"]
            store = _fresh_store(tmp_path / pair["id"], subject, all_texts)
            anchor = pair["anchor"]
            candidate = pair["candidate"]
            store.add_or_flag_duplicate(anchor["qid"], anchor["raw"], anchor["enriched"])
            check = store.check_duplicate(candidate["qid"], candidate["enriched"])
            predicted = "duplicate" if check.status in DUPLICATE_STATUSES else "unique"
            expected = pair["label"]
            passed = predicted == expected
            results.append(
                CaseResult(
                    case_id=pair["id"],
                    passed=passed,
                    score=100.0 if passed else 0.0,
                    expected=expected,
                    predicted=predicted,
                    detail=f"status={check.status} similarity={check.similarity}",
                )
            )

    metrics = classification_metrics(results)
    return SuiteResult(
        name="dedup_text",
        score=round(100.0 * metrics["f1"], 1),
        results=results,
        extra=metrics,
    )


def eval_live_enrichment(dataset: dict[str, Any]) -> SuiteResult:
    from agent1.agent import enrich_question

    results: list[CaseResult] = []
    for case in dataset["enrichment_cases"]:
        enriched = enrich_question(case["qid"], case["raw_question"], case["subject"])
        item = score_enrichment_text(case, enriched.enriched_text)
        item.detail = f"{item.detail} | model: {enriched.enriched_text}"
        results.append(item)
    score = round(sum(item.score for item in results) / len(results), 1) if results else 0.0
    return SuiteResult(name="live_enrichment", score=score, results=results)


def _has_live_key() -> bool:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return bool(key) and key not in {"test-key", "eval-offline-key", "your-openrouter-api-key-here"}


def format_scorecard(suites: list[SuiteResult], overall: float) -> str:
    lines = ["SQL Dedup Agent — Eval Scorecard", "=" * 40]
    for suite in suites:
        lines.append(f"{suite.name:18} {suite.score:6.1f} / 100")
        if suite.extra:
            lines.append(
                "  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}  Acc={accuracy:.2f}".format(
                    **suite.extra
                )
            )
        failed = [item for item in suite.results if not item.passed]
        for item in failed:
            lines.append(
                f"  FAIL {item.case_id}: expected {item.expected}, got {item.predicted} ({item.detail})"
            )
    lines.append("-" * 40)
    lines.append(f"{'OVERALL':18} {overall:6.1f} / 100")
    return "\n".join(lines)


def run_evals(*, live: bool = False) -> dict[str, Any]:
    dataset = load_dataset()
    suites = [
        eval_enrichment(dataset),
        eval_validation(dataset),
        eval_dedup_text(dataset),
    ]
    if live:
        if not _has_live_key():
            raise RuntimeError("Live evals require a real OPENROUTER_API_KEY.")
        suites.append(eval_live_enrichment(dataset))

    scores = {suite.name: suite.score for suite in suites}
    overall = weighted_overall(
        {
            "enrichment": scores["enrichment"],
            "validation": scores["validation"],
            "dedup_text": scores["dedup_text"],
        }
    )
    return {
        "overall": round(overall, 1),
        "suites": scores,
        "scorecard": format_scorecard(suites, overall),
        "details": {
            suite.name: [
                {
                    "id": item.case_id,
                    "passed": item.passed,
                    "score": item.score,
                    "expected": item.expected,
                    "predicted": item.predicted,
                    "detail": item.detail,
                }
                for item in suite.results
            ]
            for suite in suites
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQL Dedup Agent evals and print a score.")
    parser.add_argument("--live", action="store_true", help="Also score live LLM enrichment.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a scorecard.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_OVERALL_SCORE,
        help="Exit nonzero if overall score is below this threshold.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_evals(live=args.live)
    if args.json:
        printable = {key: value for key, value in report.items() if key != "scorecard"}
        print(json.dumps(printable, indent=2))
    else:
        print(report["scorecard"])
    if report["overall"] < args.min_score:
        print(f"\nOverall score {report['overall']} is below minimum {args.min_score}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
