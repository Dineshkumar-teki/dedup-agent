"""Shared parallel enrichment utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent1.agent import enrich_question
from common.config import get_settings

logger = logging.getLogger(__name__)


def enrich_rows_parallel(
    rows: list[tuple[str, str]],
    subject: str,
    max_workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[tuple[str, str, str]], list[dict[str, str]]]:
    """Enrich question rows in parallel."""
    if not rows:
        return [], []

    workers = max_workers or get_settings().enrich_concurrency
    total = len(rows)
    indexed_results: list[tuple[int, str, str, str | None, str | None]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_enrich_single, idx, qid, raw_question, subject): idx
            for idx, (qid, raw_question) in enumerate(rows)
        }
        for future in as_completed(futures):
            indexed_results.append(future.result())
            if on_progress is not None:
                on_progress(len(indexed_results), total)

    indexed_results.sort(key=lambda item: item[0])

    clean_rows: list[tuple[str, str, str]] = []
    errors: list[dict[str, str]] = []
    for _, qid, raw_question, enriched_text, error in indexed_results:
        if error is not None:
            logger.warning("Enrichment failed for qid=%s: %s", qid, error)
            errors.append(
                {
                    "qid": qid,
                    "raw_question": raw_question,
                    "enriched_text": "",
                    "violations": f"Enrichment failed: {error}",
                }
            )
        elif enriched_text is not None:
            clean_rows.append((qid, raw_question, enriched_text))

    return clean_rows, errors


def _enrich_single(
    index: int,
    qid: str,
    raw_question: str,
    subject: str,
) -> tuple[int, str, str, str | None, str | None]:
    try:
        enriched = enrich_question(qid, raw_question, subject)
        return index, qid, raw_question, enriched.enriched_text, None
    except Exception as exc:
        return index, qid, raw_question, None, str(exc)
