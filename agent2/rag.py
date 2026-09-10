"""RAG agent for storing and deduplicating enriched questions."""

from __future__ import annotations

from common.enrichment import enrich_rows_parallel
from common.io import load_questions_from_file, write_enriched_questions
from common.validation import validate_enriched_text

from .storage import ChromaRAGStore, StoreResult


def process_file(
    input_file: str,
    rag_store: ChromaRAGStore,
    output_file: str | None = None,
    subject: str = "sql",
) -> list[StoreResult]:
    rows = load_questions_from_file(input_file)
    if not rows:
        raise ValueError("No valid rows found in input file.")

    enriched_rows, error_rows = enrich_rows_parallel(rows, subject)
    if error_rows:
        raise ValueError(f"Enrichment failed for {len(error_rows)} row(s).")

    clean_rows = []
    for qid, raw_question, enriched_text in enriched_rows:
        violations = validate_enriched_text(qid, enriched_text, subject)
        if violations:
            raise ValueError(f"Validation failed for qid={qid}: {', '.join(violations)}")
        clean_rows.append((qid, raw_question, enriched_text))

    store_results = rag_store.batch_store(clean_rows)

    if output_file:
        write_enriched_questions(
            [(qid, enriched_text) for qid, _, enriched_text in clean_rows],
            output_file,
        )

    return store_results
