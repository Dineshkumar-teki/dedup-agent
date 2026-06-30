"""RAG agent for storing and deduplicating enriched questions."""

from common.io import load_questions_from_file, write_enriched_questions
from agent1.agent import enrich_question
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

    enriched_results = []
    store_results: list[StoreResult] = []
    for qid, raw_question in rows:
        enriched = enrich_question(qid, raw_question, subject)
        enriched_results.append(enriched)
        store_results.append(rag_store.add_or_flag_duplicate(qid, raw_question, enriched.enriched_text))

    if output_file:
        write_enriched_questions([(item.qid, item.enriched_text) for item in enriched_results], output_file)

    return store_results
