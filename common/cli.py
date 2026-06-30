import argparse
from typing import Iterable

from agent1.agent import enrich_question, process_file as process_enricher_file
from agent2.rag import process_file as process_rag_file
from agent2.storage import ChromaRAGStore


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich SQL questions from a CSV or Excel file containing qid and question columns."
    )
    parser.add_argument(
        "--input-file",
        "-i",
        help="Path to a CSV or Excel file containing qid and question columns.",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        help="Optional path to write enriched results as CSV.",
    )
    parser.add_argument(
        "--agent",
        choices=["enricher", "rag"],
        default="enricher",
        help="Choose which agent to run: enricher or rag.",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Store enriched results in a local Chroma DB instance (rag agent only).",
    )
    parser.add_argument(
        "--persist-dir",
        help="Optional directory where the Chroma DB will persist.",
        default="chroma_db",
    )
    parser.add_argument(
        "--qid",
        help="Single qid to enrich when not using an input file.",
    )
    parser.add_argument(
        "--question",
        help="Single raw SQL question text to enrich when not using an input file.",
    )
    parser.add_argument(
        "--subject",
        default="sql",
        help="Subject domain for enrichment: sql, python, or dsa.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.agent == "rag" and args.store:
        rag_store = ChromaRAGStore.for_subject(args.subject, persist_dir=args.persist_dir)
    else:
        rag_store = None

    if args.input_file:
        if args.agent == "enricher":
            process_enricher_file(args.input_file, args.output_file, subject=args.subject)
            return

        if args.agent == "rag":
            if rag_store is None:
                raise ValueError("RAG agent requires --store to persist results in Chroma DB.")

            store_results = process_rag_file(
                args.input_file,
                rag_store,
                args.output_file,
                subject=args.subject,
            )
            for result in store_results:
                print(
                    f"qid={result.qid}, status={result.status}, original_qid={result.original_qid}, distance={result.distance}, message={result.message}"
                )
            return

    if args.qid and args.question:
        result = enrich_question(args.qid, args.question, subject=args.subject)
        print("qid:", result.qid)
        print("enriched_text:", result.enriched_text)
        return

    parser = argparse.ArgumentParser(
        description="Enrich SQL questions from a CSV or Excel file containing qid and question columns."
    )
    parser.error("Provide either --input-file <path> or both --qid and --question.")
