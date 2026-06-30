import os
import sys
from pathlib import Path

import chromadb

os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("GEMINI_API_KEY", "dummy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent2.storage import ChromaRAGStore


def main() -> None:
    client = chromadb.PersistentClient(path="chroma_db")

    old = client.get_collection("sql_enriched_questions")
    print(f"Old collection count before migration: {old.count()}")

    data = old.get(include=["metadatas", "documents", "embeddings"])

    new_store = ChromaRAGStore.for_subject("sql", persist_dir="chroma_db")
    new_collection = new_store.collection

    if data.get("ids"):
        new_collection.add(
            ids=data["ids"],
            documents=data["documents"],
            metadatas=data["metadatas"],
            embeddings=data["embeddings"],
        )

    print(f"New collection count after migration: {new_collection.count()}")

    sample_ids = data["ids"][:3]
    if sample_ids:
        print("Sample migrated documents:")
        for item in new_collection.get(ids=sample_ids, include=["metadatas", "documents"]).get("ids", []):
            pass
        print(new_collection.get(ids=sample_ids, include=["metadatas", "documents"]))


if __name__ == "__main__":
    main()
