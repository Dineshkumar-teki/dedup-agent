import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

BORDERLINE_MARGIN = 0.03

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None


def _slugify(subject: str) -> str:
    normalized = subject.strip().lower()
    if not normalized:
        raise ValueError("Subject must not be empty.")

    slug = "".join(ch if ch.isalnum() else "_" for ch in normalized)
    slug = slug.strip("_")
    if not slug:
        raise ValueError("Subject must contain at least one alphanumeric character.")
    return slug


@dataclass(frozen=True)
class StoreResult:
    qid: str
    status: str
    original_qid: str | None = None
    distance: float | None = None
    similarity: float | None = None
    message: str | None = None


class ChromaRAGStore:
    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "sql_enriched_questions",
        duplicate_distance_threshold: float = 0.25,
    ):
        if chromadb is None or Settings is None:
            raise ImportError(
                "chromadb is required for ChromaRAGStore. Install it with 'pip install chromadb'."
            )

        persist_path = Path(persist_dir)
        if not persist_path.is_absolute():
            persist_path = ROOT_DIR / persist_path
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_path))
        # NOTE: OpenAIEmbeddings does not expose input_type parameter (search_query vs search_document).
        # We use symmetric embeddings for both indexing and querying.
        load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to the project .env file or export it before launching the app."
            )

        self.embedding = OpenAIEmbeddings(
            model="openai/text-embedding-3-large",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"source": "sql-dedup-agent", "hnsw:space": "cosine"},
        )
        self.duplicate_distance_threshold = duplicate_distance_threshold

    @classmethod
    def for_subject(
        cls,
        subject: str,
        persist_dir: str = "chroma_db",
        **kwargs,
    ) -> "ChromaRAGStore":
        collection_name = _slugify(subject)
        return cls(persist_dir=persist_dir, collection_name=collection_name, **kwargs)

    def list_subjects(self) -> list[str]:
        subjects: list[str] = []
        for collection_summary in self.client.list_collections():
            try:
                collection = self.client.get_collection(name=collection_summary.name)
                if collection.metadata.get("source") == "sql-dedup-agent":
                    subjects.append(collection_summary.name)
            except Exception:
                continue
        return subjects

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedding.embed_documents(texts)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embedding.embed_documents(texts)

    @staticmethod
    def _distance_to_similarity(distance: float | None) -> float | None:
        if distance is None:
            return None
        similarity = max(0.0, min(1.0, 1.0 - distance))
        return round(similarity * 100.0, 1)

    def _get_by_qid(self, qid: str) -> dict[str, Any] | None:
        try:
            result = self.collection.get(ids=[qid], include=["metadatas", "documents"])
        except Exception:
            return None

        if result.get("ids"):
            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        return None

    def _find_duplicate(self, enriched_text_or_embedding: str | list[float]) -> dict[str, Any] | None:
        if isinstance(enriched_text_or_embedding, list):
            query_embeddings = [enriched_text_or_embedding]
        else:
            query_embeddings = self._embed([enriched_text_or_embedding])
        query_result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=1,
            include=["metadatas", "documents", "distances"],
        )

        if not query_result.get("documents"):
            return None

        best_docs = query_result["documents"][0]
        best_metas = query_result["metadatas"][0]
        best_distances = query_result["distances"][0]

        if not best_docs:
            return None

        return {
            "document": best_docs[0],
            "metadata": best_metas[0],
            "distance": best_distances[0],
        }

    def _store_new_question(
        self,
        qid: str,
        raw_question: str,
        enriched_text: str,
        message: str,
        similarity: float | None = None,
        embedding: list[float] | None = None,
    ) -> StoreResult:
        if embedding is None:
            embedding = self._embed([enriched_text])[0]

        self.collection.add(
            ids=[qid],
            documents=[enriched_text],
            metadatas=[{"qid": qid, "raw_question": raw_question}],
            embeddings=[embedding],
        )
        return StoreResult(
            qid=qid,
            status="stored",
            original_qid=qid,
            similarity=similarity,
            message=message,
        )

    def add_or_flag_duplicate(
        self,
        qid: str,
        raw_question: str,
        enriched_text: str,
        allow_duplicate: bool = False,
    ) -> StoreResult:
        existing_by_qid = self._get_by_qid(qid)
        if existing_by_qid is not None:
            if existing_by_qid["document"] == enriched_text:
                return StoreResult(
                    qid=qid,
                    status="already_stored",
                    original_qid=qid,
                    message="Question already stored.",
                )
            return StoreResult(
                qid=qid,
                status="qid_conflict",
                original_qid=qid,
                message="QID conflict: different question exists.",
            )

        duplicate = self._find_duplicate(enriched_text)
        if duplicate is not None:
            duplicate_qid = duplicate["metadata"].get("qid")
            distance = duplicate.get("distance")
            similarity = self._distance_to_similarity(distance)
            if distance is not None and distance <= self.duplicate_distance_threshold:
                if allow_duplicate:
                    return self._store_new_question(
                        qid,
                        raw_question,
                        enriched_text,
                        message="Duplicate question stored by user request.",
                        similarity=similarity,
                    )
                if duplicate["document"] == enriched_text:
                    return StoreResult(
                        qid=qid,
                        status="duplicate",
                        original_qid=duplicate_qid,
                        distance=distance,
                        similarity=similarity,
                        message="Exact duplicate found.",
                    )
                return StoreResult(
                    qid=qid,
                    status="duplicate",
                    original_qid=duplicate_qid,
                    distance=distance,
                    similarity=similarity,
                    message="Similar question found.",
                )

        nearest_distance = None
        if duplicate is not None:
            nearest_distance = duplicate.get("distance")

        result = self._store_new_question(
            qid,
            raw_question,
            enriched_text,
            message="Question stored.",
        )

        if (
            nearest_distance is not None
            and abs(nearest_distance - self.duplicate_distance_threshold) <= BORDERLINE_MARGIN
        ):
            result = StoreResult(
                qid=result.qid,
                status="borderline_stored",
                original_qid=result.original_qid,
                distance=result.distance,
                similarity=result.similarity,
                message=result.message,
            )

        return result

    def check_duplicate(self, qid: str, enriched_text: str) -> StoreResult:
        existing = self._get_by_qid(qid)
        if existing is not None:
            if existing["document"] == enriched_text:
                return StoreResult(
                    qid=qid,
                    status="already_exists",
                    original_qid=qid,
                    message="Question already exists.",
                )
            return StoreResult(
                qid=qid,
                status="qid_conflict",
                original_qid=qid,
                message="QID conflict: different question exists.",
            )

        embedding = self._embed([enriched_text])[0]
        duplicate = self._find_duplicate(embedding)
        if duplicate is not None:
            duplicate_qid = duplicate["metadata"].get("qid")
            distance = duplicate.get("distance")
            similarity = self._distance_to_similarity(distance)
            if distance is not None and distance <= self.duplicate_distance_threshold:
                if duplicate["document"] == enriched_text:
                    return StoreResult(
                        qid=qid,
                        status="duplicate",
                        original_qid=duplicate_qid,
                        distance=distance,
                        similarity=similarity,
                        message="Exact duplicate found.",
                    )
                return StoreResult(
                    qid=qid,
                    status="duplicate",
                    original_qid=duplicate_qid,
                    distance=distance,
                    similarity=similarity,
                    message="Similar question found.",
                )

        return StoreResult(
            qid=qid,
            status="new",
            original_qid=qid,
            message="No duplicate found.",
        )

    def batch_store(self, rows: list[tuple[str, str, str]]) -> list[StoreResult]:
        enriched_texts = [enriched_text for _, _, enriched_text in rows]
        embeddings = self.embed_batch(enriched_texts)

        results: list[StoreResult] = []
        for (qid, raw_question, enriched_text), embedding in zip(rows, embeddings):
            existing_by_qid = self._get_by_qid(qid)
            if existing_by_qid is not None:
                if existing_by_qid["document"] == enriched_text:
                    results.append(
                        StoreResult(
                            qid=qid,
                            status="already_stored",
                            original_qid=qid,
                            message="Question already stored.",
                        )
                    )
                else:
                    results.append(
                        StoreResult(
                            qid=qid,
                            status="qid_conflict",
                            original_qid=qid,
                            message="QID conflict: different question exists.",
                        )
                    )
                continue

            duplicate = self._find_duplicate(embedding)
            if duplicate is not None:
                duplicate_qid = duplicate["metadata"].get("qid")
                distance = duplicate.get("distance")
                similarity = self._distance_to_similarity(distance)
                if distance is not None and distance <= self.duplicate_distance_threshold:
                    if duplicate["document"] == enriched_text:
                        results.append(
                            StoreResult(
                                qid=qid,
                                status="duplicate",
                                original_qid=duplicate_qid,
                                distance=distance,
                                similarity=similarity,
                                message="Exact duplicate found.",
                            )
                        )
                    else:
                        results.append(
                            StoreResult(
                                qid=qid,
                                status="duplicate",
                                original_qid=duplicate_qid,
                                distance=distance,
                                similarity=similarity,
                                message="Similar question found.",
                            )
                        )
                    continue

            nearest_distance = None
            if duplicate is not None:
                nearest_distance = duplicate.get("distance")

            result = self._store_new_question(
                qid,
                raw_question,
                enriched_text,
                message="Question stored.",
                embedding=embedding,
            )

            if (
                nearest_distance is not None
                and abs(nearest_distance - self.duplicate_distance_threshold) <= BORDERLINE_MARGIN
            ):
                result = StoreResult(
                    qid=result.qid,
                    status="borderline_stored",
                    original_qid=result.original_qid,
                    distance=result.distance,
                    similarity=result.similarity,
                    message=result.message,
                )

            results.append(result)

        return results
