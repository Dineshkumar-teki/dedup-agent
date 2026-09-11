"""Intent-weighted bag-of-words embeddings for offline dedup evals."""

from __future__ import annotations

import math
import re
from collections import Counter

# Phrases that should dominate cosine similarity when enrichment is doing its job.
INTENT_PHRASES: tuple[tuple[str, float], ...] = (
    ("left join", 8.0),
    ("inner join", 8.0),
    ("right join", 8.0),
    ("self-join", 8.0),
    ("self join", 8.0),
    ("second-highest", 10.0),
    ("second highest", 10.0),
    ("third-highest", 10.0),
    ("third highest", 10.0),
    ("two-sum", 8.0),
    ("three-sum", 8.0),
    ("two sum", 8.0),
    ("three sum", 8.0),
    ("ranking", 3.0),
    ("aggregation", 3.0),
    ("grouping", 3.0),
    ("distinct", 5.0),
    ("palindrome", 6.0),
    ("anagram", 6.0),
    ("promise", 4.0),
    ("by customer", 8.0),
    ("by day", 8.0),
    ("per customer", 8.0),
    ("per day", 8.0),
    ("forwards and backwards", 6.0),
    ("equals its reverse", 6.0),
    ("react", 3.0),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _intent_tokens(text: str) -> Counter[str]:
    lowered = text.lower()
    counts: Counter[str] = Counter()
    for phrase, weight in INTENT_PHRASES:
        if phrase in lowered:
            counts[f"intent:{phrase}"] += weight
    for token in tokenize(text):
        counts[token] += 1.0
    for left, right in zip(tokenize(text), tokenize(text)[1:]):
        counts[f"{left} {right}"] += 1.5
    return counts


class IntentBagEmbeddings:
    """Deterministic embeddings that overweight operation/rank phrases."""

    def __init__(self, texts: list[str]) -> None:
        vocab: set[str] = set()
        self._docs = [_intent_tokens(text) for text in texts]
        for doc in self._docs:
            vocab.update(doc)
        self._index = {token: idx for idx, token in enumerate(sorted(vocab))}
        self._cache: dict[str, list[float]] = {}
        for text, doc in zip(texts, self._docs):
            self._cache[text] = self._vector_from_counts(doc)

    def _vector_from_counts(self, counts: Counter[str]) -> list[float]:
        vector = [0.0] * max(len(self._index), 1)
        for token, weight in counts.items():
            idx = self._index.get(token)
            if idx is not None:
                vector[idx] = weight
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            cached = self._cache.get(text)
            if cached is not None:
                vectors.append(cached)
                continue
            vector = self._vector_from_counts(_intent_tokens(text))
            self._cache[text] = vector
            vectors.append(vector)
        return vectors
