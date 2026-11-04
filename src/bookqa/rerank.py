"""Reranking: a second, more careful look at the top search hits.

Hybrid search embeds the question and each passage *separately* and compares
the vectors. A reranker reads the question and a passage *together*, which is
much more accurate but too slow to run on every chunk. So search narrows
thousands of chunks to ~50 candidates, and the reranker orders those.

Reranking is pair-oriented: the unit of work is (question, passage). When
several questions are answered at once, their candidate pairs are flattened
into one list and scored in shared batches, so the model is loaded once and
the GPU always sees full batches. Scores are then scattered back per question.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from bookqa.config import RerankerConfig
from bookqa.index import Hits
from bookqa.keyword import tokenize


class Reranker(Protocol):
    def score(self, queries: list[str], documents: list[str]) -> np.ndarray:
        """Relevance of ``documents[i]`` to ``queries[i]``, as float32[n], higher is better."""
        ...

    def close(self) -> None: ...


class NoReranker:
    """Keeps the search order and treats every candidate as relevant."""

    def score(self, queries: list[str], documents: list[str]) -> np.ndarray:
        return np.ones(len(documents), dtype=np.float32)

    def close(self) -> None:
        pass


class OverlapReranker:
    """Share of query terms that appear in the passage. No model; for tests."""

    def score(self, queries: list[str], documents: list[str]) -> np.ndarray:
        out = np.zeros(len(documents), dtype=np.float32)
        for i, (query, doc) in enumerate(zip(queries, documents, strict=True)):
            terms = set(tokenize(query))
            if terms:
                out[i] = len(terms & set(tokenize(doc))) / len(terms)
        return out

    def close(self) -> None:
        pass


def make_reranker(config: RerankerConfig) -> Reranker:
    if config.backend == "none":
        return NoReranker()
    if config.backend == "overlap":
        return OverlapReranker()
    if config.backend == "qwen3":
        from bookqa.local_models import Qwen3Reranker

        return Qwen3Reranker(config)
    raise ValueError(f"unknown reranker backend: {config.backend}")


def rerank_many(
    reranker: Reranker, queries: list[str], hits: list[Hits], texts: list[str], k: int
) -> list[Hits]:
    """Rerank each question's hits and keep the top ``k`` per question.

    ``texts[row]`` is the passage text for a row id. All pairs from all
    questions are scored in one call. Stable sorting keeps the search order
    among equal scores, so ``NoReranker`` returns the original ranking.
    """
    owner = np.concatenate([np.full(len(h), q, dtype=np.int64) for q, h in enumerate(hits)])
    rows = np.concatenate([h.rows for h in hits]) if hits else np.empty(0, dtype=np.int64)
    if len(rows) == 0:
        return [h.top(0) for h in hits]
    scores = reranker.score([queries[q] for q in owner], [texts[r] for r in rows])
    out = []
    for q in range(len(hits)):
        mine = np.flatnonzero(owner == q)
        order = mine[np.argsort(-scores[mine], kind="stable")][:k]
        out.append(Hits(rows[order], scores[order].astype(np.float32)))
    return out
