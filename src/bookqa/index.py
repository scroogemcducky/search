"""Hybrid search over any table with a text column.

The same index type serves chunks and claims. It needs only "row ``i`` has
text ``texts[i]``", which is the payoff of ids being row indices: search code
never needs to know what kind of row it's searching.

An index is two arrays saved side by side:

    <name>_vectors.npy   float32[n_rows, dim], unit-length rows
    <name>_bm25.npz      inverted keyword index (see keyword.py)

plus a small JSON file recording which embedding model made the vectors, so a
model change is detected instead of silently mixing incompatible vectors.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from bookqa.embed import Embedder
from bookqa.keyword import BM25

RRF_K = 60  # standard constant from the reciprocal rank fusion paper


@dataclasses.dataclass(frozen=True)
class Hits:
    """Search results as two parallel arrays, best first."""

    rows: np.ndarray  # int64 row ids into the searched table
    scores: np.ndarray  # float32, higher is better; scale depends on the method

    def __len__(self) -> int:
        return len(self.rows)

    def top(self, k: int) -> Hits:
        return Hits(self.rows[:k], self.scores[:k])


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> Hits:
    """Best ``k`` rows by score.

    ``argpartition`` finds the top k in linear time; only those k are sorted.
    ``mask`` (bool[n_rows]) restricts results, e.g. to one book's rows.
    """
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
    k = min(k, len(scores))
    if k <= 0:
        return Hits(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.lexsort((part, -scores[part]))]  # ties broken by row id, for stability
    return Hits(order.astype(np.int64), scores[order].astype(np.float32))


def reciprocal_rank_fusion(rankings: list[Hits], n_rows: int, k: int) -> Hits:
    """Merge rankings by summing 1 / (RRF_K + rank) for each row.

    RRF uses ranks, not raw scores, so BM25 scores (unbounded) and cosine
    similarities (-1 to 1) can be combined without calibrating them.
    """
    fused = np.zeros(n_rows, dtype=np.float32)
    for ranking in rankings:
        ranks = np.arange(1, len(ranking) + 1, dtype=np.float32)
        np.add.at(fused, ranking.rows, 1.0 / (RRF_K + ranks))
    seen = np.zeros(n_rows, dtype=bool)
    for ranking in rankings:
        seen[ranking.rows] = True
    return top_k(fused, k, seen)


@dataclasses.dataclass(frozen=True)
class SearchIndex:
    vectors: np.ndarray
    bm25: BM25
    fingerprint: str

    @classmethod
    def build(cls, texts: list[str], embedder: Embedder) -> SearchIndex:
        vectors = embedder.embed_documents(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        return cls(vectors, BM25.build(texts), embedder.fingerprint())

    def __len__(self) -> int:
        return len(self.vectors)

    def vector_search(self, query_vector: np.ndarray, k: int, mask=None) -> Hits:
        if len(self) == 0:
            return top_k(np.zeros(0, dtype=np.float32), k)
        return top_k(self.vectors @ query_vector, k, mask)

    def keyword_search(self, query: str, k: int, mask=None) -> Hits:
        scores = self.bm25.scores(query)
        hits = top_k(scores, k, mask)
        keep = hits.scores > 0  # rows sharing no term with the query aren't matches
        return Hits(hits.rows[keep], hits.scores[keep])

    def search(
        self, query: str, query_vector: np.ndarray, k: int, mask: np.ndarray | None = None
    ) -> Hits:
        """Hybrid search: vector and keyword rankings fused with RRF.

        Each method contributes a deeper list (3k) than requested, so a row
        ranked moderately by both can still surface in the fused top k.
        """
        depth = max(3 * k, 50)
        rankings = [
            self.vector_search(query_vector, depth, mask),
            self.keyword_search(query, depth, mask),
        ]
        return reciprocal_rank_fusion(rankings, len(self), k)

    def save(self, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{name}_vectors.npy", self.vectors)
        self.bm25.save(directory / f"{name}_bm25.npz")
        meta = {"fingerprint": self.fingerprint, "rows": len(self)}
        (directory / f"{name}_index.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path, name: str, expected_fingerprint: str | None = None):
        meta_path = directory / f"{name}_index.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no '{name}' index in {directory}; run `bookqa index` first")
        meta = json.loads(meta_path.read_text())
        if expected_fingerprint is not None and meta["fingerprint"] != expected_fingerprint:
            raise ValueError(
                f"the '{name}' index was built with {meta['fingerprint']}, but the config now "
                f"says {expected_fingerprint}; rebuild it with `bookqa index`"
            )
        # mmap_mode="r" maps the matrix from disk instead of copying it into memory.
        vectors = np.load(directory / f"{name}_vectors.npy", mmap_mode="r")
        return cls(vectors, BM25.load(directory / f"{name}_bm25.npz"), meta["fingerprint"])
