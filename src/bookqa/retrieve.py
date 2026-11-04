"""Gather the evidence for a batch of questions.

For each question this finds the best passages and a *balanced* set of
claims. All questions are processed together, one stage at a time: embed
every question, search for every question, rerank every candidate. Each model
loads once per batch rather than once per question.

The result is one flat ``Evidence`` table for all questions. Its
``question`` column says which question each row belongs to, so a question's
evidence is a contiguous slice and there's no nested list-of-lists.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from bookqa.claims import claim_search_text
from bookqa.config import Config
from bookqa.embed import Embedder
from bookqa.index import Hits, SearchIndex
from bookqa.ingest import contextual_text
from bookqa.rerank import Reranker, rerank_many
from bookqa.tables import Books, Chunks, Claims, Table, col, strings

PASSAGE, CLAIM = 0, 1


@dataclasses.dataclass(frozen=True)
class Evidence(Table):
    """Evidence for many questions. Rows are grouped by question, best first."""

    question: np.ndarray = col(np.int32)
    kind: np.ndarray = col(np.int8)  # PASSAGE or CLAIM
    row: np.ndarray = col(np.int32)  # row in Chunks or Claims, depending on kind
    score: np.ndarray = col(np.float32)  # reranker score
    label: list[str] = strings()  # "P1", "C3": what the answer cites

    def for_question(self, q: int) -> Evidence:
        return self.take(np.flatnonzero(self.question == q))


def balance(hits: Hits, book_of_row: np.ndarray, per_book: int, total: int) -> Hits:
    """Pick up to ``total`` hits, at most ``per_book`` per book, interleaving books.

    Plain top-k lets one book that discusses a topic at length crowd out the
    others, which is exactly wrong for comparison questions. Here each hit
    gets its rank *within its book*. Hits are then ordered by that rank, so
    every book's best hit comes first, then every book's second best, and so
    on. All of this is sorting and array arithmetic, with no per-book Python loop.
    """
    if len(hits) == 0:
        return hits
    book = book_of_row[hits.rows]
    by_book = np.lexsort((-hits.scores, book))  # grouped by book, best first within it
    sorted_book = book[by_book]
    run_start = np.flatnonzero(np.r_[True, sorted_book[1:] != sorted_book[:-1]])
    run_length = np.diff(np.r_[run_start, len(by_book)])
    rank_sorted = np.arange(len(by_book)) - np.repeat(run_start, run_length)
    rank = np.empty_like(rank_sorted)
    rank[by_book] = rank_sorted

    keep = np.flatnonzero(rank < per_book)
    keep = keep[np.lexsort((-hits.scores[keep], rank[keep]))][:total]
    return Hits(hits.rows[keep], hits.scores[keep])


def retrieve(
    questions: list[str],
    books: Books,
    chunks: Chunks,
    claims: Claims,
    chunk_index: SearchIndex,
    claim_index: SearchIndex,
    embedder: Embedder,
    reranker: Reranker,
    config: Config,
) -> Evidence:
    a = config.answer
    vectors = embedder.embed_queries(questions)

    chunk_hits = [
        chunk_index.search(q, v, config.reranker.candidates)
        for q, v in zip(questions, vectors, strict=True)
    ]
    claim_hits = [
        claim_index.search(q, v, config.reranker.candidates)
        for q, v in zip(questions, vectors, strict=True)
    ]

    # Rerank passages and claims in one flattened batch, so the reranker
    # loads once. Claim rows are shifted past the chunk rows so a single
    # text list covers both.
    offset = len(chunks)
    texts = contextual_text(books, chunks) + claim_search_text(books, chunks, claims)
    shifted = [Hits(h.rows + offset, h.scores) for h in claim_hits]
    ranked = rerank_many(reranker, questions + questions, chunk_hits + shifted, texts, len(texts))
    n = len(questions)
    passages = [ranked[q].top(a.passages) for q in range(n)]
    ranked_claims = [Hits(ranked[n + q].rows - offset, ranked[n + q].scores) for q in range(n)]

    parts = []
    for q in range(len(questions)):
        p = _relevant(passages[q], a.min_relevance)
        c = balance(
            _relevant(ranked_claims[q], a.min_relevance),
            claims.book,
            a.max_claims_per_book,
            a.claims,
        )
        parts.append(
            Evidence(
                question=np.full(len(p) + len(c), q, dtype=np.int32),
                kind=np.r_[np.full(len(p), PASSAGE), np.full(len(c), CLAIM)].astype(np.int8),
                row=np.r_[p.rows, c.rows].astype(np.int32),
                score=np.r_[p.scores, c.scores].astype(np.float32),
                label=[f"P{i + 1}" for i in range(len(p))] + [f"C{i + 1}" for i in range(len(c))],
            )
        )
    return Evidence.concat(parts)


def _relevant(hits: Hits, threshold: float) -> Hits:
    keep = hits.scores >= threshold
    return Hits(hits.rows[keep], hits.scores[keep])
