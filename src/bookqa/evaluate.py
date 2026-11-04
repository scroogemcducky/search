"""Measure retrieval quality on your own questions.

Leaderboard scores come from benchmark datasets that look nothing like your
books, so this module answers the question that matters: for *these* books
and *these* questions, does the right passage come back, and how high?

Questions live in the project's ``questions.jsonl``, one per line:

    {"id": "q1", "question": "Do short naps help memory?",
     "relevant": [{"book": "the-case-for-napping.md", "contains": "20-minute nap"}]}

A relevant passage is named by its book file and a short phrase from it,
not by chunk number. Chunk numbers change whenever chunking settings change,
but a phrase still finds the right chunk. A question with relevant passages
from two or more books counts as a comparison question.

The run is organised by model, not by question, so each model loads once:

    1. for each embedder: load it, embed the corpus (cached on disk) and the
       questions, search, then release it
    2. load the reranker once, and rerank every embedder's candidates for
       every question in one flattened batch
    3. score every ranking
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bookqa.config import Config, EmbeddingConfig, embedding_variant
from bookqa.embed import make_embedder
from bookqa.index import SearchIndex
from bookqa.ingest import contextual_text
from bookqa.keyword import BM25
from bookqa.quotes import locate
from bookqa.rerank import Reranker, rerank_many
from bookqa.tables import Books, Chunks


@dataclasses.dataclass
class EvalQuestion:
    id: str
    question: str
    relevant_rows: np.ndarray  # chunk rows that answer the question
    relevant_books: np.ndarray  # distinct books among those rows

    @property
    def is_comparison(self) -> bool:
        return len(self.relevant_books) >= 2


def load_questions(path: Path, books: Books, chunks: Chunks) -> list[EvalQuestion]:
    """Read questions.jsonl and resolve each relevant phrase to chunk rows."""
    if not path.exists():
        raise FileNotFoundError(f"no questions file at {path}")
    by_source = {source: i for i, source in enumerate(books.source)}
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        rows: list[int] = []
        for ref in item.get("relevant", []):
            book = by_source.get(ref["book"])
            if book is None:
                raise ValueError(f"{path.name}:{n}: unknown book file '{ref['book']}'")
            candidates = np.flatnonzero(chunks.book == book)
            found = [int(r) for r in candidates if locate(ref["contains"], chunks.text[r], 0.95)]
            if not found:
                raise ValueError(
                    f"{path.name}:{n}: no chunk of '{ref['book']}' contains '{ref['contains']}'"
                )
            rows.extend(found)
        relevant = np.unique(np.array(rows, dtype=np.int64))
        out.append(
            EvalQuestion(
                id=str(item.get("id", n)),
                question=item["question"],
                relevant_rows=relevant,
                relevant_books=np.unique(chunks.book[relevant]),
            )
        )
    return out


# --- Metrics ----------------------------------------------------------------------


def metrics(ranking: np.ndarray, q: EvalQuestion, book_of_row: np.ndarray, ks: list[int]) -> dict:
    """Scores for one ranking of one question.

    * recall@k: share of relevant chunks found in the top k
    * mrr: 1 / rank of the first relevant chunk (0 if none is found)
    * ndcg@10: rewards relevant chunks for being near the top
    * books@k: for comparison questions, share of the relevant books that
      have at least one relevant chunk in the top k
    """
    hit = np.isin(ranking, q.relevant_rows)
    n_rel = len(q.relevant_rows)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = hit[:k].sum() / n_rel if n_rel else 0.0
    first = np.flatnonzero(hit)
    out["mrr"] = 1.0 / (first[0] + 1) if len(first) else 0.0
    top = hit[:10]
    gains = top / np.log2(np.arange(2, len(top) + 2))
    ideal = (1 / np.log2(np.arange(2, min(10, n_rel) + 2))).sum()
    out["ndcg@10"] = gains.sum() / ideal if ideal else 0.0
    if q.is_comparison:
        for k in ks:
            found = np.unique(book_of_row[ranking[:k][hit[:k]]])
            out[f"books@{k}"] = len(found) / len(q.relevant_books)
    return out


def summarise(rows: list[dict]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r if k != "id"}, key=_metric_order)
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def _metric_order(name: str) -> tuple:
    kind, _, k = name.partition("@")
    return (["recall", "mrr", "ndcg", "books"].index(kind), int(k) if k else 0)


# --- The run ----------------------------------------------------------------------


@dataclasses.dataclass
class EvalResult:
    name: str  # e.g. "Qwen/Qwen3-Embedding-4B + rerank"
    summary: dict[str, float]
    per_question: list[dict]


def _cached_vectors(
    config: EmbeddingConfig, texts: list[str], cache_dir: Path, log: Callable[[str], None]
) -> tuple[np.ndarray, Callable[[list[str]], np.ndarray], Callable[[], None]]:
    """Corpus vectors for one embedder, from disk if this exact corpus was embedded before."""
    embedder = make_embedder(config)
    digest = hashlib.sha256(json.dumps([embedder.fingerprint(), texts]).encode()).hexdigest()[:16]
    path = cache_dir / f"vectors-{digest}.npy"
    if path.exists():
        vectors = np.load(path)
    else:
        log(f"  embedding {len(texts)} chunks")
        vectors = embedder.embed_documents(texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    return vectors, embedder.embed_queries, embedder.close


def evaluate(
    questions: list[EvalQuestion],
    books: Books,
    chunks: Chunks,
    config: Config,
    make_reranker: Callable[[], Reranker] | None,
    cache_dir: Path,
    log: Callable[[str], None] = lambda _: None,
) -> list[EvalResult]:
    texts = contextual_text(books, chunks)
    queries = [q.question for q in questions]
    depth = max([*config.eval.ks, config.reranker.candidates])
    variants = [config.embedding] + [
        embedding_variant(config.embedding, o) for o in config.eval.embedders
    ]

    # 1. One model at a time: embed, search, release. BM25 is shared by all.
    bm25 = BM25.build(texts)
    keyword_index = SearchIndex(np.zeros((len(texts), 1), dtype=np.float32), bm25, "")
    names = ["keyword only (BM25)"]
    rankings = [[keyword_index.keyword_search(q, depth) for q in queries]]
    for variant in variants:
        label = _label(variant)
        log(f"Searching with {label}")
        vectors, embed_queries, close = _cached_vectors(variant, texts, cache_dir, log)
        index = SearchIndex(vectors, bm25, "")
        query_vectors = embed_queries(queries)
        close()
        names.append(f"{label} (vector only)")
        rankings.append([index.vector_search(v, depth) for v in query_vectors])
        names.append(f"{label} (hybrid)")
        rankings.append(
            [index.search(q, v, depth) for q, v in zip(queries, query_vectors, strict=True)]
        )

    # 2. Rerank every hybrid ranking in one flattened batch.
    if make_reranker is not None:
        hybrid = [i for i, n in enumerate(names) if n.endswith("(hybrid)")]
        log(f"Reranking the top {config.reranker.candidates} for {len(hybrid)} embedder(s)")
        reranker = make_reranker()
        flat_queries = [q for _ in hybrid for q in queries]
        flat_hits = [h.top(config.reranker.candidates) for i in hybrid for h in rankings[i]]
        reranked = rerank_many(reranker, flat_queries, flat_hits, texts, depth)
        reranker.close()
        for n, i in enumerate(hybrid):
            names.append(names[i].replace("(hybrid)", "(hybrid + rerank)"))
            rankings.append(reranked[n * len(queries) : (n + 1) * len(queries)])

    # 3. Score.
    results = []
    for name, ranked in zip(names, rankings, strict=True):
        per_question = [
            {"id": q.id, **metrics(h.rows, q, chunks.book, config.eval.ks)}
            for q, h in zip(questions, ranked, strict=True)
        ]
        results.append(EvalResult(name, summarise(per_question), per_question))
    return results


def _label(config: EmbeddingConfig) -> str:
    name = "hashing" if config.backend == "hashing" else config.model
    return f"{name} {config.dimensions}d" if config.dimensions else name


def report(results: list[EvalResult], questions: list[EvalQuestion], ks: list[int]) -> str:
    """A Markdown table of mean scores, plus advice drawn from the numbers."""
    columns = list(results[0].summary)
    lines = [
        f"Evaluated {len(questions)} questions "
        f"({sum(q.is_comparison for q in questions)} comparison questions).",
        "",
        "| method | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for r in results:
        lines.append(f"| {r.name} | " + " | ".join(f"{r.summary[c]:.3f}" for c in columns) + " |")

    lo, hi = f"recall@{min(ks)}", f"recall@{max(ks)}"
    best = max(results, key=lambda r: r.summary.get("ndcg@10", 0))
    lines += ["", f"Best by nDCG@10: **{best.name}**."]
    if best.summary.get(hi, 0) - best.summary.get(lo, 0) > 0.2:
        lines.append(
            f"The right passages often reach the top {max(ks)} but not the top {min(ks)} "
            f"({best.summary[hi]:.2f} vs {best.summary[lo]:.2f}). Improve reranking before "
            "changing the embedding model."
        )
    return "\n".join(lines) + "\n"
