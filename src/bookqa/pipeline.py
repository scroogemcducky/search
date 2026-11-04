"""The pipeline's stages, each reading and writing files in ``.bookqa/``.

    ingest   books/            -> books.parquet, chunks.parquet
    extract  chunks            -> claims.parquet             (LLM, cached)
    index    chunks, claims    -> *_vectors.npy, *_bm25.npz  (embedder)
    ask      questions         -> output/<question>/*.md     (embedder, reranker, LLM)
    eval     questions.jsonl   -> .bookqa/eval/report.md     (embedders, reranker)

Every stage reads its inputs from disk and writes its outputs to disk, so
each can be re-run alone, and a failure part-way only costs that stage.

``manifest.json`` records a fingerprint of the chunk table each later stage
was built from. If the books or chunk settings change, stages that depend on
the old chunks refuse to run until rebuilt, instead of quietly mixing claims
from one chunking with vectors from another.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

from bookqa.answer import Answer, Corpus, answer_questions, write_answer
from bookqa.claims import ExtractionStats, claim_search_text, extract_claims
from bookqa.config import Project
from bookqa.embed import HashingEmbedder, make_embedder
from bookqa.evaluate import evaluate, load_questions, report
from bookqa.index import SearchIndex
from bookqa.ingest import contextual_text, ingest
from bookqa.llm import LLM, ResponseCache, make_llm
from bookqa.offline import offline_llm
from bookqa.prompts import load_prompt
from bookqa.rerank import OverlapReranker, make_reranker
from bookqa.retrieve import retrieve
from bookqa.tables import Books, Chunks, Claims

Log = Callable[[str], None]


@dataclasses.dataclass
class Pipeline:
    project: Project
    offline: bool = False  # use model-free stand-ins for every model
    log: Log = print

    # --- Model factories (offline mode swaps in stand-ins) ---------------------

    def embedder(self):
        if self.offline:
            return HashingEmbedder(256)
        return make_embedder(self.project.config.embedding)

    def reranker(self):
        if self.offline:
            return OverlapReranker()
        return make_reranker(self.project.config.reranker)

    def llm(self) -> LLM:
        cache = ResponseCache(self.project.data("llm_cache.sqlite"))
        if self.offline:
            return offline_llm(cache)
        return make_llm(self.project.config.llm, self.project.data("llm_cache.sqlite"))

    # --- Manifest ----------------------------------------------------------------

    def _manifest(self) -> dict:
        path = self.project.data("manifest.json")
        return json.loads(path.read_text()) if path.exists() else {}

    def _record(self, stage: str, chunks_id: str) -> None:
        manifest = self._manifest()
        manifest[stage] = {"chunks": chunks_id, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.project.data_dir.mkdir(parents=True, exist_ok=True)
        self.project.data("manifest.json").write_text(json.dumps(manifest, indent=2))

    def _require(self, stage: str, chunks_id: str) -> None:
        recorded = self._manifest().get(stage, {}).get("chunks")
        if recorded is None:
            raise RuntimeError(f"run `bookqa {stage}` first")
        if recorded != chunks_id:
            raise RuntimeError(
                f"the books or chunk settings changed since `bookqa {stage}` last ran; "
                "run `bookqa build` to rebuild"
            )

    # --- Loading -------------------------------------------------------------------

    def load_chunks(self) -> tuple[Books, Chunks, str]:
        path = self.project.data("chunks.parquet")
        if not path.exists():
            raise RuntimeError("no chunks yet; run `bookqa ingest` first")
        books = Books.load(self.project.data("books.parquet"))
        chunks = Chunks.load(path)
        return books, chunks, chunks_fingerprint(chunks)

    def load_corpus(self) -> tuple[Corpus, str]:
        books, chunks, chunks_id = self.load_chunks()
        self._require("extract", chunks_id)
        return Corpus(books, chunks, Claims.load(self.project.data("claims.parquet"))), chunks_id

    # --- Stages --------------------------------------------------------------------

    def ingest(self) -> tuple[Books, Chunks]:
        books, chunks = ingest(self.project.books_dir, self.project.config.chunking)
        if len(books) == 0:
            raise RuntimeError(f"no books found in {self.project.books_dir}")
        books.save(self.project.data("books.parquet"))
        chunks.save(self.project.data("chunks.parquet"))
        self._record("ingest", chunks_fingerprint(chunks))
        words = sum(len(t.split()) for t in chunks.text)
        self.log(f"Read {len(books)} books: {len(chunks)} chunks, {words:,} words")
        return books, chunks

    def extract(self) -> ExtractionStats:
        books, chunks, chunks_id = self.load_chunks()
        llm = self.llm()
        prompt = load_prompt("extract_claims", self.project.prompts_dir)
        start = time.time()

        def progress(done: int, total: int) -> None:
            if done == total or done % 10 == 0:
                rate = (time.time() - start) / max(done, 1)
                self.log(f"  {done}/{total} calls, about {rate * (total - done) / 60:.0f} min left")

        self.log("Extracting claims")
        claims, stats = extract_claims(
            books, chunks, llm, prompt, self.project.config.claims, progress
        )
        claims.save(self.project.data("claims.parquet"))
        self._record("extract", chunks_id)
        self.log(stats.summary())
        return stats

    def index(self) -> None:
        corpus, chunks_id = self.load_corpus()
        embedder = self.embedder()
        self.log(f"Embedding {len(corpus.chunks)} chunks and {len(corpus.claims)} claims")
        SearchIndex.build(contextual_text(corpus.books, corpus.chunks), embedder).save(
            self.project.data_dir, "chunk"
        )
        SearchIndex.build(
            claim_search_text(corpus.books, corpus.chunks, corpus.claims), embedder
        ).save(self.project.data_dir, "claim")
        embedder.close()
        self._record("index", chunks_id)

    def build(self) -> None:
        self.ingest()
        self.extract()
        self.index()

    def ask(self, questions: list[str], out_dir: Path | None = None) -> list[Answer]:
        corpus, chunks_id = self.load_corpus()
        self._require("index", chunks_id)
        out_dir = out_dir or self.project.root / "output"

        embedder = self.embedder()
        fingerprint = embedder.fingerprint()
        chunk_index = SearchIndex.load(self.project.data_dir, "chunk", fingerprint)
        claim_index = SearchIndex.load(self.project.data_dir, "claim", fingerprint)

        self.log(f"Retrieving evidence for {len(questions)} question(s)")
        # The embedder is released right after embedding the questions, and the
        # reranker only loads when retrieve first needs it, so the two models
        # are never in memory at the same time.
        evidence = retrieve(
            questions,
            corpus.books,
            corpus.chunks,
            corpus.claims,
            chunk_index,
            claim_index,
            _ReleasingEmbedder(embedder),
            _LazyReranker(self.reranker),
            self.project.config,
        )
        answers = answer_questions(
            questions,
            evidence,
            corpus,
            self.llm(),
            self.project.config,
            self.project.prompts_dir,
            self.log,
        )
        for a in answers:
            folder = out_dir / slugify(a.question)
            write_answer(a, corpus, self.project.config, folder)
            status = "verified" if not a.problems else f"{len(a.problems)} problem(s) remain"
            self.log(f"  {folder.relative_to(self.project.root)}/ ({status})")
        return answers

    def eval(self) -> str:
        books, chunks, _ = self.load_chunks()
        questions = load_questions(self.project.questions_file, books, chunks)
        config = self.project.config
        if self.offline:
            config = dataclasses.replace(
                config,
                embedding=dataclasses.replace(config.embedding, backend="hashing", dimensions=256),
                eval=dataclasses.replace(config.eval, embedders=[]),
            )
        make = None if config.reranker.backend == "none" else self.reranker
        results = evaluate(
            questions, books, chunks, config, make, self.project.data("eval"), self.log
        )
        text = report(results, questions, config.eval.ks)
        self.project.data("eval").mkdir(parents=True, exist_ok=True)
        self.project.data("eval/report.md").write_text(text)
        per_question = [{"method": r.name, **row} for r in results for row in r.per_question]
        self.project.data("eval/per_question.json").write_text(json.dumps(per_question, indent=1))
        return text


class _ReleasingEmbedder:
    """Embeds the questions, then frees the model before reranking starts."""

    def __init__(self, inner):
        self.inner = inner

    def embed_queries(self, texts):
        vectors = self.inner.embed_queries(texts)
        self.inner.close()
        return vectors


class _LazyReranker:
    """Loads the reranker on first use and frees it after."""

    def __init__(self, factory):
        self.factory = factory

    def score(self, queries, documents):
        reranker = self.factory()
        try:
            return reranker.score(queries, documents)
        finally:
            reranker.close()

    def close(self):
        pass


def chunks_fingerprint(chunks: Chunks) -> str:
    h = hashlib.sha256()
    for i in range(len(chunks)):
        h.update(f"{chunks.book[i]}\x00{chunks.text[i]}\x01".encode())
    return h.hexdigest()[:16]


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "question"
