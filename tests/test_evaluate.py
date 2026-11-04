import json
from pathlib import Path

import numpy as np
import pytest

from bookqa.config import ChunkingConfig, Config, embedding_variant
from bookqa.evaluate import EvalQuestion, evaluate, load_questions, metrics, report
from bookqa.ingest import ingest
from bookqa.rerank import OverlapReranker

DEMO = Path(__file__).parent.parent / "examples" / "demo"


def question(rows, books):
    return EvalQuestion("q", "?", np.array(rows), np.array(books))


def test_metrics():
    q = question([3, 7], [0, 1])
    book_of_row = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = metrics(np.array([5, 3, 1, 7]), q, book_of_row, ks=[1, 2, 4])
    assert m["recall@1"] == 0 and m["recall@2"] == 0.5 and m["recall@4"] == 1
    assert m["mrr"] == 0.5
    assert m["books@2"] == 0.5 and m["books@4"] == 1
    assert 0 < m["ndcg@10"] < 1
    perfect = metrics(np.array([3, 7, 1]), q, book_of_row, ks=[2])
    assert perfect["ndcg@10"] == pytest.approx(1.0)
    assert metrics(np.array([1, 2]), q, book_of_row, ks=[2])["mrr"] == 0


def test_load_questions_resolves_phrases():
    books, chunks = ingest(DEMO / "books", ChunkingConfig())
    questions = load_questions(DEMO / "questions.jsonl", books, chunks)
    assert [q.id for q in questions][:2] == ["short-naps", "naps-and-health"]
    assert questions[0].is_comparison and not questions[1].is_comparison
    for q in questions:
        assert len(q.relevant_rows)


def test_load_questions_reports_bad_phrase(tmp_path):
    books, chunks = ingest(DEMO / "books", ChunkingConfig())
    path = tmp_path / "q.jsonl"
    ref = {"book": "sleep-across-cultures.txt", "contains": "words that are not in it"}
    path.write_text(json.dumps({"question": "x", "relevant": [ref]}))
    with pytest.raises(ValueError, match="no chunk"):
        load_questions(path, books, chunks)


def test_embedding_variant_rejects_typos():
    base = Config().embedding
    assert embedding_variant(base, {"model": "m"}).model == "m"
    assert embedding_variant(base, {"model": "m"}).dimensions == base.dimensions
    with pytest.raises(ValueError):
        embedding_variant(base, {"modle": "m"})


def test_evaluate_compares_methods(tmp_path):
    config = Config()
    config.chunking.target_words = 40
    config.embedding.backend = "hashing"
    config.embedding.dimensions = 128
    config.eval.embedders = [{"dimensions": 512, "model": "wide-hash"}]
    config.eval.ks = [1, 5]
    books, chunks = ingest(DEMO / "books", config.chunking)
    questions = load_questions(DEMO / "questions.jsonl", books, chunks)
    results = evaluate(questions, books, chunks, config, OverlapReranker, tmp_path)
    names = [r.name for r in results]
    assert names[0] == "keyword only (BM25)"
    assert sum("rerank" in n for n in names) == 2
    assert len(names) == 7
    assert list(tmp_path.glob("vectors-*.npy"))  # corpus vectors cached
    text = report(results, questions, config.eval.ks)
    assert "| keyword only (BM25) |" in text and "recall@5" in text
