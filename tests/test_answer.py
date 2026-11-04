from pathlib import Path

import numpy as np
from fakes import UNCITED, fake_llm

from bookqa.answer import Corpus, answer_questions, parse_positions, write_answer
from bookqa.claims import claim_search_text, extract_claims
from bookqa.config import Config
from bookqa.embed import HashingEmbedder
from bookqa.index import Hits, SearchIndex
from bookqa.ingest import contextual_text, ingest
from bookqa.prompts import load_prompt
from bookqa.rerank import OverlapReranker
from bookqa.retrieve import CLAIM, PASSAGE, balance, retrieve
from bookqa.verify import cited_labels, sentences, verify

DEMO_BOOKS = Path(__file__).parent.parent / "examples" / "demo" / "books"


def test_balance_interleaves_books_and_caps_each():
    book_of_row = np.array([0, 0, 0, 0, 1, 2])
    hits = Hits(np.array([0, 1, 2, 3, 4, 5]), np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
    out = balance(hits, book_of_row, per_book=2, total=10)
    # Every book's best first, then second bests; book 0 capped at two.
    assert out.rows.tolist() == [0, 4, 5, 1]
    assert balance(hits, book_of_row, per_book=2, total=2).rows.tolist() == [0, 4]


def test_cited_labels_and_sentences():
    text = (
        "## Heading\n\n- Naps help memory in adults [C1, P2]. Walks help too, says one author. [C3]"
    )
    assert cited_labels(text) == ["C1", "P2", "C3"]
    assert sentences(text) == [
        "Naps help memory in adults [C1, P2].",
        "Walks help too, says one author [C3].",
    ]


def test_verify_finds_each_problem_kind():
    text = (
        "A large study found naps improve memory [C2]. "
        "Naps are linked to better sleep overall [C9]. "
        "The author has a personal view on this one."
    )
    problems = verify(text, {"C1", "C2", "P1"}, research_labels={"C1"}, research_words=["study"])
    assert [p.kind for p in problems] == ["evidence_mismatch", "unknown_source", "uncited"]
    assert verify("A large study found naps improve memory [C1].", {"C1"}, {"C1"}, ["study"]) == []


def test_parse_positions_drops_unknown_labels():
    reply = (
        '{"positions": [{"position": "Yes", "claims": ["C1", "C7"]},'
        ' {"position": "No", "claims": ["C9"]}]}'
    )
    assert parse_positions(reply, {"C1", "C2"}) == [{"position": "Yes", "claims": ["C1"]}]
    assert parse_positions("garbage", {"C1"}) == []


def test_answer_pipeline_on_demo(tmp_path):
    config = Config()
    config.chunking.target_words = 60
    config.answer.passages = 3
    config.answer.max_claims_per_book = 2
    books, chunks = ingest(DEMO_BOOKS, config.chunking)
    llm = fake_llm()
    claims, _ = extract_claims(books, chunks, llm, load_prompt("extract_claims"), config.claims)
    embedder = HashingEmbedder(256)
    chunk_index = SearchIndex.build(contextual_text(books, chunks), embedder)
    claim_index = SearchIndex.build(claim_search_text(books, chunks, claims), embedder)

    questions = ["Do short naps improve memory?", "Is the siesta a social institution?"]
    evidence = retrieve(
        questions,
        books,
        chunks,
        claims,
        chunk_index,
        claim_index,
        embedder,
        OverlapReranker(),
        config,
    )
    for q in range(2):
        ev = evidence.for_question(q)
        assert 0 < (ev.kind == PASSAGE).sum() <= 3
        claim_books = claims.book[ev.row[ev.kind == CLAIM]]
        assert np.bincount(claim_books).max() <= 2

    corpus = Corpus(books, chunks, claims)
    answers = answer_questions(questions, evidence, corpus, llm, config)
    for a in answers:
        assert a.revised  # the fake's first draft has an uncited sentence
        assert a.problems == []
        assert UNCITED not in a.text
        assert set(a.formats) == {"article", "slides"}
        assert a.format_problems == {"article": [], "slides": []}
        assert a.positions

    paths = write_answer(answers[0], corpus, config, tmp_path)
    assert [p.name for p in paths] == ["answer.md", "article.md", "slides.md"]
    body = paths[0].read_text()
    assert body.startswith("# Do short naps improve memory?")
    assert "## Sources" in body and "Quote:" in body
