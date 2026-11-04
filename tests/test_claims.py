from pathlib import Path

import numpy as np

from bookqa.claims import UNCLASSIFIED, ExtractionStats, extract_claims, parse, plan
from bookqa.config import ChunkingConfig, ClaimsConfig
from bookqa.ingest import ingest
from bookqa.llm import ResponseCache, ScriptedLLM, parse_json_object
from bookqa.offline import offline_llm
from bookqa.prompts import load_prompt
from bookqa.quotes import locate

DEMO_BOOKS = Path(__file__).parent.parent / "examples" / "demo" / "books"
LQ, RQ, DASH, NBSP = chr(0x201C), chr(0x201D), chr(0x2013), chr(0x00A0)
SOURCE = f"He said {LQ}naps{NBSP}help{RQ} {DASH} and  they do. Short naps improve memory."


def test_locate_exact_through_punctuation_and_spacing():
    span = locate('he said "naps help" - and they do.', SOURCE)
    assert span is not None
    assert SOURCE[span[0] : span[1]] == f"He said {LQ}naps{NBSP}help{RQ} {DASH} and  they do."


def test_locate_near_match_returns_source_wording():
    span = locate("Short naps improves memory", SOURCE)
    assert SOURCE[span[0] : span[1]] == "Short naps improve memory."


def test_locate_rejects_invented_text():
    assert locate("Long naps cause heart disease in everyone", SOURCE) is None
    assert locate("", SOURCE) is None


def test_parse_json_object_ignores_fences():
    assert parse_json_object('Sure!\n```json\n{"a": [1]}\n```') == {"a": [1]}


def test_plan_never_mixes_chapters():
    from bookqa.tables import Chunks

    chunks = Chunks(
        book=np.array([0, 0, 0, 0, 0, 1], dtype=np.int32),
        chapter=np.array([0, 0, 0, 1, 1, 0], dtype=np.int32),
        section=np.zeros(6, dtype=np.int32),
        chapter_title=[""] * 6,
        section_title=[""] * 6,
        text=[""] * 6,
    )
    assert [i.tolist() for i in plan(chunks, 2)] == [[0, 1], [2], [3, 4], [5]]


def test_parse_validates_each_claim():
    books, chunks = ingest(DEMO_BOOKS, ChunkingConfig())
    item = np.array([0])
    real = chunks.text[0].split(".")[0]
    reply = (
        '{"claims": ['
        f'{{"passage": 1, "claim": "ok", "evidence_type": "Study", "quote": "{real}"}},'
        f'{{"passage": 2, "claim": "bad passage", "quote": "{real}"}},'
        '{"passage": 1, "claim": "made up", "quote": "nothing like the text at all"},'
        f'{{"passage": 1, "claim": "odd type", "evidence_type": "vibes", "quote": "{real}"}}'
        "]}"
    )
    stats = ExtractionStats()
    rows = parse(reply, item, books, chunks, ClaimsConfig(), stats)
    assert rows["claim"] == ["ok", "odd type"]
    assert rows["evidence"] == [0, UNCLASSIFIED]
    assert rows["speaker"] == [books.author[0]] * 2
    assert (stats.bad_passage, stats.quote_not_found, stats.unclassified) == (1, 1, 1)
    assert parse("not json", item, books, chunks, ClaimsConfig(), stats)["claim"] == []
    assert stats.bad_replies == 1


def test_extract_claims_end_to_end_with_cache(tmp_path):
    books, chunks = ingest(DEMO_BOOKS, ChunkingConfig(target_words=60, max_words=120))
    config = ClaimsConfig(chunks_per_call=2)
    prompt = load_prompt("extract_claims")
    llm = offline_llm(ResponseCache(tmp_path / "cache.sqlite"))
    claims, stats = extract_claims(books, chunks, llm, prompt, config)
    assert stats.claims == len(claims) > 0
    assert stats.quote_not_found == 0
    assert set(claims.book.tolist()) == {0, 1, 2}
    for i in range(len(claims)):
        assert claims.quote[i] in chunks.text[claims.chunk[i]]  # exact source wording
        assert claims.book[i] == chunks.book[claims.chunk[i]]
    assert (claims.evidence == 0).any() and (claims.evidence == 3).any()

    # A second run is served entirely from the cache.
    again = offline_llm(ResponseCache(tmp_path / "cache.sqlite"))
    claims2, _ = extract_claims(books, chunks, again, prompt, config)
    assert again.calls == 0
    assert claims2.claim == claims.claim


def test_prompt_override(tmp_path):
    (tmp_path / "extract_claims.md").write_text("sys $title\n<<<USER>>>\nuser $author")
    system, user = load_prompt("extract_claims", tmp_path).render(title="T", author="A")
    assert (system, user) == ("sys T", "user A")


def test_think_tags_are_stripped():
    llm = ScriptedLLM(lambda s, u: "<think>hmm</think>\nanswer")
    assert llm.complete("s", "u") == "answer"
