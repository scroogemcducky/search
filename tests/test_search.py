import numpy as np
import pytest

from bookqa.embed import HashingEmbedder, normalize, qwen_query
from bookqa.index import Hits, SearchIndex, reciprocal_rank_fusion, top_k
from bookqa.keyword import BM25, stem, tokenize

DOCS = [
    "Short naps of twenty minutes improve memory.",
    "Daylight walks lift afternoon alertness.",
    "A cohort study linked long daily napping to heart disease.",
    "The siesta is a social institution in many towns.",
]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("naps", "nap"),
        ("napping", "nap"),
        ("napped", "nap"),
        ("studies", "study"),
        ("process", "process"),
        ("classes", "class"),
        ("walks", "walk"),
        ("fall", "fall"),
    ],
)
def test_stem(word, expected):
    assert stem(word) == expected


def test_tokenize_drops_stopwords_and_folds_case():
    assert tokenize("The NAPS of a Siesta") == ["nap", "siesta"]


def test_bm25_ranks_matching_documents():
    bm25 = BM25.build(DOCS)
    scores = bm25.scores("napping study")
    assert scores.argmax() == 2  # matches both terms
    assert scores[1] == 0 and scores[3] == 0
    assert scores[0] > 0


def test_bm25_roundtrip(tmp_path):
    bm25 = BM25.build(DOCS)
    bm25.save(tmp_path / "k.npz")
    loaded = BM25.load(tmp_path / "k.npz")
    np.testing.assert_allclose(loaded.scores("siesta naps"), bm25.scores("siesta naps"))


def test_bm25_empty_corpus():
    assert BM25.build([]).scores("anything").shape == (0,)


def test_top_k_orders_and_masks():
    scores = np.array([0.1, 0.9, 0.5, 0.9], dtype=np.float32)
    assert top_k(scores, 3).rows.tolist() == [1, 3, 2]
    mask = np.array([True, False, True, False])
    assert top_k(scores, 5, mask).rows.tolist() == [2, 0]


def test_rrf_rewards_agreement():
    a = Hits(np.array([0, 1, 2]), np.zeros(3, dtype=np.float32))
    b = Hits(np.array([2, 1, 3]), np.zeros(3, dtype=np.float32))
    fused = reciprocal_rank_fusion([a, b], n_rows=5, k=4)
    assert fused.rows[0] in (1, 2)
    assert set(fused.rows.tolist()) == {0, 1, 2, 3}


def test_hashing_embedder_is_deterministic_and_unit_length():
    e = HashingEmbedder(64)
    v = e.embed_documents(DOCS)
    assert v.shape == (4, 64) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1, rtol=1e-5)
    np.testing.assert_array_equal(v, e.embed_documents(DOCS))


def test_normalize_handles_zero_rows():
    assert not np.isnan(normalize(np.zeros((2, 3)))).any()


def test_qwen_query_format():
    assert qwen_query("Find it", "q") == "Instruct: Find it\nQuery: q"
    assert qwen_query("", "q") == "q"


def test_index_search_and_roundtrip(tmp_path):
    e = HashingEmbedder(128)
    index = SearchIndex.build(DOCS, e)
    query = "siesta towns"
    hits = index.search(query, e.embed_queries([query])[0], k=2)
    assert hits.rows[0] == 3
    index.save(tmp_path, "chunk")
    loaded = SearchIndex.load(tmp_path, "chunk", e.fingerprint())
    np.testing.assert_array_equal(
        loaded.search(query, e.embed_queries([query])[0], k=2).rows, hits.rows
    )
    with pytest.raises(ValueError, match="rebuild"):
        SearchIndex.load(tmp_path, "chunk", "other-model")
