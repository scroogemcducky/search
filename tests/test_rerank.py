import numpy as np
import pytest

from bookqa.config import RerankerConfig
from bookqa.index import Hits
from bookqa.rerank import NoReranker, OverlapReranker, make_reranker, rerank_many

TEXTS = ["siesta towns", "short naps help memory", "daylight walks", "naps and memory study"]


def hits(rows):
    return Hits(np.array(rows, dtype=np.int64), np.zeros(len(rows), dtype=np.float32))


def test_overlap_reranker_reorders_each_question():
    out = rerank_many(
        OverlapReranker(),
        ["naps memory", "daylight"],
        [hits([0, 2, 1, 3]), hits([1, 2])],
        TEXTS,
        k=2,
    )
    assert out[0].rows.tolist() == [1, 3]
    assert out[1].rows.tolist()[0] == 2


def test_no_reranker_keeps_search_order():
    out = rerank_many(NoReranker(), ["q"], [hits([3, 1, 0])], TEXTS, k=2)
    assert out[0].rows.tolist() == [3, 1]


def test_empty_hits():
    out = rerank_many(NoReranker(), ["q"], [hits([])], TEXTS, k=5)
    assert len(out[0]) == 0


def test_unknown_backend():
    with pytest.raises(ValueError):
        make_reranker(RerankerConfig(backend="nope"))
