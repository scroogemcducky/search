import numpy as np
import pytest

from bookqa.tables import Chunks, Claims


def make_chunks() -> Chunks:
    return Chunks(
        book=np.array([0, 0, 1], dtype=np.int32),
        chapter=np.array([0, 1, 0], dtype=np.int32),
        section=np.zeros(3, dtype=np.int32),
        chapter_title=["One", "Two", "Intro"],
        section_title=["", "", ""],
        text=["alpha", "beta", "gamma"],
    )


def test_columns_must_have_equal_length():
    with pytest.raises(ValueError, match="differ in length"):
        Chunks(
            book=np.zeros(2, dtype=np.int32),
            chapter=np.zeros(3, dtype=np.int32),
            section=np.zeros(3, dtype=np.int32),
            chapter_title=["", "", ""],
            section_title=["", "", ""],
            text=["", "", ""],
        )


def test_take_and_concat():
    chunks = make_chunks()
    picked = chunks.take([2, 0])
    assert picked.text == ["gamma", "alpha"]
    assert picked.book.tolist() == [1, 0]
    both = Chunks.concat([chunks, picked])
    assert len(both) == 5
    assert both.book.dtype == np.int32


def test_save_load_roundtrip(tmp_path):
    chunks = make_chunks()
    chunks.save(tmp_path / "chunks.parquet")
    loaded = Chunks.load(tmp_path / "chunks.parquet")
    assert loaded.text == chunks.text
    assert loaded.book.dtype == np.int32
    np.testing.assert_array_equal(loaded.chapter, chunks.chapter)


def test_empty_table_roundtrip(tmp_path):
    claims = Claims.empty()
    assert len(claims) == 0
    claims.save(tmp_path / "claims.parquet")
    assert len(Claims.load(tmp_path / "claims.parquet")) == 0
