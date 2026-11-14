"""Columnar tables: the core data model.

Every entity in the pipeline (books, chunks, claims) is stored as a *struct of
arrays* rather than a list of objects. A table is a frozen dataclass whose
fields are equal-length columns. Row ``i`` of every column describes entity
``i``, and an entity's id *is* its row index.

Why this shape:

* Most work is a pass over one or two columns, not over whole entities.
  Filtering chunks by book reads only the ``book`` column, which is a
  contiguous ``int32`` array, instead of chasing pointers through objects.
* Ids being row indices means joins are array indexing (``chunks.book[ids]``)
  rather than dictionary lookups.
* Columns map one-to-one onto Parquet columns and NumPy arrays, so saving,
  loading and handing data to a model need no conversion layer.
* Embedding matrices live *beside* a table, not inside it: row ``i`` of
  ``chunk_vectors.npy`` is the vector for chunk ``i``.

Numeric columns are ``np.ndarray``. Text columns are ``list[str]`` because
NumPy has no good variable-length string type and text is only touched at the
edges (tokenising, prompting, printing).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclasses.dataclass(frozen=True)
class Table:
    """Base class for columnar tables. Subclasses only declare columns."""

    def __post_init__(self) -> None:
        lengths = {f.name: len(getattr(self, f.name)) for f in dataclasses.fields(self)}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"{type(self).__name__} columns differ in length: {lengths}")

    def __len__(self) -> int:
        first = dataclasses.fields(self)[0]
        return len(getattr(self, first.name))

    @classmethod
    def empty(cls) -> Self:
        cols = {}
        for f in dataclasses.fields(cls):
            dtype = _numeric_dtype(f)
            cols[f.name] = [] if dtype is None else np.empty(0, dtype=dtype)
        return cls(**cols)

    def take(self, rows: np.ndarray | list[int]) -> Self:
        """Return a new table holding only ``rows``, in that order."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = {}
        for f in dataclasses.fields(self):
            col = getattr(self, f.name)
            cols[f.name] = col[rows] if isinstance(col, np.ndarray) else [col[i] for i in rows]
        return type(self)(**cols)

    @classmethod
    def concat(cls, parts: list[Self]) -> Self:
        if not parts:
            return cls.empty()
        cols = {}
        for f in dataclasses.fields(cls):
            values = [getattr(p, f.name) for p in parts]
            if isinstance(values[0], np.ndarray):
                cols[f.name] = np.concatenate(values)
            else:
                cols[f.name] = [v for part in values for v in part]
        return cls(**cols)

    def row(self, i: int) -> dict:
        """One row as a dict. For printing and debugging, not hot loops."""
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)[i]
            out[f.name] = value.item() if isinstance(value, np.generic) else value
        return out

    def save(self, path: Path) -> None:
        arrays = {f.name: pa.array(getattr(self, f.name)) for f in dataclasses.fields(self)}
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(arrays), path)

    @classmethod
    def load(cls, path: Path) -> Self:
        table = pq.read_table(path)
        cols = {}
        for f in dataclasses.fields(cls):
            column = table.column(f.name)
            dtype = _numeric_dtype(f)
            if dtype is None:
                cols[f.name] = column.to_pylist()
            else:
                cols[f.name] = column.to_numpy().astype(dtype, copy=False)
        return cls(**cols)


def _numeric_dtype(field: dataclasses.Field) -> np.dtype | None:
    """Read the NumPy dtype from a field's metadata, or None for text columns."""
    return field.metadata.get("dtype")


def col(dtype: type) -> dataclasses.Field:
    """Declare a numeric column with a fixed NumPy dtype."""
    return dataclasses.field(metadata={"dtype": np.dtype(dtype)})


def strings() -> dataclasses.Field:
    """Declare a text column."""
    return dataclasses.field(metadata={})


@dataclasses.dataclass(frozen=True)
class Books(Table):
    """One row per book. ``book`` ids elsewhere index into this table."""

    title: list[str] = strings()
    author: list[str] = strings()
    source: list[str] = strings()  # file name the book was read from


@dataclasses.dataclass(frozen=True)
class Chunks(Table):
    """One row per retrievable passage.

    Rows are sorted by (book, chapter, section, position), so all chunks of a
    book form one contiguous run of rows and neighbours are adjacent rows.
    """

    book: np.ndarray = col(np.int32)
    chapter: np.ndarray = col(np.int32)  # 0-based chapter number within the book
    section: np.ndarray = col(np.int32)  # 0-based section number within the chapter
    chapter_title: list[str] = strings()
    section_title: list[str] = strings()
    text: list[str] = strings()


@dataclasses.dataclass(frozen=True)
class Claims(Table):
    """One row per claim extracted from a chunk.

    ``evidence`` is a small integer code into the project's evidence-type list
    (for example 0 = study, 3 = opinion). Codes keep the column compact and make
    "only claims backed by studies" a single vectorised comparison.
    """

    chunk: np.ndarray = col(np.int32)  # chunk the claim came from
    book: np.ndarray = col(np.int32)  # denormalised from chunk, for fast filtering
    evidence: np.ndarray = col(np.int8)
    claim: list[str] = strings()  # the claim, restated in one sentence
    speaker: list[str] = strings()  # who makes it: the author, or someone they cite
    stance: list[str] = strings()  # the speaker's position, in a few words
    quote: list[str] = strings()  # verbatim supporting passage from the chunk
