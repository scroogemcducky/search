"""BM25 keyword search over flat arrays.

Embeddings are good at meaning but weak at exact terms: names, numbers, rare
words. BM25 is the reverse. Hybrid search runs both.

The index is an *inverted index* stored as three flat arrays, in the same
layout as a compressed sparse column (CSC) matrix:

    term_start[t] : term_start[t + 1]   slice of postings for term t
    posting_doc[j]                      a document containing that term
    posting_tf[j]                       how often the term occurs there

Scoring a query term is then one slice and a handful of vectorised
operations over that slice. There's no per-document Python loop and no
dictionary of lists, and the whole index saves as a single ``.npz`` file.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np

RIGHT_QUOTE = chr(0x2019)  # typographic apostrophe, common in EPUBs
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his
    how i if in into is it its itself just me more most my myself no nor not now of off
    on once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through
    to too under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves""".split()  # noqa: SIM905
)


def stem(word: str) -> str:
    """A deliberately tiny stemmer: folds common English inflections together.

    "naps", "napping" and "napped" become "nap". It is crude, but for keyword
    recall that's enough, and it has no dependencies or surprises.
    """
    for suffix, keep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if suffix == "s" and word.endswith(("ss", "us", "is")):
            break  # "process", "status", "analysis" are not plurals
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: len(word) - len(suffix)] + keep
            if keep == "" and len(word) > 3 and word[-1] == word[-2] and word[-1] not in "ls":
                word = word[:-1]  # napp -> nap
            break
    return word


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(text.lower().replace(RIGHT_QUOTE, "'"))
    return [stem(w.split("'")[0]) for w in words if w not in STOPWORDS]


@dataclasses.dataclass(frozen=True)
class BM25:
    vocab: dict[str, int]
    term_start: np.ndarray  # int64[n_terms + 1]
    posting_doc: np.ndarray  # int32[n_postings], sorted by term then doc
    posting_tf: np.ndarray  # float32[n_postings]
    doc_len: np.ndarray  # float32[n_docs]
    k1: float = 1.2
    b: float = 0.75

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @classmethod
    def build(cls, texts: list[str]) -> BM25:
        vocab: dict[str, int] = {}
        doc_ids: list[int] = []
        term_ids: list[int] = []
        doc_len = np.zeros(len(texts), dtype=np.float32)
        for doc, text in enumerate(texts):
            tokens = tokenize(text)
            doc_len[doc] = len(tokens)
            for token in tokens:
                term_ids.append(vocab.setdefault(token, len(vocab)))
                doc_ids.append(doc)

        # One (term, doc) pair per token occurrence. Sorting by term then doc
        # and collapsing runs gives postings with term frequencies.
        terms = np.array(term_ids, dtype=np.int64)
        docs = np.array(doc_ids, dtype=np.int64)
        pairs = np.unique(terms * max(len(texts), 1) + docs, return_counts=True)
        key, tf = pairs
        posting_term = key // max(len(texts), 1)
        posting_doc = (key % max(len(texts), 1)).astype(np.int32)
        term_start = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.add.at(term_start, posting_term + 1, 1)
        return cls(
            vocab=vocab,
            term_start=np.cumsum(term_start),
            posting_doc=posting_doc,
            posting_tf=tf.astype(np.float32),
            doc_len=doc_len,
        )

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query`` (float32[n_docs])."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if self.n_docs == 0:
            return out
        avg_len = max(float(self.doc_len.mean()), 1e-9)
        for term in set(tokenize(query)):
            t = self.vocab.get(term)
            if t is None:
                continue
            lo, hi = self.term_start[t], self.term_start[t + 1]
            docs = self.posting_doc[lo:hi]
            tf = self.posting_tf[lo:hi]
            df = hi - lo
            idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5))
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[docs] / avg_len)
            out[docs] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def save(self, path: Path) -> None:
        terms = np.array(sorted(self.vocab, key=self.vocab.__getitem__), dtype=object)
        np.savez_compressed(
            path,
            terms=terms.astype(str),
            term_start=self.term_start,
            posting_doc=self.posting_doc,
            posting_tf=self.posting_tf,
            doc_len=self.doc_len,
        )

    @classmethod
    def load(cls, path: Path) -> BM25:
        with np.load(path, allow_pickle=False) as z:
            terms = z["terms"].tolist()
            return cls(
                vocab={t: i for i, t in enumerate(terms)},
                term_start=z["term_start"],
                posting_doc=z["posting_doc"],
                posting_tf=z["posting_tf"],
                doc_len=z["doc_len"],
            )
