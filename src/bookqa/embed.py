"""Text embedding backends.

Every backend turns a batch of strings into one ``float32`` matrix with
unit-length rows, so cosine similarity is a plain dot product and a whole
corpus can be scored with a single matrix-vector multiply.

Backends:

* ``sentence-transformers``: runs a model locally (Apple Silicon GPU via MPS).
* ``openai``: any server with an OpenAI-compatible ``/embeddings`` endpoint,
  such as Ollama, LM Studio or Gemini's compatibility endpoint.
* ``hashing``: no model at all. Words are hashed into buckets. Useless for
  meaning, but deterministic and instant, so tests and dry runs use it.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Iterator
from typing import Protocol

import numpy as np

from bookqa.config import EmbeddingConfig
from bookqa.httpjson import auth_headers, post_json


class Embedder(Protocol):
    def fingerprint(self) -> str:
        """Identifies the model and settings, to detect stale saved vectors."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def close(self) -> None:
        """Release model memory. Matters on a 24 GB machine running several models."""
        ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def batches(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def qwen_query(instruction: str, query: str) -> str:
    """Qwen3-Embedding's query format. Documents are embedded without it."""
    return f"Instruct: {instruction}\nQuery: {query}" if instruction else query


def _matryoshka(matrix: np.ndarray, dimensions: int) -> np.ndarray:
    """Keep the first ``dimensions`` values and re-normalise.

    Qwen3 embeddings are trained so that a prefix of the vector is itself a
    usable embedding. Fewer dimensions mean a smaller matrix and faster search
    for a small loss in quality.
    """
    if dimensions and dimensions < matrix.shape[1]:
        matrix = matrix[:, :dimensions]
    return normalize(matrix)


class HashingEmbedder:
    """Signed feature hashing of words and word pairs. For tests only."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions or 256

    def fingerprint(self) -> str:
        return f"hashing:{self.dimensions}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"\w+", text.lower())
            for feature in words + [a + " " + b for a, b in itertools.pairwise(words)]:
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest())
                out[row, h % self.dimensions] += 1.0 if (h >> 32) & 1 else -1.0
        return normalize(out)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def close(self) -> None:
        pass


class OpenAIEmbedder:
    """Calls ``POST {base_url}/embeddings``."""

    def __init__(self, config: EmbeddingConfig):
        if not config.base_url:
            raise ValueError("embedding.base_url is required for the openai backend")
        self.config = config
        self.url = config.base_url.rstrip("/") + "/embeddings"
        self.headers = auth_headers(config.api_key_env)

    def fingerprint(self) -> str:
        return f"openai:{self.config.base_url}:{self.config.model}:{self.config.dimensions}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        parts = []
        for batch in batches(texts, self.config.batch_size):
            reply = post_json(
                self.url, {"model": self.config.model, "input": batch}, self.headers, 600
            )
            rows = sorted(reply["data"], key=lambda d: d["index"])
            parts.append(np.array([r["embedding"] for r in rows], dtype=np.float32))
        return _matryoshka(np.concatenate(parts), self.config.dimensions)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([qwen_query(self.config.query_instruction, t) for t in texts])

    def close(self) -> None:
        pass


def make_embedder(config: EmbeddingConfig) -> Embedder:
    if config.backend == "hashing":
        return HashingEmbedder(config.dimensions)
    if config.backend == "openai":
        return OpenAIEmbedder(config)
    if config.backend == "sentence-transformers":
        from bookqa.local_models import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(config)
    raise ValueError(f"unknown embedding backend: {config.backend}")
