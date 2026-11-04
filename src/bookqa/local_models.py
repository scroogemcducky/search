"""Models that run on this machine through PyTorch.

Imported lazily so the rest of the package (and the tests) work without
PyTorch installed. Install with ``uv sync --extra local``.

On Apple Silicon the GPU is reached through PyTorch's MPS backend. Models are
loaded in float16, which halves memory: Qwen3-Embedding-4B takes about 8 GB
instead of 16 GB, which matters with 24 GB of unified memory shared by the
CPU, GPU and any LLM server.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

from bookqa.config import EmbeddingConfig
from bookqa.embed import normalize, qwen_query


def pick_device(requested: str = "") -> str:
    import torch

    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def half_precision_kwargs(device: str) -> dict[str, Any]:
    import torch

    return {"torch_dtype": torch.float16} if device in {"mps", "cuda"} else {}


def free_memory(device: str) -> None:
    import torch

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = pick_device(config.device)
        self.model = SentenceTransformer(
            config.model,
            device=self.device,
            truncate_dim=config.dimensions or None,
            model_kwargs=half_precision_kwargs(self.device),
            tokenizer_kwargs={"padding_side": "left"},  # Qwen3 pools the last token
        )

    def fingerprint(self) -> str:
        return f"sentence-transformers:{self.config.model}:{self.config.dimensions}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.config.batch_size,
        )
        return normalize(vectors)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([qwen_query(self.config.query_instruction, t) for t in texts])

    def close(self) -> None:
        del self.model
        free_memory(self.device)
