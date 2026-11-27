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

from bookqa.config import EmbeddingConfig, RerankerConfig
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

    return {"dtype": torch.float16} if device in {"mps", "cuda"} else {}


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


class Qwen3Reranker:
    """Qwen3-Reranker, following the usage on its model card.

    The model is a causal LM asked whether the document meets the query; the
    score is the probability of "yes" versus "no" for the next token.
    """

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        'the Query and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    MAX_LENGTH = 8192

    def __init__(self, config: RerankerConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = pick_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(config.model, **half_precision_kwargs(self.device))
            .to(self.device)
            .eval()
        )
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")

    def _format(self, query: str, document: str) -> str:
        body = f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: {document}"
        return self.PREFIX + body + self.SUFFIX

    def score(self, queries: list[str], documents: list[str]) -> np.ndarray:
        import torch

        prompts = [self._format(q, d) for q, d in zip(queries, documents, strict=True)]
        # Sorting by length puts similar lengths in one batch, so less padding is computed.
        order = np.argsort([len(p) for p in prompts], kind="stable")
        out = np.zeros(len(prompts), dtype=np.float32)
        size = self.config.batch_size
        with torch.no_grad():
            for start in range(0, len(order), size):
                idx = order[start : start + size]
                inputs = self.tokenizer(
                    [prompts[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits[:, -1, :]
                pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
                out[idx] = torch.softmax(pair, dim=1)[:, 1].cpu().numpy()
        return out

    def close(self) -> None:
        del self.model
        free_memory(self.device)
