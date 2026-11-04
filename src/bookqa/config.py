"""Project configuration.

A *project* is a directory holding one corpus and everything specific to it:

    my-project/
      bookqa.toml        settings (this module)
      books/             input books (.txt, .md, .epub)
      prompts/           optional prompt overrides
      questions.jsonl    optional evaluation questions
      .bookqa/           generated data (tables, vectors, caches)

The code knows only generic concepts (books, chunks, claims, questions).
Anything about a particular subject, such as the questions to ask, the prompt
wording, the evidence taxonomy or model instructions, lives in the project
directory, so the same code serves any corpus.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any

CONFIG_NAME = "bookqa.toml"


@dataclasses.dataclass
class ChunkingConfig:
    target_words: int = 250  # aim for chunks about this long
    max_words: int = 450  # never exceed this, splitting long paragraphs if needed


@dataclasses.dataclass
class EmbeddingConfig:
    # "sentence-transformers" (local), "openai" (any OpenAI-compatible server),
    # or "hashing" (no model at all; for tests and dry runs).
    backend: str = "sentence-transformers"
    # 4B rather than 8B: in float16 the 8B needs ~16 GB, too much next to a
    # reranker and an LLM in 24 GB of unified memory. `bookqa eval` can compare them.
    model: str = "Qwen/Qwen3-Embedding-4B"
    # Qwen3 vectors can be truncated (Matryoshka). 0 keeps the full width.
    dimensions: int = 1024
    # Prepended to queries only. Qwen reports 1-5% gains from a task instruction.
    query_instruction: str = "Given a question, retrieve book passages that answer it"
    batch_size: int = 8
    device: str = ""  # "" picks mps / cuda / cpu automatically
    base_url: str = ""  # for backend = "openai"
    api_key_env: str = ""  # name of the env var holding the API key, if any


@dataclasses.dataclass
class RerankerConfig:
    backend: str = "qwen3"  # "qwen3", "overlap" (lexical, no model) or "none"
    model: str = "Qwen/Qwen3-Reranker-4B"
    instruction: str = "Given a question, judge whether the passage helps answer it"
    candidates: int = 50  # how many hybrid-search hits are reranked
    batch_size: int = 4
    device: str = ""


@dataclasses.dataclass
class LLMConfig:
    # "openai" talks to any OpenAI-compatible chat endpoint: Ollama, LM Studio,
    # mlx_lm.server, llama.cpp server, or a hosted API.
    backend: str = "openai"
    # Ollama's default. Start it with OLLAMA_CONTEXT_LENGTH=16384: answer prompts
    # run to ~8k tokens and a smaller context silently truncates them.
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3:14b"
    api_key_env: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_s: float = 600.0
    # Requests sent at once. Local servers batch concurrent requests on the GPU
    # (Ollama: OLLAMA_NUM_PARALLEL), so 2-4 raises throughput; 1 is safest.
    parallel: int = 2
    # Some models think aloud inside <think> tags before answering. Qwen3 skips
    # that when the prompt ends with /no_think; set "" to allow thinking.
    prompt_suffix: str = " /no_think"


@dataclasses.dataclass
class ClaimsConfig:
    # Order matters: a claim's evidence code is its index in this list.
    evidence_types: list[str] = dataclasses.field(
        default_factory=lambda: ["study", "clinical observation", "anecdote", "opinion"]
    )
    # Shown to the model to explain the evidence types. Rewrite it with the list.
    evidence_guide: str = (
        "- study: a systematic study, trial, survey or dataset is cited\n"
        "- clinical observation: a professional's observation of patients or clients\n"
        "- anecdote: a single story or personal experience\n"
        "- opinion: a belief, interpretation or recommendation without cited evidence"
    )
    chunks_per_call: int = 4  # adjacent chunks sent to the LLM in one extraction call


@dataclasses.dataclass
class AnswerConfig:
    passages: int = 12  # passages given to the answer prompt
    claims: int = 40  # claims retrieved before grouping
    max_claims_per_book: int = 8  # keeps one prolific book from dominating
    # Reranker score below which a claim or passage counts as off-topic (0 to 1).
    min_relevance: float = 0.1
    # Output formats; each needs a prompt file named format_<name>.md.
    formats: list[str] = dataclasses.field(default_factory=lambda: ["article", "slides"])
    # Verification flags a sentence that uses one of these words but cites
    # only claims whose evidence type is not in research_evidence.
    research_words: list[str] = dataclasses.field(
        default_factory=lambda: ["study", "studies", "trial", "research", "survey", "data"]
    )
    research_evidence: list[str] = dataclasses.field(default_factory=lambda: ["study"])
    revise: bool = True  # ask the model once to fix an answer that fails verification


@dataclasses.dataclass
class EvalConfig:
    # Embedders to compare with the main [embedding] settings. Each entry
    # overrides some [embedding] keys, for example:
    #   [[eval.embedders]]
    #   model = "Qwen/Qwen3-Embedding-8B"
    embedders: list[dict] = dataclasses.field(default_factory=list)
    ks: list[int] = dataclasses.field(default_factory=lambda: [5, 10, 50])


@dataclasses.dataclass
class Config:
    books_dir: str = "books"
    prompts_dir: str = "prompts"
    questions_file: str = "questions.jsonl"
    chunking: ChunkingConfig = dataclasses.field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = dataclasses.field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = dataclasses.field(default_factory=RerankerConfig)
    llm: LLMConfig = dataclasses.field(default_factory=LLMConfig)
    claims: ClaimsConfig = dataclasses.field(default_factory=ClaimsConfig)
    answer: AnswerConfig = dataclasses.field(default_factory=AnswerConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)


def _build[T](cls: type[T], data: dict[str, Any], where: str) -> T:
    """Construct a config dataclass from a TOML table, rejecting unknown keys."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown setting '{where}{key}' in {CONFIG_NAME}")
        sub = known[key].default_factory  # type: ignore[misc]
        if isinstance(value, dict) and dataclasses.is_dataclass(sub):
            value = _build(sub, value, f"{where}{key}.")
        kwargs[key] = value
    return cls(**kwargs)


def embedding_variant(base: EmbeddingConfig, overrides: dict[str, Any]) -> EmbeddingConfig:
    """``base`` with some keys replaced. Unknown keys are rejected, as in ``load_config``."""
    known = {f.name for f in dataclasses.fields(EmbeddingConfig)}
    for key in overrides:
        if key not in known:
            raise ValueError(f"unknown setting 'eval.embedders.{key}' in {CONFIG_NAME}")
    return dataclasses.replace(base, **overrides)


def load_config(path: Path) -> Config:
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        return _build(Config, tomllib.load(f), "")


class Project:
    """Paths and settings for one project directory."""

    def __init__(self, root: Path, config: Config | None = None):
        self.root = root.resolve()
        self.config = config if config is not None else load_config(self.root / CONFIG_NAME)

    @property
    def books_dir(self) -> Path:
        return self.root / self.config.books_dir

    @property
    def prompts_dir(self) -> Path:
        return self.root / self.config.prompts_dir

    @property
    def questions_file(self) -> Path:
        return self.root / self.config.questions_file

    @property
    def data_dir(self) -> Path:
        return self.root / ".bookqa"

    def data(self, name: str) -> Path:
        return self.data_dir / name
