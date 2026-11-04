"""Command-line interface: ``bookqa <command> [--project DIR]``."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from bookqa.claims import evidence_label
from bookqa.config import CONFIG_NAME, Project
from bookqa.pipeline import Pipeline

CONFIG_TEMPLATE = """\
# bookqa project settings. Every setting is optional; defaults are shown.
# See docs/design.md in the bookqa repository for why each default was chosen.

books_dir = "books"
prompts_dir = "prompts"          # put prompt overrides here (bookqa init --prompts copies them)
questions_file = "questions.jsonl"

[chunking]
target_words = 250
max_words = 450

[embedding]
backend = "sentence-transformers" # or "openai" (Ollama, LM Studio, Gemini...), "hashing"
model = "Qwen/Qwen3-Embedding-4B"
dimensions = 1024
query_instruction = "Given a question, retrieve book passages that answer it"
batch_size = 8

[reranker]
backend = "qwen3"                 # or "none"
model = "Qwen/Qwen3-Reranker-4B"
candidates = 50
batch_size = 4

[llm]
backend = "openai"                # any OpenAI-compatible server
base_url = "http://localhost:11434/v1"  # Ollama's default
model = "qwen3:14b"
parallel = 2

[claims]
evidence_types = ["study", "clinical observation", "anecdote", "opinion"]
chunks_per_call = 4

[answer]
passages = 12
claims = 40
max_claims_per_book = 8
formats = ["article", "slides"]

[eval]
ks = [5, 10, 50]
# Compare other embedders against [embedding] with `bookqa eval`:
# [[eval.embedders]]
# model = "Qwen/Qwen3-Embedding-8B"
"""


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.directory)
    if (root / CONFIG_NAME).exists():
        sys.exit(f"{root / CONFIG_NAME} already exists")
    (root / "books").mkdir(parents=True, exist_ok=True)
    (root / CONFIG_NAME).write_text(CONFIG_TEMPLATE)
    (root / "questions.jsonl").touch()
    if args.prompts:
        target = root / "prompts"
        target.mkdir(exist_ok=True)
        for item in resources.files("bookqa.prompt_templates").iterdir():
            if item.name.endswith(".md"):
                (target / item.name).write_text(item.read_text(encoding="utf-8"))
    print(f"Created {root}. Put .txt, .md or .epub books in {root / 'books'}, then run:")
    print(f"  bookqa build --project {root}")


def cmd_status(pipeline: Pipeline) -> None:
    project = pipeline.project
    manifest_path = project.data("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for stage in ("ingest", "extract", "index"):
        when = manifest.get(stage, {}).get("at", "not run")
        print(f"{stage:8} {when}")
    if project.data("chunks.parquet").exists():
        books, chunks, _ = pipeline.load_chunks()
        counts = np.bincount(chunks.book, minlength=len(books))
        claims = None
        if project.data("claims.parquet").exists():
            claims = pipeline.load_corpus()[0].claims
        claim_counts = (
            np.bincount(claims.book, minlength=len(books)) if claims is not None else None
        )
        print()
        for b in range(len(books)):
            extra = f", {claim_counts[b]} claims" if claim_counts is not None else ""
            print(f"[{b}] {books.title[b]} ({books.source[b]}): {counts[b]} chunks{extra}")


def cmd_claims(pipeline: Pipeline, args: argparse.Namespace) -> None:
    corpus, _ = pipeline.load_corpus()
    claims, config = corpus.claims, pipeline.project.config.claims
    mask = np.ones(len(claims), dtype=bool)
    if args.book is not None:
        mask &= claims.book == args.book
    if args.evidence:
        types = [t.lower() for t in config.evidence_types]
        if args.evidence.lower() not in types:
            sys.exit(f"unknown evidence type; choose from: {', '.join(config.evidence_types)}")
        mask &= claims.evidence == types.index(args.evidence.lower())
    for i in np.flatnonzero(mask)[: args.limit]:
        kind = evidence_label(int(claims.evidence[i]), config)
        book = corpus.books.title[claims.book[i]]
        print(f"[{book}] ({kind}) {claims.speaker[i]}: {claims.claim[i]}")
        print(f"    “{claims.quote[i]}”")


def cmd_ask(pipeline: Pipeline, args: argparse.Namespace) -> None:
    questions = list(args.questions)
    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("{"):
                questions.append(json.loads(line)["question"])
            elif line:
                questions.append(line)
    if not questions:
        sys.exit("give a question, or --file with one question per line")
    pipeline.ask(questions, Path(args.out) if args.out else None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bookqa", description="Cited question answering across a collection of books."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help, description=help)
        p.add_argument("-p", "--project", default=".", help="project directory (default: .)")
        p.add_argument(
            "--offline",
            action="store_true",
            help="use model-free stand-ins for every model, to check the plumbing",
        )
        return p

    init = sub.add_parser("init", help="create a project directory")
    init.add_argument("directory")
    init.add_argument("--prompts", action="store_true", help="copy default prompts to edit")

    add("build", help="run ingest, extract and index")
    add("ingest", help="read books and split them into chunks")
    add("extract", help="extract claims from every chapter (LLM; resumable)")
    add("index", help="embed chunks and claims for search")
    add("status", help="show what has been built")
    add("eval", help="compare retrieval methods on questions.jsonl")

    claims = add("claims", help="list extracted claims")
    claims.add_argument("--book", type=int, help="book number, as shown by status")
    claims.add_argument("--evidence", help="only this evidence type")
    claims.add_argument("--limit", type=int, default=50)

    ask = add("ask", help="answer questions with citations")
    ask.add_argument("questions", nargs="*")
    ask.add_argument("--file", help="questions file: plain lines or questions.jsonl")
    ask.add_argument("--out", help="output directory (default: <project>/output)")

    args = parser.parse_args(argv)
    if args.command == "init":
        cmd_init(args)
        return

    pipeline = Pipeline(Project(Path(args.project)), offline=args.offline)
    try:
        match args.command:
            case "build":
                pipeline.build()
            case "ingest":
                pipeline.ingest()
            case "extract":
                pipeline.extract()
            case "index":
                pipeline.index()
            case "status":
                cmd_status(pipeline)
            case "claims":
                cmd_claims(pipeline, args)
            case "ask":
                cmd_ask(pipeline, args)
            case "eval":
                print(pipeline.eval())
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
