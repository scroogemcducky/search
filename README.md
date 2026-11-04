# bookqa

Answers questions across a small collection of books, with every statement cited back to an exact quote. It's built for comparison questions, where several books agree or disagree, and it turns each answer into a web article and a slideshow script.

Everything runs locally on an Apple Silicon Mac. The code is generic: the subject of a corpus lives in its project folder, not in this repository.

## How it works

1. **Split** each book by its own chapters and sections into chunks of about 250 words.
2. **Extract claims** from every chapter with an LLM. Each claim gets a speaker, stance and evidence type (study, clinical observation, anecdote or opinion). Its supporting quote is checked against the book and stored in the book's exact wording.
3. **Search** chunks and claims with keywords (BM25) and embeddings (Qwen3-Embedding) together, then rerank with Qwen3-Reranker.
4. **Balance and group:** take the best claims from every book, not just the top-ranked one, and group them into positions that agree or disagree.
5. **Answer** once with citations, then rewrite that answer as an article and a slideshow script.
6. **Verify** every sentence: it must cite a source it was given, and research wording must cite a research-type claim. Failures get one revision. Anything still wrong is flagged at the top of the file.

[`docs/design.md`](docs/design.md) explains each design choice. The pull requests that built this project (#1 to #8) walk through it one step at a time.

## Try it without any models

`--offline` swaps every model for a simple rule-based stand-in. The output is nonsense, but it shows each stage working in about a second:

```bash
uv sync
```

```bash
cp -r examples/demo /tmp/demo
```

```bash
uv run bookqa build --offline -p /tmp/demo
```

```bash
uv run bookqa ask --offline -p /tmp/demo "How long should a nap be?"
```

Then open `/tmp/demo/output/how-long-should-a-nap-be/answer.md`.

## Set up on a Mac (M-series, 24 GB)

The embedder and reranker run in PyTorch on the Mac's GPU. The LLM runs in [Ollama](https://ollama.com), though any OpenAI-compatible server works.

```bash
brew install uv ollama
```

```bash
uv sync --extra local
```

Answer prompts run to about 8,000 tokens, and Ollama may cut prompts off at a smaller default context length without warning. Start the server with a larger context:

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

In another terminal:

```bash
ollama pull qwen3:14b
```

The first real run downloads Qwen3-Embedding-4B and Qwen3-Reranker-4B from Hugging Face, about 8 GB each. Only one PyTorch model is loaded at a time, and each is released after use, so peak memory stays under about 18 GB alongside Ollama.

## Use it

Create a project and add books (`.txt`, `.md` or `.epub`):

```bash
uv run bookqa init ~/my-project --prompts
```

```bash
cp ~/books/*.epub ~/my-project/books/
```

Build it. Claim extraction makes one LLM call per four chunks, which can take hours for 20 books. It's safe to stop and restart, because finished calls are cached:

```bash
uv run bookqa build -p ~/my-project
```

Ask questions, one at a time or from a file:

```bash
uv run bookqa ask -p ~/my-project "Do the authors agree about X?"
```

```bash
uv run bookqa ask -p ~/my-project --file questions.txt
```

Each question gets a folder in `output/` with `answer.md`, `article.md` and `slides.md`, each ending in a Sources section.

Other commands:

| command | what it does |
|---|---|
| `bookqa status` | shows what's built, plus chunk and claim counts per book |
| `bookqa claims --evidence study` | lists extracted claims, filtered by book or evidence type |
| `bookqa eval` | compares retrieval methods and models on your own questions |
| `bookqa ingest` / `extract` / `index` | runs one stage of `build` |

## Measure before you tune

Write 30 to 50 real questions in `questions.jsonl`. For each one, name the passages that answer it by book file and a phrase from the passage:

```json
{"id": "q1", "question": "...", "relevant": [{"book": "smith.epub", "contains": "a phrase from the passage"}]}
```

`bookqa eval` reports recall, MRR, nDCG and cross-book coverage for keyword, vector, hybrid and reranked search. Add `[[eval.embedders]]` entries to `bookqa.toml` to compare models, such as the 8B against the default 4B.

## Configure

`bookqa init` writes a commented `bookqa.toml` showing every setting. The main ones:

- **`[embedding]`:** the model, its dimensions and its query instruction. Set `backend = "openai"` and a `base_url` to use Ollama, LM Studio or Gemini instead of PyTorch.
- **`[llm]`:** `base_url` and `model` for any OpenAI-compatible server.
- **`[claims]`:** the evidence types and the guide that explains them to the model.
- **`prompts/`:** edit any prompt for this project. `init --prompts` copies the defaults there.

## Develop

```bash
uv run pytest
```

```bash
uv run ruff check .
```

```bash
uv run ruff format .
```

The tests use the offline stand-ins, so they need no models or network.

## Status

The pipeline and all model-free code are tested end to end. The real model code (`local_models.py` and the HTTP clients) was written against the model cards and the OpenAI API format, but **it hasn't been run yet**, because the environment this was built in couldn't download models. The first real `bookqa build` is the test. If something breaks there, it will be in `local_models.py`, `llm.py` or `embed.py`.
