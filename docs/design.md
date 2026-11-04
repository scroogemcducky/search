# Design notes

This file explains the main design choices and why they were made. Each pull request adds to it, so reading it top to bottom follows the order the project was built in.

## 1. Data-oriented design

Data-oriented design starts from the data and how it moves, not from objects and their methods. The questions it asks are: what data exists, how much of it is there, and which parts does each step read and write?

For this project the answers are:

- **There isn't much data.** Twenty books of 100,000 words each is about 2 million words, roughly 8,000 chunks of 250 words. With 1,024-dimension float32 vectors, the embedding matrix is about 32 MB. Everything fits in memory many times over, so we don't need a vector database or a server.
- **Each step is a pass over a whole table.** Chunking turns books into rows. Embedding turns a text column into a matrix. Search multiplies one query vector by that matrix. Extraction turns chunk rows into claim rows. Nothing needs to update a single record in place.
- **Model calls are slow and everything else is cheap.** Embedding, reranking and LLM calls take seconds to minutes. Layout matters less for speed than for making batching easy, because batching is what makes those calls fast.

So the core data model in `src/bookqa/tables.py` is a set of **columnar tables** (a *struct of arrays*):

```
Chunks                       chunk_vectors.npy
row  book chapter section text      row  float32[1024]
0    0    0       0       "..."     0    [....]
1    0    0       1       "..."     1    [....]
2    1    0       0       "..."     2    [....]
```

- **An id is a row index.** Chunk 2 is row 2 of every column and row 2 of the vector matrix. A join is plain array indexing, such as `chunks.book[hit_ids]`, with no dictionaries and no object graph.
- **Numeric columns are contiguous NumPy arrays.** Filtering "chunks from book 3" is `chunks.book == 3`, one vectorised comparison over a dense `int32` array.
- **Vectors sit beside the table, not inside it.** Search only needs the matrix, so it never touches text.
- **Tables are immutable and rebuilt per stage.** Each stage reads its input tables and writes new ones to `.bookqa/` as Parquet, a columnar format. That makes every stage restartable, and you can inspect its output with any Parquet tool, such as pandas, DuckDB or `pyarrow`.

The object-oriented alternative, a `Chunk` class with a `.embedding` field and a `.book` reference, is easier to start with. It scatters the data a search needs across thousands of heap objects, though, and every batch call to a model then starts by gathering fields back into arrays.

## 2. Configuration and projects

The code knows only generic concepts: books, chunks, claims and questions. Everything specific to one corpus lives in its project directory: the questions, the prompt wording, the list of evidence types and the model instructions. One install can serve many projects, and the subject of a private project never appears in this repository.

Settings are TOML (`bookqa.toml`), which Python reads with the standard library. Unknown keys are an error rather than being ignored, so a typo such as `modle = ...` fails loudly instead of silently using the default.

## 3. Reading books and chunking

Ingestion makes two passes, each of which turns one table into the next:

```
files --parse--> Paragraphs --pack--> Chunks
```

- **Parse** (`parse_text`, `parse_epub`) knows file formats and nothing else. It emits one row per paragraph, tagged with chapter and section numbers and titles.
- **Pack** (`pack`) knows nothing about formats. It walks paragraph rows in order and groups neighbours from the same section into chunks of about `target_words`.

Supporting a new format, such as PDF, only takes a parser that emits paragraph rows. The chunking rules stay in one place.

**Chunks follow the book's structure.** A chunk never crosses a section boundary, so its chapter and section labels are always exact, and a citation such as "chapter 2, *Keep It Short*" is reliable. Chunks are about 250 words. That's long enough to hold a claim and its evidence, and short enough that a reranker can judge each one cleanly. A paragraph longer than `max_words` is split at sentence boundaries.

**Context is attached when the chunk is used, not stored.** `contextual_text` prefixes each chunk with `Book > Chapter > Section` before embedding or prompting. A passage that only says "this approach" can then still match a question that names the chapter's topic. The stored `text` column stays verbatim, which the citation check needs later.

**Book ids come from file names in sorted order.** Re-running ingestion over the same folder always gives the same ids.

`examples/demo/books` holds three short invented books on afternoon naps that partly disagree. They exist to test the pipeline end to end, especially cross-book comparison.

## 4. Hybrid search

Search runs two methods and merges their results:

- **Embeddings** (`embed.py`) find passages that mean the same thing in different words.
- **BM25 keyword search** (`keyword.py`) finds exact terms such as names, numbers and rare words, which embeddings tend to blur.
- **Reciprocal rank fusion** (`index.py`) merges the two rankings. Each row scores `1 / (60 + rank)` per list. Because RRF uses ranks rather than raw scores, unbounded BM25 scores and cosine similarities combine without any calibration.

### Data layout

**Vectors are one matrix.** All embeddings are unit length and stored as one `float32[n_rows, dim]` array, so scoring every chunk against a query is a single matrix-vector product, `vectors @ q`. At 8,000 chunks and 1,024 dimensions, that's 8 million multiply-adds, under a millisecond. A vector database would add a server, a network hop and approximate results to save time we don't need to save. The matrix is memory-mapped on load, so opening an index costs almost nothing.

**The keyword index is three flat arrays.** The usual inverted index is a dictionary from term to a list of postings. Here it's the same layout as a sparse column matrix:

```
term_start   [0, 3, 5, ...]       postings for term t are term_start[t]:term_start[t+1]
posting_doc  [0, 4, 9, 2, 7, ...] document ids
posting_tf   [1, 2, 1, 1, 3, ...] term counts
```

Scoring one query term is a slice plus a few vectorised operations, with no Python loop over documents. Building it is one `np.unique` over (term, document) pairs. The whole index saves as one `.npz` file.

**Top-k uses `argpartition`.** It finds the best k in linear time, and only those k get sorted.

**One index type serves every table.** `SearchIndex` only needs "row i has text i", so the same code will index chunks now and claims later. That's possible because ids are row indices.

### Models on a 24 GB M4

The default embedder is **Qwen3-Embedding-4B**, not the 8B from the notes. In float16 the 8B needs about 16 GB, which doesn't leave room for a reranker and an LLM in 24 GB of unified memory. The 4B scored within a point of the 8B on retrieval and uses half the memory. Models load one stage at a time and are released after use (`Embedder.close()`), so peak memory stays near the largest single model. `bookqa eval` (PR 7) compares the two on your own books, so you can switch if the 8B earns it.

Vectors are truncated to 1,024 dimensions (Qwen3 supports this "Matryoshka" truncation). That makes the matrix 2.5 times smaller than the full 2,560 for a small loss in quality.

A saved index records which model built it. Loading it with a different model configured raises an error instead of silently comparing incompatible vectors.

A `hashing` embedder needs no model. It's useless for meaning but instant and deterministic, so tests run without downloads.

## 5. Reranking

Hybrid search embeds the question and each passage *separately*. A reranker reads them *together*, which is more accurate but too slow to run on every chunk. So search narrows thousands of chunks to about 50 candidates, and the reranker orders those.

The default is **Qwen3-Reranker-4B**, as your notes suggest. It's a small language model asked "does this document meet the query? yes or no", and the score is the probability it gives to "yes". `local_models.Qwen3Reranker` follows the prompt format from the model card exactly.

**Work is flattened into pairs.** The reranker's unit of work is a (question, passage) pair. When many questions are answered in one run, `rerank_many` flattens every question's candidates into one list of pairs and scores them in shared batches, then scatters the scores back by an `owner` array. The model loads once, and every batch is full regardless of how candidates are spread across questions. This is the data-oriented habit of turning nested loops ("for each question, for each candidate") into one flat array of work items plus an index saying who owns each item.

**Batches are sorted by length.** The model pads every sequence in a batch to the longest one, so grouping prompts of similar length cuts wasted computation.

The `none` and `overlap` backends exist for tests and for measuring how much the reranker helps, which your notes ask for.

## 6. Claim extraction

Your notes call this the biggest design decision, and it's what separates this pipeline from plain retrieval-augmented generation. Top-k passage search answers "where is X discussed?" well. A comparison question such as "do these authors agree about X?" needs **every book's position** on X, not whichever five passages scored highest. So every chapter is turned into claim rows once, up front:

| column | meaning |
|---|---|
| `chunk`, `book` | where the claim came from (`book` is copied from the chunk so filters don't need a join) |
| `evidence` | `int8` code into the project's evidence types: study, clinical observation, anecdote, opinion |
| `claim` | the claim restated as one self-contained sentence |
| `speaker` | the author, or someone the author cites |
| `stance` | the speaker's position, in five words or fewer |
| `quote` | the supporting passage, in the book's exact wording |

### The stage as table passes

```
Chunks --plan--> work items --render--> prompts --LLM--> replies --parse--> Claims
```

- **Plan:** chunks are sorted by book and chapter, so a chapter is a contiguous run of rows. Finding every chapter boundary is one vectorised `np.diff` over the `book` and `chapter` columns. Each run is cut into work items of four adjacent chunks, which gives the model a little context and makes a quarter as many calls.
- **Render:** all prompts are built before any is sent.
- **LLM:** `complete_many` receives the whole batch and sends it in parallel, `llm.parallel` requests at a time. Local servers such as Ollama batch concurrent requests on the GPU.
- **Parse:** each reply becomes claim rows, and each claim is checked as it's parsed.

### Quotes are checked when claims are extracted, not after

Models often copy text *almost* exactly: they straighten curly quotes, change a dash or drop a word. `quotes.locate` normalises both texts while keeping a map back to the original characters, then accepts an exact match or a close match covering at least 85% of the quote. The stored quote is **cut from the book itself**, not taken from the model. A claim whose quote can't be found is dropped and counted. Every stored claim is therefore guaranteed to trace back to real text, which makes the final verification step (step 6 in your notes) simple and exact.

**Evidence types are codes, not strings.** "Only claims backed by studies" is then `claims.evidence == 0`. A type the model invents gets the code `-1` (unclassified) instead of a guess, so an opinion is never silently promoted to a finding.

### Overnight runs and the response cache

Extracting claims from 20 books takes hundreds of LLM calls, possibly hours on a laptop. Every reply is cached in SQLite, keyed by a hash of the model and the prompt. If a run stops, re-running it replays finished calls from disk in seconds and continues where it stopped. Changing a prompt or the model changes the key, so stale replies are never reused.

### Prompts are files

Each prompt is a text file in `src/bookqa/prompt_templates/`, overridable per project by a file of the same name in the project's `prompts/` folder. Placeholders use `$name`, so the JSON examples' curly braces need no escaping. Qwen3 models "think aloud" before answering, so the client adds `/no_think` to prompts by default and strips any `<think>` block from replies.

Tests use a small rule-based fake model (`tests/fakes.py`) that answers each prompt in the right format.

## 7. Answering: balance, group, write, verify

This covers steps 3 to 6 of your notes. Every stage runs over **all questions at once**:

```
questions --embed--> vectors --search--> candidates --rerank--> Evidence
Evidence --group--> positions --answer--> drafts --verify/revise--> answers --format--> article, slides
```

Each model loads once per batch, and every LLM stage is a single `complete_many` call. Answering 50 questions overnight costs the same model loads as answering one.

### One flat `Evidence` table

Each question gets a different number of passages and claims. The object-oriented shape would be `list[QuestionResult]`, each holding lists of hits. Instead there's one table for all questions:

```
question kind     row  score label
0        passage  812  0.97  P1
0        claim    143  0.91  C1
1        passage  77   0.88  P1
```

A question's evidence is `evidence.question == q`, and every per-question step is a filter on one column.

### Balanced claims

Plain top-k lets one book that covers a topic at length crowd out the others, which is the wrong failure for comparison questions. `balance` gives each hit its rank *within its book*, then orders hits by that rank: every book's best claim, then every book's second best, and so on, with a per-book cap. It's all sorting and array arithmetic: one `lexsort` groups hits by book, `np.diff` finds where each book's run starts, and subtracting gives the within-book ranks.

Claims and passages with a reranker score below `min_relevance` are dropped first, so balancing never pads an answer with off-topic material just to represent every book.

### Positions instead of clusters

Your notes say to "group claims by question". Embedding clusters group claims that are about the same thing. That's the opposite of what we need, since "naps improve memory" and "naps don't improve memory" sit close together in embedding space. So the LLM sorts each question's claims into **positions**: groups that agree on the answer. The answer prompt then sees the disagreement already laid out, with each claim's book, speaker and evidence type.

### Verification is rules, not another model

`verify.py` checks each sentence of the answer:

- **uncited:** it states something but cites nothing.
- **unknown source:** it cites a label the answer wasn't given, meaning the model made one up.
- **evidence mismatch:** it uses research wording ("study", "trial", "data") but cites only claims whose evidence type isn't research. This is your "don't present an opinion as a finding" requirement.

Quotes were already confirmed against the source text during extraction, so these checks complete the chain: sentence → citation → claim → exact quote → book, chapter and section. A failed answer gets one revision call listing its problems, and the version with fewer problems is kept. Remaining problems are printed at the top of the output file, never hidden.

### One answer, many formats

The cited answer is written once. The article and slideshow script are rewrites of that answer, not fresh generations from the sources, so they can't drift into new claims. They're verified with the same checker. Each output file ends with its own **Sources** section that maps every label to book, chapter, section and exact quote.
