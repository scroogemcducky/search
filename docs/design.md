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
