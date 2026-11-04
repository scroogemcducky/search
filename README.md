# search

A pipeline that answers questions across a collection of books, with citations.

## How it works

1. **Split** each book into chapters and sections, and keep book, chapter and section context on every chunk.
2. **Extract claims** from each chapter. Each claim records its author, stance, evidence type, supporting passage and location.
3. **Search** claims and passages with both keyword and embedding search, then rerank the results.
4. **Group** the claims for each question into agreements and disagreements.
5. **Answer** with a cited answer, which can then be reused for articles and slideshow scripts.
6. **Verify** that every claim in the output traces back to a source passage.

## Models

- Embeddings: Qwen3-Embedding-8B, compared against Qwen3-Embedding-4B and gemini-embedding-001
- Reranker: Qwen3-Reranker-4B

## Configuration

Questions, prompts, taxonomies and model instructions belong in a private config, not in the code. The code only deals with books, claims and questions.

## Status

Early work in progress.
