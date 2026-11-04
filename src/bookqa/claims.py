"""Claim extraction: turn chunks into structured claim rows, once, up front.

This is the step your notes call the biggest design decision. Top-k passage
search answers "where is X discussed?" well, but a comparison question
("do these authors agree about X?") needs every book's position on X, not
whichever passages happened to rank highest. Extracting claims first turns
each book into a table of positions that can be searched, filtered by
evidence type and grouped across books.

The stage is a pipeline of whole-table passes:

    Chunks --plan--> work items --render--> prompts --LLM--> replies --parse--> Claims

Planning groups adjacent chunks of one chapter into a work item, so the
model sees a little context and makes fewer calls. All prompts are built
before any is sent, so the LLM client can run them in parallel and cache
each reply.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from bookqa.config import ClaimsConfig
from bookqa.llm import LLM, parse_json_object
from bookqa.prompts import Prompt
from bookqa.quotes import locate
from bookqa.tables import Books, Chunks, Claims

UNCLASSIFIED = -1  # evidence code for a type the model invented


@dataclasses.dataclass
class ExtractionStats:
    calls: int = 0
    claims: int = 0
    bad_replies: int = 0  # unparseable JSON
    bad_passage: int = 0  # passage number out of range
    quote_not_found: int = 0  # quote doesn't appear in the passage; claim dropped
    unclassified: int = 0  # evidence type not in the configured list

    def summary(self) -> str:
        return (
            f"{self.claims} claims from {self.calls} calls; dropped {self.quote_not_found} "
            f"with unverifiable quotes and {self.bad_passage} with bad passage numbers; "
            f"{self.bad_replies} unparseable replies; {self.unclassified} unclassified"
        )


def plan(chunks: Chunks, per_call: int) -> list[np.ndarray]:
    """Group consecutive chunks of the same chapter into work items.

    Chunks are sorted by book and chapter, so a chapter is a contiguous run of
    rows. Run boundaries are where ``(book, chapter)`` changes, found with one
    vectorised comparison, and each run is cut into pieces of ``per_call``.
    """
    n = len(chunks)
    if n == 0:
        return []
    change = np.flatnonzero((np.diff(chunks.book) != 0) | (np.diff(chunks.chapter) != 0)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    items = []
    for start, end in zip(starts, ends, strict=True):
        for lo in range(start, end, per_call):
            items.append(np.arange(lo, min(lo + per_call, end)))
    return items


def render(
    prompt: Prompt, books: Books, chunks: Chunks, item: np.ndarray, config: ClaimsConfig
) -> tuple[str, str]:
    first = item[0]
    book = chunks.book[first]
    passages = []
    for n, row in enumerate(item, start=1):
        heading = f" ({chunks.section_title[row]})" if chunks.section_title[row] else ""
        passages.append(f"[Passage {n}]{heading}\n{chunks.text[row]}")
    return prompt.render(
        title=books.title[book],
        author=books.author[book] or "an unknown author",
        chapter=chunks.chapter_title[first] or f"chapter {chunks.chapter[first] + 1}",
        evidence_types=", ".join(config.evidence_types),
        evidence_guide=config.evidence_guide,
        passages="\n\n".join(passages),
    )


def parse(
    reply: str,
    item: np.ndarray,
    books: Books,
    chunks: Chunks,
    config: ClaimsConfig,
    stats: ExtractionStats,
) -> dict[str, list]:
    """Turn one reply into claim rows, keeping only claims whose quote checks out."""
    rows: dict[str, list] = {f.name: [] for f in dataclasses.fields(Claims)}
    try:
        found = parse_json_object(reply).get("claims", [])
    except ValueError:
        stats.bad_replies += 1
        return rows
    types = [t.lower() for t in config.evidence_types]
    for c in found if isinstance(found, list) else []:
        if not isinstance(c, dict) or not str(c.get("claim", "")).strip():
            continue
        try:
            passage = int(c.get("passage", 0))
        except (TypeError, ValueError):
            passage = 0
        if not 1 <= passage <= len(item):
            stats.bad_passage += 1
            continue
        chunk = int(item[passage - 1])
        span = locate(str(c.get("quote", "")), chunks.text[chunk])
        if span is None:
            stats.quote_not_found += 1
            continue
        evidence = str(c.get("evidence_type", "")).strip().lower()
        code = types.index(evidence) if evidence in types else UNCLASSIFIED
        stats.unclassified += code == UNCLASSIFIED
        book = int(chunks.book[chunk])
        rows["chunk"].append(chunk)
        rows["book"].append(book)
        rows["evidence"].append(code)
        rows["claim"].append(str(c["claim"]).strip())
        rows["speaker"].append(str(c.get("speaker", "")).strip() or books.author[book])
        rows["stance"].append(str(c.get("stance", "")).strip())
        # Store the book's exact wording, not the model's copy of it.
        rows["quote"].append(chunks.text[chunk][span[0] : span[1]])
    return rows


def extract_claims(
    books: Books,
    chunks: Chunks,
    llm: LLM,
    prompt: Prompt,
    config: ClaimsConfig,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Claims, ExtractionStats]:
    items = plan(chunks, config.chunks_per_call)
    rendered = [render(prompt, books, chunks, item, config) for item in items]
    system = rendered[0][0] if rendered else ""
    replies = llm.complete_many(system, [user for _, user in rendered], True, progress)

    stats = ExtractionStats(calls=len(items))
    parts = []
    for item, reply in zip(items, replies, strict=True):
        rows = parse(reply, item, books, chunks, config, stats)
        parts.append(
            Claims(
                chunk=np.array(rows["chunk"], dtype=np.int32),
                book=np.array(rows["book"], dtype=np.int32),
                evidence=np.array(rows["evidence"], dtype=np.int8),
                claim=rows["claim"],
                speaker=rows["speaker"],
                stance=rows["stance"],
                quote=rows["quote"],
            )
        )
    claims = Claims.concat(parts)
    stats.claims = len(claims)
    return claims, stats


def evidence_label(code: int, config: ClaimsConfig) -> str:
    return config.evidence_types[code] if 0 <= code < len(config.evidence_types) else "unclassified"


def claim_search_text(books: Books, chunks: Chunks, claims: Claims) -> list[str]:
    """What gets embedded for each claim: where it's from, who says it, and the claim."""
    out = []
    for i in range(len(claims)):
        chunk = claims.chunk[i]
        where = " > ".join(
            w for w in (books.title[claims.book[i]], chunks.chapter_title[chunk]) if w
        )
        out.append(f"{where}\n\n{claims.speaker[i]}: {claims.claim[i]} ({claims.stance[i]})")
    return out
