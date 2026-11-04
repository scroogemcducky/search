"""Turn evidence into verified, cited answers and derived formats.

Steps 4 to 6 of the pipeline, each run over *all* questions as one batch:

    group claims into positions   -> one LLM call per question, sent together
    write the cited answer        -> one call per question, sent together
    verify, then revise failures  -> rule checks; one more call per failed answer
    derive formats                -> one call per question per format

The answer is written once and every format (article, slides) is derived
from it. Verification runs on each format too, so a rewrite that drops or
invents a citation is caught.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bookqa.claims import evidence_label
from bookqa.config import Config
from bookqa.llm import LLM, parse_json_object
from bookqa.prompts import load_prompt
from bookqa.retrieve import CLAIM, PASSAGE, Evidence
from bookqa.tables import Books, Chunks, Claims
from bookqa.verify import Problem, cited_labels, verify


@dataclasses.dataclass
class Corpus:
    """The tables an answer can cite, bundled for convenience."""

    books: Books
    chunks: Chunks
    claims: Claims


@dataclasses.dataclass
class Answer:
    question: str
    evidence: Evidence  # this question's rows only
    positions: list[dict]  # [{"position": str, "claims": [label, ...]}]
    text: str
    problems: list[Problem]
    revised: bool = False
    formats: dict[str, str] = dataclasses.field(default_factory=dict)
    format_problems: dict[str, list[Problem]] = dataclasses.field(default_factory=dict)


# --- Rendering evidence for prompts --------------------------------------------


def location(corpus: Corpus, chunk: int) -> str:
    c = corpus.chunks
    parts = [f'"{corpus.books.title[c.book[chunk]]}"']
    if corpus.books.author[c.book[chunk]]:
        parts.append(f"by {corpus.books.author[c.book[chunk]]}")
    if c.chapter_title[chunk]:
        parts.append(f"chapter: {c.chapter_title[chunk]}")
    if c.section_title[chunk]:
        parts.append(f"section: {c.section_title[chunk]}")
    return ", ".join(parts)


def claim_line(corpus: Corpus, ev: Evidence, i: int, config: Config) -> str:
    claims, row = corpus.claims, ev.row[i]
    kind = evidence_label(int(claims.evidence[row]), config.claims)
    book = corpus.books.title[claims.book[row]]
    return (
        f"[{ev.label[i]}] ({book}; speaker: {claims.speaker[row]}; evidence: {kind}) "
        f"{claims.claim[row]}"
    )


def passage_block(corpus: Corpus, ev: Evidence, i: int) -> str:
    chunk = ev.row[i]
    return f"[{ev.label[i]}] {location(corpus, chunk)}\n{corpus.chunks.text[chunk]}"


def _rows(ev: Evidence, kind: int) -> np.ndarray:
    return np.flatnonzero(ev.kind == kind)


def sources_block(corpus: Corpus, ev: Evidence, config: Config) -> str:
    claims = "\n".join(claim_line(corpus, ev, i, config) for i in _rows(ev, CLAIM))
    passages = "\n\n".join(passage_block(corpus, ev, i) for i in _rows(ev, PASSAGE))
    return f"Claims:\n{claims or '(none)'}\n\nPassages:\n\n{passages or '(none)'}"


def positions_block(corpus: Corpus, ev: Evidence, positions: list[dict], config: Config) -> str:
    index = {ev.label[i]: i for i in _rows(ev, CLAIM)}
    out, placed = [], set()
    for n, pos in enumerate(positions, start=1):
        lines = [f"Position {n}: {pos['position']}"]
        for label in pos["claims"]:
            lines.append("  " + claim_line(corpus, ev, index[label], config))
            placed.add(label)
        out.append("\n".join(lines))
    rest = [label for label in index if label not in placed]
    if rest and not positions:  # grouping failed; show claims ungrouped rather than lose them
        out.append("\n".join(claim_line(corpus, ev, index[label], config) for label in rest))
    return "\n\n".join(out) or "(no relevant claims found)"


def parse_positions(reply: str, labels: set[str]) -> list[dict]:
    """Keep positions whose claim labels exist; drop the rest quietly."""
    try:
        found = parse_json_object(reply).get("positions", [])
    except ValueError:
        return []
    out = []
    for pos in found if isinstance(found, list) else []:
        if not isinstance(pos, dict):
            continue
        claims = [c for c in pos.get("claims", []) if isinstance(c, str) and c in labels]
        if claims and str(pos.get("position", "")).strip():
            out.append({"position": str(pos["position"]).strip(), "claims": claims})
    return out


# --- The batch ------------------------------------------------------------------


def _checker(corpus: Corpus, ev: Evidence, config: Config) -> Callable[[str], list[Problem]]:
    research_codes = [
        i
        for i, t in enumerate(config.claims.evidence_types)
        if t in config.answer.research_evidence
    ]
    claim_rows = _rows(ev, CLAIM)
    research = {
        ev.label[i] for i in claim_rows if int(corpus.claims.evidence[ev.row[i]]) in research_codes
    }
    available = set(ev.label)
    return lambda text: verify(text, available, research, config.answer.research_words)


def _complete(llm: LLM, rendered: list[tuple[str, str]], json_mode: bool = False) -> list[str]:
    """Send a batch of rendered prompts. All prompts of one kind share a system part."""
    if not rendered:
        return []
    return llm.complete_many(rendered[0][0], [user for _, user in rendered], json_mode)


def answer_questions(
    questions: list[str],
    evidence: Evidence,
    corpus: Corpus,
    llm: LLM,
    config: Config,
    prompts_dir: Path | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> list[Answer]:
    per_question = [evidence.for_question(q) for q in range(len(questions))]
    checkers = [_checker(corpus, ev, config) for ev in per_question]

    def claims_text(ev: Evidence) -> str:
        return "\n".join(claim_line(corpus, ev, i, config) for i in _rows(ev, CLAIM))

    def passages_text(ev: Evidence) -> str:
        return "\n\n".join(passage_block(corpus, ev, i) for i in _rows(ev, PASSAGE)) or "(none)"

    # 1. Group each question's claims into positions.
    log("Grouping claims into positions")
    group = load_prompt("group_claims", prompts_dir)
    with_claims = [q for q, ev in enumerate(per_question) if len(_rows(ev, CLAIM))]
    replies = _complete(
        llm,
        [
            group.render(question=questions[q], claims=claims_text(per_question[q]))
            for q in with_claims
        ],
        json_mode=True,
    )
    positions: list[list[dict]] = [[] for _ in questions]
    for q, reply in zip(with_claims, replies, strict=True):
        ev = per_question[q]
        positions[q] = parse_positions(reply, {ev.label[i] for i in _rows(ev, CLAIM)})

    # 2. Write one cited answer per question, and verify it.
    log("Writing answers")
    prompt = load_prompt("answer", prompts_dir)
    texts = _complete(
        llm,
        [
            prompt.render(
                question=questions[q],
                positions=positions_block(corpus, ev, positions[q], config),
                passages=passages_text(ev),
            )
            for q, ev in enumerate(per_question)
        ],
    )
    answers = [
        Answer(questions[q], per_question[q], positions[q], texts[q], checkers[q](texts[q]))
        for q in range(len(questions))
    ]

    # 3. Give each failed answer one chance to fix the listed problems.
    failed = [q for q, a in enumerate(answers) if a.problems]
    if failed and config.answer.revise:
        log(f"Revising {len(failed)} answer(s) that failed verification")
        revise = load_prompt("revise", prompts_dir)
        revised = _complete(
            llm,
            [
                revise.render(
                    question=questions[q],
                    sources=sources_block(corpus, per_question[q], config),
                    draft=answers[q].text,
                    problems="\n".join(f"- {p.describe()}" for p in answers[q].problems),
                )
                for q in failed
            ],
        )
        for q, text in zip(failed, revised, strict=True):
            problems = checkers[q](text)
            if len(problems) < len(answers[q].problems):  # keep the better of the two
                answers[q].text, answers[q].problems, answers[q].revised = text, problems, True

    # 4. Derive each output format from the answer, and verify it too.
    for name in config.answer.formats:
        log(f"Writing {name} versions")
        fmt = load_prompt(f"format_{name}", prompts_dir)
        outputs = _complete(llm, [fmt.render(question=a.question, answer=a.text) for a in answers])
        for q, text in enumerate(outputs):
            answers[q].formats[name] = text
            answers[q].format_problems[name] = checkers[q](text)
    return answers


# --- Output ---------------------------------------------------------------------


def sources_markdown(answer: Answer, corpus: Corpus, config: Config, text: str) -> str:
    """A "Sources" section listing every label cited in ``text``."""
    ev = answer.evidence
    index = {label: i for i, label in enumerate(ev.label)}
    lines = ["## Sources", ""]
    for label in cited_labels(text):
        if label not in index:
            continue
        i = index[label]
        if ev.kind[i] == CLAIM:
            row = ev.row[i]
            claims = corpus.claims
            kind = evidence_label(int(claims.evidence[row]), config.claims)
            lines.append(
                f"- **[{label}]** {location(corpus, claims.chunk[row])}. "
                f"{claims.speaker[row]} ({kind}): {claims.claim[row]} "
                f"Quote: “{claims.quote[row]}”"
            )
        else:
            chunk = ev.row[i]
            excerpt = corpus.chunks.text[chunk]
            excerpt = excerpt if len(excerpt) <= 300 else excerpt[:300].rsplit(" ", 1)[0] + "…"
            lines.append(f"- **[{label}]** {location(corpus, chunk)}. “{excerpt}”")
    return "\n".join(lines) + "\n"


def problems_markdown(problems: list[Problem]) -> str:
    if not problems:
        return ""
    items = "\n".join(f"> - {p.describe()}" for p in problems)
    return f"> **Verification found {len(problems)} problem(s):**\n{items}\n\n"


def write_answer(answer: Answer, corpus: Corpus, config: Config, directory: Path) -> list[Path]:
    """Write the answer and each format as Markdown, each with its own sources."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    docs = {"answer": (answer.text, answer.problems)}
    docs.update({n: (t, answer.format_problems[n]) for n, t in answer.formats.items()})
    for name, (text, problems) in docs.items():
        heading = "" if name != "answer" else f"# {answer.question}\n\n"
        body = (
            f"{heading}{problems_markdown(problems)}{text.strip()}\n\n"
            f"{sources_markdown(answer, corpus, config, text)}"
        )
        path = directory / f"{name}.md"
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written
