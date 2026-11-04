"""Check that an answer's statements trace back to its sources.

This is step 6 of the pipeline. It's cheap because claim extraction already
did the hard part: every claim's quote is guaranteed to be exact text from its
chunk. What's left is checking the answer's *use* of the sources:

* every statement carries at least one citation,
* every citation names a source the answer was actually given, and
* a sentence that talks about research cites at least one claim whose
  evidence type is research. That catches an opinion dressed up as a finding.

The checks are deterministic text rules, so they're fast and predictable, and
a failure points at the exact sentence.
"""

from __future__ import annotations

import dataclasses
import re

CITATION = re.compile(r"\[((?:[CP]\d+)(?:\s*[,;]\s*[CP]\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[\"'(]?[A-Z0-9])")
# "Naps help. [C1]" -> "Naps help [C1]." so a citation stays with its sentence.
_TRAILING_CITATIONS = re.compile(r"([.!?])((?:\s*\[[CP][^\]]*\])+)")
_LABEL = re.compile(r"[CP]\d+")


@dataclasses.dataclass(frozen=True)
class Problem:
    kind: str  # "uncited", "unknown_source" or "evidence_mismatch"
    sentence: str
    detail: str = ""

    def describe(self) -> str:
        text = {
            "uncited": "This sentence has no citation",
            "unknown_source": f"This sentence cites {self.detail}, which isn't in the sources",
            "evidence_mismatch": (
                "This sentence describes research but cites no research-type claim"
                f" (it cites {self.detail})"
            ),
        }[self.kind]
        return f'{text}: "{self.sentence}"'


def cited_labels(text: str) -> list[str]:
    """Every source label cited in ``text``, in order of first appearance."""
    seen: dict[str, None] = {}
    for group in CITATION.findall(text):
        for label in _LABEL.findall(group):
            seen.setdefault(label, None)
    return list(seen)


def sentences(markdown: str) -> list[str]:
    """Split Markdown into sentences worth checking.

    Headings, bold labels such as "**On screen:**" and very short fragments
    are skipped. They don't make claims of their own.
    """
    out = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("|") or line == "---":
            continue
        line = re.sub(r"^([-*+]|\d+[.)])\s+", "", line)  # list markers
        line = re.sub(r"^\*\*[^*]{1,40}:\*\*\s*", "", line)  # "**Narration:** ..."
        line = _TRAILING_CITATIONS.sub(r"\2\1", line)
        for sentence in _SENTENCE.split(line):
            sentence = sentence.strip()
            if len(re.sub(CITATION, "", sentence).split()) >= 5:
                out.append(sentence)
    return out


def verify(
    markdown: str,
    available: set[str],
    research_labels: set[str],
    research_words: list[str],
) -> list[Problem]:
    """Problems found in ``markdown``; an empty list means it passed.

    ``available`` holds the labels the answer was given. ``research_labels``
    holds the subset that are claims with a research evidence type.
    """
    words = re.compile(r"\b(" + "|".join(map(re.escape, research_words)) + r")\b", re.I)
    problems = []
    for sentence in sentences(markdown):
        labels = cited_labels(sentence)
        if not labels:
            problems.append(Problem("uncited", sentence))
            continue
        unknown = [label for label in labels if label not in available]
        if unknown:
            problems.append(Problem("unknown_source", sentence, ", ".join(unknown)))
        claims_cited = [label for label in labels if label.startswith("C")]
        cites_only_claims = claims_cited and len(claims_cited) == len(labels)
        if (
            research_words
            and words.search(CITATION.sub("", sentence))
            and cites_only_claims
            and not research_labels.intersection(claims_cited)
        ):
            problems.append(Problem("evidence_mismatch", sentence, ", ".join(labels)))
    return problems
