"""Finding quoted text inside source passages.

Every claim carries a quote that must appear in its source chunk. Models
often copy text *almost* exactly: curly quotes become straight, a dash
changes, whitespace shifts, a word is dropped. ``locate`` matches through
those differences and returns the exact span of the source, so the stored
quote is always the book's own wording.
"""

from __future__ import annotations

import difflib

# Typographic characters folded to their plain equivalents. Written as code
# points because several are invisible or look identical to ASCII in an editor.
_FOLD = str.maketrans(
    {
        chr(0x2018): "'",  # left single quote
        chr(0x2019): "'",  # right single quote / apostrophe
        chr(0x201C): '"',  # left double quote
        chr(0x201D): '"',  # right double quote
        chr(0x2013): "-",  # en dash
        chr(0x2014): "-",  # em dash
        chr(0x00A0): " ",  # no-break space
    }
)


def _normalise(text: str) -> tuple[str, list[int]]:
    """Lowercase, fold punctuation variants and collapse whitespace.

    Returns the normalised text and, for each of its characters, the index of
    the source character it came from, so a match maps back to the original.
    """
    out: list[str] = []
    origin: list[int] = []
    space = False
    for i, ch in enumerate(text.translate(_FOLD).lower()):
        if ch.isspace():
            space = bool(out)
            continue
        if space:
            out.append(" ")
            origin.append(i)
            space = False
        out.append(ch)
        origin.append(i)
    return "".join(out), origin


def locate(quote: str, source: str, min_ratio: float = 0.85) -> tuple[int, int] | None:
    """Character span of ``quote`` in ``source``, or None if it isn't there.

    An exact match after normalisation is tried first. Otherwise the best
    near match is accepted if at least ``min_ratio`` of the quote's characters
    line up with the source.
    """
    q, _ = _normalise(quote.strip().strip("\"'"))
    s, origin = _normalise(source)
    if not q or not s:
        return None
    start = s.find(q)
    if start >= 0:
        return origin[start], origin[start + len(q) - 1] + 1

    matcher = difflib.SequenceMatcher(None, s, q, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return None
    # Anchor on the longest shared run, then take a quote-sized window around it.
    anchor = max(blocks, key=lambda b: b.size)
    lo = max(0, anchor.a - anchor.b)
    hi = min(len(s), lo + len(q))
    window = difflib.SequenceMatcher(None, s[lo:hi], q, autojunk=False)
    matched = sum(b.size for b in window.get_matching_blocks())
    if matched / len(q) < min_ratio:
        return None
    # Widen to whole words so the quote doesn't start or end mid-word.
    while lo > 0 and s[lo - 1] != " ":
        lo -= 1
    while hi < len(s) and s[hi] != " ":
        hi += 1
    return origin[lo], origin[hi - 1] + 1
