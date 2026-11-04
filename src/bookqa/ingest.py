"""Read books and split them into chunks along their own structure.

The pipeline has two passes, each turning one table into the next:

    files --parse--> Paragraphs --pack--> Chunks

Parsing knows about file formats and nothing else: it emits one row per
paragraph, tagged with its chapter and section. Packing knows nothing about
file formats: it walks paragraph rows in order and groups neighbours from the
same section into chunks of roughly ``target_words``. Keeping the passes apart
means a new format (PDF, HTML) only needs a parser that emits paragraph rows.
"""

from __future__ import annotations

import dataclasses
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from bookqa.config import ChunkingConfig
from bookqa.tables import Books, Chunks, Table, col, strings

SUPPORTED = {".txt", ".md", ".markdown", ".epub"}


@dataclasses.dataclass(frozen=True)
class Paragraphs(Table):
    """Intermediate table: one row per paragraph of one book, in reading order."""

    chapter: np.ndarray = col(np.int32)
    section: np.ndarray = col(np.int32)
    chapter_title: list[str] = strings()
    section_title: list[str] = strings()
    text: list[str] = strings()


@dataclasses.dataclass
class ParsedBook:
    title: str
    author: str
    paragraphs: Paragraphs


class _ParagraphBuilder:
    """Accumulates paragraph rows while a parser walks a document."""

    def __init__(self) -> None:
        self.chapter = -1
        self.section = 0
        self.chapter_title = ""
        self.section_title = ""
        self.rows: dict[str, list] = {f.name: [] for f in dataclasses.fields(Paragraphs)}

    def start_chapter(self, title: str) -> None:
        self.chapter += 1
        self.section = 0
        self.chapter_title = title.strip()
        self.section_title = ""

    def start_section(self, title: str) -> None:
        if self.chapter < 0:
            self.start_chapter("")
        # The first section of a chapter keeps number 0 unless text came before it.
        if self._has_text_in_current_section():
            self.section += 1
        self.section_title = title.strip()

    def add(self, text: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        if self.chapter < 0:
            self.start_chapter("")
        self.rows["chapter"].append(self.chapter)
        self.rows["section"].append(self.section)
        self.rows["chapter_title"].append(self.chapter_title)
        self.rows["section_title"].append(self.section_title)
        self.rows["text"].append(text)

    def _has_text_in_current_section(self) -> bool:
        return bool(
            self.rows["chapter"]
            and self.rows["chapter"][-1] == self.chapter
            and self.rows["section"][-1] == self.section
        )

    def build(self) -> Paragraphs:
        return Paragraphs(
            chapter=np.array(self.rows["chapter"], dtype=np.int32),
            section=np.array(self.rows["section"], dtype=np.int32),
            chapter_title=self.rows["chapter_title"],
            section_title=self.rows["section_title"],
            text=self.rows["text"],
        )


# --- Plain text and Markdown -------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TXT_CHAPTER = re.compile(
    r"^\s*(chapter|part|book)\s+([0-9]+|[ivxlcdm]+|[a-z-]+)\b.*$", re.IGNORECASE
)
_META = re.compile(r"^(title|author)\s*:\s*(.+)$", re.IGNORECASE)


def parse_text(raw: str, fallback_title: str) -> ParsedBook:
    """Parse Markdown or plain text.

    Optional ``Title:`` and ``Author:`` lines at the very top set metadata.
    Markdown headings set structure: if the top heading level appears once,
    before any text, it is the book title; the next level down marks chapters
    and the one below that marks sections. Plain text without Markdown
    headings falls back to lines such as "Chapter 3" or "CHAPTER IV".
    """
    lines = raw.replace("\r\n", "\n").split("\n")
    meta: dict[str, str] = {}
    while lines and (m := _META.match(lines[0].strip())):
        meta[m.group(1).lower()] = m.group(2).strip()
        lines.pop(0)

    headings = [
        (i, len(m.group(1)), m.group(2))
        for i, ln in enumerate(lines)
        if (m := _MD_HEADING.match(ln))
    ]
    levels = sorted({level for _, level, _ in headings})
    title_line = None
    if levels:
        top = [h for h in headings if h[1] == levels[0]]
        text_before = any(ln.strip() for ln in lines[: top[0][0]])
        if len(top) == 1 and not text_before and len(levels) > 1:
            title_line = top[0][0]
            meta.setdefault("title", top[0][2])
            levels = levels[1:]
    chapter_level = levels[0] if levels else None
    section_level = levels[1] if len(levels) > 1 else None

    builder = _ParagraphBuilder()
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            builder.add(" ".join(buffer))
            buffer.clear()

    for i, line in enumerate(lines):
        if i == title_line:
            continue
        m = _MD_HEADING.match(line)
        if m and chapter_level is not None:
            flush()
            level = len(m.group(1))
            if level == chapter_level:
                builder.start_chapter(m.group(2))
            elif level == section_level:
                builder.start_section(m.group(2))
            else:  # deeper headings become part of the text
                builder.add(m.group(2))
            continue
        if chapter_level is None and _TXT_CHAPTER.match(line) and len(line.split()) <= 12:
            flush()
            builder.start_chapter(line)
            continue
        if line.strip():
            buffer.append(line.strip())
        else:
            flush()
    flush()
    return ParsedBook(meta.get("title", fallback_title), meta.get("author", ""), builder.build())


# --- EPUB ---------------------------------------------------------------------

_NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}
_BLOCK_TAGS = {"p", "li", "blockquote", "pre", "dd", "dt", "td", "div"}
_SKIP_TAGS = {"script", "style", "nav", "head"}


class _XhtmlWalker(HTMLParser):
    """Turns one EPUB content document into builder calls.

    ``h1``/``h2`` open a section within the current chapter (the chapter itself
    is opened per spine document). Deeper headings are kept as text.
    """

    def __init__(self, builder: _ParagraphBuilder) -> None:
        super().__init__(convert_charrefs=True)
        self.builder = builder
        self.text: list[str] = []
        self.heading: list[str] | None = None
        self.skip = 0
        self.first_heading: str | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self.skip += 1
        elif tag in {"h1", "h2", "h3"}:
            self._flush()
            self.heading = []
        elif tag in _BLOCK_TAGS or tag == "br":
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag in {"h1", "h2", "h3"} and self.heading is not None:
            title = " ".join("".join(self.heading).split())
            self.heading = None
            if self.first_heading is None:
                self.first_heading = title
                self.builder.chapter_title = self.builder.chapter_title or title
                self.builder.section_title = ""
            elif title:
                self.builder.start_section(title)
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        (self.heading if self.heading is not None else self.text).append(data)

    def _flush(self) -> None:
        if self.text:
            self.builder.add("".join(self.text))
            self.text = []

    def close(self) -> None:
        super().close()
        self._flush()


def parse_epub(path: Path) -> ParsedBook:
    with zipfile.ZipFile(path) as z:
        container = ElementTree.fromstring(z.read("META-INF/container.xml"))
        rootfile = container.find(".//c:rootfile", _NS)
        if rootfile is None:
            raise ValueError(f"{path.name}: no rootfile in container.xml")
        opf_path = rootfile.attrib["full-path"]
        opf = ElementTree.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)

        title = opf.findtext(".//dc:title", default=path.stem, namespaces=_NS).strip()
        author = opf.findtext(".//dc:creator", default="", namespaces=_NS).strip()
        manifest = {
            item.attrib["id"]: item.attrib["href"]
            for item in opf.findall(".//opf:manifest/opf:item", _NS)
        }
        builder = _ParagraphBuilder()
        for ref in opf.findall(".//opf:spine/opf:itemref", _NS):
            if ref.attrib.get("linear") == "no":
                continue
            href = manifest.get(ref.attrib["idref"])
            if href is None:
                continue
            doc = z.read(posixpath.normpath(posixpath.join(base, href))).decode("utf-8", "replace")
            before = len(builder.rows["text"])
            builder.start_chapter("")
            walker = _XhtmlWalker(builder)
            walker.feed(doc)
            walker.close()
            if len(builder.rows["text"]) == before:  # empty document, e.g. a cover image
                builder.chapter -= 1
    return ParsedBook(title, author, builder.build())


def parse_book(path: Path) -> ParsedBook:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return parse_epub(path)
    if suffix in SUPPORTED:
        return parse_text(path.read_text(encoding="utf-8", errors="replace"), path.stem)
    raise ValueError(f"unsupported book format: {path.name}")


# --- Packing paragraphs into chunks -------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _split_long(text: str, max_words: int) -> list[str]:
    """Split one over-long paragraph at sentence boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _SENTENCE_END.split(text):
        n = len(sentence.split())
        if current and count + n > max_words:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def pack(paragraphs: Paragraphs, book: int, config: ChunkingConfig) -> Chunks:
    """Group consecutive paragraphs of one section into chunks.

    A chunk never crosses a section boundary, so its chapter and section
    labels are always exact. A chunk is closed once it reaches
    ``target_words``, or before a paragraph would push it past ``max_words``.
    """
    rows: dict[str, list] = {f.name: [] for f in dataclasses.fields(Chunks)}
    current: list[str] = []
    words = 0

    def close(i: int) -> None:
        nonlocal current, words
        if not current:
            return
        rows["book"].append(book)
        rows["chapter"].append(paragraphs.chapter[i])
        rows["section"].append(paragraphs.section[i])
        rows["chapter_title"].append(paragraphs.chapter_title[i])
        rows["section_title"].append(paragraphs.section_title[i])
        rows["text"].append("\n\n".join(current))
        current, words = [], 0

    n = len(paragraphs)
    for i in range(n):
        new_section = i > 0 and (
            paragraphs.chapter[i] != paragraphs.chapter[i - 1]
            or paragraphs.section[i] != paragraphs.section[i - 1]
        )
        if new_section:
            close(i - 1)
        for piece in _split_long(paragraphs.text[i], config.max_words):
            n_words = len(piece.split())
            if current and words + n_words > config.max_words:
                close(i)
            current.append(piece)
            words += n_words
            if words >= config.target_words:
                close(i)
    if n:
        close(n - 1)

    return Chunks(
        book=np.array(rows["book"], dtype=np.int32),
        chapter=np.array(rows["chapter"], dtype=np.int32),
        section=np.array(rows["section"], dtype=np.int32),
        chapter_title=rows["chapter_title"],
        section_title=rows["section_title"],
        text=rows["text"],
    )


def book_files(books_dir: Path) -> list[Path]:
    """Book files in a stable order, so book ids don't change between runs."""
    if not books_dir.is_dir():
        raise FileNotFoundError(f"books directory not found: {books_dir}")
    return sorted(p for p in books_dir.iterdir() if p.suffix.lower() in SUPPORTED)


def ingest(books_dir: Path, config: ChunkingConfig) -> tuple[Books, Chunks]:
    titles, authors, sources, parts = [], [], [], []
    for book_id, path in enumerate(book_files(books_dir)):
        parsed = parse_book(path)
        titles.append(parsed.title)
        authors.append(parsed.author)
        sources.append(path.name)
        parts.append(pack(parsed.paragraphs, book_id, config))
    return Books(title=titles, author=authors, source=sources), Chunks.concat(parts)


def contextual_text(books: Books, chunks: Chunks, rows: np.ndarray | None = None) -> list[str]:
    """Chunk text prefixed with where it comes from.

    This is what gets embedded and shown to models. The prefix lets a passage
    that says "this approach" still match a question naming the chapter's topic.
    """
    rows = np.arange(len(chunks)) if rows is None else rows
    out = []
    for i in rows:
        where = [books.title[chunks.book[i]], chunks.chapter_title[i], chunks.section_title[i]]
        out.append(" > ".join(w for w in where if w) + "\n\n" + chunks.text[i])
    return out
