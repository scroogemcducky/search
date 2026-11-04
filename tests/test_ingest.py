import zipfile
from pathlib import Path

import numpy as np

from bookqa.config import ChunkingConfig
from bookqa.ingest import contextual_text, ingest, pack, parse_epub, parse_text

DEMO_BOOKS = Path(__file__).parent.parent / "examples" / "demo" / "books"


def test_markdown_title_chapters_and_sections():
    parsed = parse_text(
        "Author: A. Writer\n\n# My Book\n\n## One\n\n### First\n\nPara one.\n\n"
        "Para two\ncontinues here.\n\n### Second\n\nPara three.\n\n## Two\n\nPara four.\n",
        "fallback",
    )
    assert parsed.title == "My Book"
    assert parsed.author == "A. Writer"
    p = parsed.paragraphs
    assert p.text == ["Para one.", "Para two continues here.", "Para three.", "Para four."]
    assert p.chapter.tolist() == [0, 0, 0, 1]
    assert p.section.tolist() == [0, 0, 1, 0]
    assert p.chapter_title == ["One", "One", "One", "Two"]
    assert p.section_title == ["First", "First", "Second", ""]


def test_plain_text_chapter_lines():
    parsed = parse_text("Intro text.\n\nCHAPTER IV\n\nBody.\n\nChapter 5: Later\n\nMore.\n", "t")
    assert parsed.title == "t"
    assert parsed.paragraphs.chapter_title == ["", "CHAPTER IV", "Chapter 5: Later"]


def test_pack_respects_sections_and_sizes():
    parsed = parse_text(
        "## A\n\n" + "word " * 30 + "\n\n" + "word " * 30 + "\n\n## B\n\nshort\n", "t"
    )
    chunks = pack(parsed.paragraphs, book=7, config=ChunkingConfig(target_words=50, max_words=80))
    assert chunks.book.tolist() == [7, 7]
    assert chunks.chapter.tolist() == [0, 1]
    assert [len(t.split()) for t in chunks.text] == [60, 1]


def test_pack_splits_overlong_paragraph_at_sentences():
    long = " ".join(f"Sentence number {i} is here." for i in range(40))  # 200 words
    parsed = parse_text(long, "t")
    chunks = pack(parsed.paragraphs, 0, ChunkingConfig(target_words=1000, max_words=50))
    assert len(chunks) == 4
    assert all(len(t.split()) <= 50 for t in chunks.text)
    assert " ".join(chunks.text) == long


def _write_epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
        z.writestr(
            "OEBPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata>'
            "<dc:title>Epub Book</dc:title><dc:creator>E. Author</dc:creator></metadata>"
            '<manifest><item id="cover" href="cover.xhtml"/><item id="c1" href="text/c1.xhtml"/>'
            '<item id="c2" href="text/c2.xhtml"/></manifest>'
            '<spine><itemref idref="cover"/><itemref idref="c1"/><itemref idref="c2"/></spine>'
            "</package>",
        )
        z.writestr("OEBPS/cover.xhtml", "<html><body><img src='c.jpg'/></body></html>")
        z.writestr(
            "OEBPS/text/c1.xhtml",
            "<html><head><title>x</title><style>p{}</style></head><body><h1>Start</h1>"
            "<p>First &amp; best.</p><h2>Detail</h2><p>Second.</p></body></html>",
        )
        z.writestr("OEBPS/text/c2.xhtml", "<html><body><h1>End</h1><p>Third.</p></body></html>")


def test_epub(tmp_path):
    path = tmp_path / "b.epub"
    _write_epub(path)
    parsed = parse_epub(path)
    assert (parsed.title, parsed.author) == ("Epub Book", "E. Author")
    p = parsed.paragraphs
    assert p.text == ["First & best.", "Second.", "Third."]
    assert p.chapter.tolist() == [0, 0, 1]
    assert p.section.tolist() == [0, 1, 0]
    assert p.chapter_title == ["Start", "Start", "End"]
    assert p.section_title == ["", "Detail", ""]


def test_ingest_demo_corpus():
    books, chunks = ingest(DEMO_BOOKS, ChunkingConfig())
    assert books.source == sorted(books.source)
    assert books.title == [
        "Against the Afternoon Nap",
        "Sleep Across Cultures",
        "The Case for Napping",
    ]
    assert np.all(np.diff(chunks.book) >= 0)  # rows grouped by book
    assert set(chunks.book.tolist()) == {0, 1, 2}
    first = contextual_text(books, chunks, np.array([0]))[0]
    assert first.startswith("Against the Afternoon Nap > The Night Is What Counts\n\n")
