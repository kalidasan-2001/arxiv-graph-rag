"""Programmatically generated minimal test PDFs.

No checked-in binary fixtures (avoids repo bloat/licensing questions) and
no live network -- `pymupdf` itself can write simple PDFs, so every test
PDF is generated fresh and deterministically at test time.
"""

import pymupdf


def make_pdf_bytes(pages: list[str], *, fontsize: float = 11) -> bytes:
    """Build a PDF with one page per string in `pages`.

    Each string's lines are placed one below the other on that page.
    """

    doc = pymupdf.open()
    try:
        for page_text in pages:
            page = doc.new_page()
            y = 72.0
            for line in page_text.split("\n"):
                if line:
                    page.insert_text((72, y), line, fontsize=fontsize)
                y += fontsize * 1.6
        return doc.tobytes()
    finally:
        doc.close()


def make_empty_page_pdf_bytes() -> bytes:
    """A two-page PDF where the second page has no text at all."""

    doc = pymupdf.open()
    try:
        page1 = doc.new_page()
        page1.insert_text((72, 72), "Some content on page one.", fontsize=11)
        doc.new_page()  # page two: intentionally blank
        return doc.tobytes()
    finally:
        doc.close()


def make_encrypted_pdf_bytes(*, user_password: str = "secret") -> bytes:
    """A single-page PDF protected with a user password."""

    doc = pymupdf.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), "Confidential content.", fontsize=11)
        return doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=user_password,
            owner_pw=user_password,
        )
    finally:
        doc.close()


_SCIENTIFIC_PAPER_PAGE_1 = """A Great Paper Title
Alice Author, Bob Author

Abstract
This paper presents a deterministic approach to testing scientific PDF
parsers using synthetic documents.

1 Introduction
Scientific papers follow common structural conventions that a parser can
exploit. This introduction continues onto the following page."""

_SCIENTIFIC_PAPER_PAGE_2 = """continued introduction text describing the problem in more detail.

2 Related Work
Prior work on document parsing has focused on general-purpose text
extraction rather than scientific structure.

3 Methodology
We implement a deterministic section-boundary algorithm based on heading
pattern matching."""

_SCIENTIFIC_PAPER_PAGE_3 = """4 Experiments
We evaluate our approach on a corpus of synthetic test documents.

5 Results
The parser correctly recovers section boundaries in all test cases.

6 Limitations
The heading vocabulary is necessarily incomplete.

7 Conclusion
We presented a conservative, deterministic scientific PDF parser.

References
[1] Some Author. A Related Paper. 2023.
[2] Another Author. Another Related Paper. 2022."""


def make_scientific_paper_pdf_bytes() -> bytes:
    """A synthetic three-page "scientific paper" with a full set of
    conventional section headings, for higher-level parser tests."""

    return make_pdf_bytes(
        [_SCIENTIFIC_PAPER_PAGE_1, _SCIENTIFIC_PAPER_PAGE_2, _SCIENTIFIC_PAPER_PAGE_3]
    )
