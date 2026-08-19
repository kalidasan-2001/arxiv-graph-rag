"""Unit tests for `PyMuPDFParser`.

All PDFs are generated in-process via `tests.parsing.pdf_fixtures` -- no
checked-in binaries, no live network.
"""

import pytest

from app.core.exceptions import InvalidPdfError, PdfParseError, UnsupportedPdfError
from app.domain.enums import SectionType
from app.ingestion.parsing.models import ParseWarning
from app.ingestion.parsing.pymupdf_parser import PyMuPDFParser
from tests.parsing.pdf_fixtures import (
    make_empty_page_pdf_bytes,
    make_encrypted_pdf_bytes,
    make_pdf_bytes,
    make_scientific_paper_pdf_bytes,
)

_PAPER_ID = "paper:arxiv:2401.12345"
_VERSION_ID = "paper-version:arxiv:2401.12345:v1"


def _write(tmp_path, data: bytes, name: str = "paper.pdf"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


class TestSinglePagePdf:
    def test_extracts_text_and_page_count(self, tmp_path) -> None:
        path = _write(tmp_path, make_pdf_bytes(["Hello scientific world."]))
        document = PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

        assert document.page_count == 1
        assert "Hello scientific world." in document.full_text
        assert document.paper_id == _PAPER_ID
        assert document.paper_version_id == _VERSION_ID

    def test_parser_identity_is_recorded(self, tmp_path) -> None:
        path = _write(tmp_path, make_pdf_bytes(["Some text."]))
        parser = PyMuPDFParser()
        document = parser.parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

        assert document.parser_name == parser.name == "pymupdf"
        assert document.parser_version == parser.version


class TestMultiPagePdf:
    def test_extracts_all_pages_in_order(self, tmp_path) -> None:
        path = _write(tmp_path, make_pdf_bytes(["Page one text.", "Page two text.", "Page three text."]))
        document = PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

        assert document.page_count == 3
        assert [p.page_number for p in document.pages] == [1, 2, 3]
        assert "Page one" in document.pages[0].text
        assert "Page three" in document.pages[2].text


class TestEmptyPage:
    def test_empty_page_is_flagged_but_does_not_fail_parsing(self, tmp_path) -> None:
        path = _write(tmp_path, make_empty_page_pdf_bytes())
        document = PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

        assert document.page_count == 2
        assert ParseWarning.EMPTY_PAGE_DETECTED in document.warnings


class TestScientificPaperStructure:
    def test_recovers_expected_sections_with_page_provenance(self, tmp_path) -> None:
        path = _write(tmp_path, make_scientific_paper_pdf_bytes())
        document = PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

        types = [s.section_type for s in document.sections]
        assert SectionType.ABSTRACT in types
        assert SectionType.INTRODUCTION in types
        assert SectionType.METHODOLOGY in types
        assert SectionType.RESULTS in types
        assert SectionType.LIMITATIONS in types
        assert SectionType.CONCLUSION in types
        assert SectionType.REFERENCES in types

        introduction = next(s for s in document.sections if s.section_type == SectionType.INTRODUCTION)
        assert introduction.page_start == 1
        assert introduction.page_end == 2  # continues onto page 2 in the fixture

    def test_references_section_preserves_text(self, tmp_path) -> None:
        path = _write(tmp_path, make_scientific_paper_pdf_bytes())
        document = PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

        references = next(s for s in document.sections if s.section_type == SectionType.REFERENCES)
        assert "Some Author" in references.text

    def test_all_sections_carry_stable_ids(self, tmp_path) -> None:
        path = _write(tmp_path, make_scientific_paper_pdf_bytes())
        document = PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

        for section in document.sections:
            assert section.paper_id == _PAPER_ID
            assert section.paper_version_id == _VERSION_ID
            assert section.section_id.startswith("section:")


class TestUnknownSections:
    def test_unrecognized_headings_are_preserved_as_other(self, tmp_path) -> None:
        pdf_text = (
            "Abstract\nAn abstract.\n\n"
            "1 Introduction\nIntro.\n\n"
            "5 Ablation Study\nWe ablate components.\n\n"
            "6 Ethics Statement\nWe considered ethics."
        )
        path = _write(tmp_path, make_pdf_bytes([pdf_text]))
        document = PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

        other_titles = {s.title for s in document.sections if s.section_type == SectionType.OTHER}
        assert "Ablation Study" in other_titles
        assert "Ethics Statement" in other_titles


class TestMalformedOrUnsupportedPdfs:
    def test_non_pdf_bytes_renamed_as_pdf_is_rejected(self, tmp_path) -> None:
        path = _write(tmp_path, b"This is definitely not a PDF file at all.")
        with pytest.raises(UnsupportedPdfError):
            PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

    def test_truncated_pdf_is_rejected(self, tmp_path) -> None:
        # A file cut in half often still has enough intact structure for
        # PyMuPDF to recover gracefully (a deliberate leniency in PDF
        # readers) -- truncating to well before any valid xref/trailer
        # reliably reproduces a genuinely unopenable file instead.
        full = make_scientific_paper_pdf_bytes()
        truncated = full[:200]
        path = _write(tmp_path, truncated)
        with pytest.raises((UnsupportedPdfError, PdfParseError)):
            PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

    def test_encrypted_pdf_is_rejected(self, tmp_path) -> None:
        path = _write(tmp_path, make_encrypted_pdf_bytes())
        with pytest.raises(UnsupportedPdfError):
            PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

    def test_pdf_with_no_extractable_text_raises_parse_error(self, tmp_path) -> None:
        path = _write(tmp_path, make_pdf_bytes([""]))
        with pytest.raises(PdfParseError):
            PyMuPDFParser().parse(path, paper_id=_PAPER_ID, paper_version_id=_VERSION_ID)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises((UnsupportedPdfError, PdfParseError, InvalidPdfError, OSError)):
            PyMuPDFParser().parse(
                tmp_path / "does_not_exist.pdf", paper_id=_PAPER_ID, paper_version_id=_VERSION_ID
            )
