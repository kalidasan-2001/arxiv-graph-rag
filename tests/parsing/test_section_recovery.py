"""Unit tests for `app.ingestion.parsing.section_recovery`.

Pure functions operating on `ParsedPage` lists -- no PDF library, no
network -- always run.
"""

from app.domain.enums import SectionType
from app.ingestion.parsing.models import ParsedPage, ParseWarning
from app.ingestion.parsing.section_recovery import (
    compute_quality_warnings,
    detect_heading,
    recover_sections,
)


class TestHeadingDetection:
    def test_numbered_heading(self) -> None:
        assert detect_heading("1 Introduction") == ("Introduction", SectionType.INTRODUCTION)

    def test_numbered_heading_with_period(self) -> None:
        assert detect_heading("1. Introduction") == ("Introduction", SectionType.INTRODUCTION)

    def test_roman_numeral_heading(self) -> None:
        assert detect_heading("I. INTRODUCTION") == ("INTRODUCTION", SectionType.INTRODUCTION)

    def test_bare_heading_methods(self) -> None:
        assert detect_heading("Methods") == ("Methods", SectionType.METHODOLOGY)

    def test_numbered_methodology_variant(self) -> None:
        assert detect_heading("3 Methodology") == ("Methodology", SectionType.METHODOLOGY)

    def test_experimental_setup(self) -> None:
        assert detect_heading("Experimental Setup") == ("Experimental Setup", SectionType.EXPERIMENTS)

    def test_results(self) -> None:
        assert detect_heading("Results") == ("Results", SectionType.RESULTS)

    def test_limitations(self) -> None:
        assert detect_heading("Limitations") == ("Limitations", SectionType.LIMITATIONS)

    def test_conclusion(self) -> None:
        assert detect_heading("Conclusion") == ("Conclusion", SectionType.CONCLUSION)

    def test_references(self) -> None:
        assert detect_heading("References") == ("References", SectionType.REFERENCES)

    def test_bibliography_maps_to_references(self) -> None:
        assert detect_heading("Bibliography") == ("Bibliography", SectionType.REFERENCES)

    def test_subsection_is_not_a_top_level_boundary(self) -> None:
        assert detect_heading("1.1 Background") is None

    def test_ordinary_sentence_is_not_a_heading(self) -> None:
        assert detect_heading("This is a regular sentence in a paragraph.") is None

    def test_blank_line_is_not_a_heading(self) -> None:
        assert detect_heading("   ") is None

    def test_bare_ambiguous_word_is_not_a_heading(self) -> None:
        # Not in the vocabulary and not numbered -- must not be misdetected.
        assert detect_heading("Overview") is None


class TestUnknownHeadings:
    def test_ablation_study_is_recognized_as_other(self) -> None:
        assert detect_heading("Ablation Study") == ("Ablation Study", SectionType.OTHER)

    def test_ethics_statement_is_recognized_as_other(self) -> None:
        assert detect_heading("Ethics Statement") == ("Ethics Statement", SectionType.OTHER)

    def test_appendix_is_recognized_as_other(self) -> None:
        assert detect_heading("Appendix") == ("Appendix", SectionType.OTHER)

    def test_numbered_unrecognized_heading_still_becomes_other(self) -> None:
        # Strong structural (numbered) signal -- preserved as OTHER even
        # though "Broader Societal Impact" isn't in the vocabulary at all.
        assert detect_heading("9 Broader Societal Impact") == (
            "Broader Societal Impact",
            SectionType.OTHER,
        )


def _pages(*texts: str) -> list[ParsedPage]:
    return [ParsedPage(page_number=i, text=text) for i, text in enumerate(texts, start=1)]


class TestSectionRecoveryOrder:
    def test_recovers_sections_in_document_order(self) -> None:
        pages = _pages(
            "Abstract\nAn abstract.\n\n1 Introduction\nIntro text.",
            "2 Methodology\nMethod text.\n\nReferences\n[1] A reference.",
        )
        sections, warnings = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )

        assert [s.section_type for s in sections] == [
            SectionType.ABSTRACT,
            SectionType.INTRODUCTION,
            SectionType.METHODOLOGY,
            SectionType.REFERENCES,
        ]
        assert [s.order for s in sections] == [0, 1, 2, 3]
        assert warnings == []

    def test_original_title_text_is_preserved(self) -> None:
        pages = _pages("I. INTRODUCTION\nSome text.")
        sections, _warnings = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )
        assert sections[0].title == "INTRODUCTION"
        assert sections[0].section_type == SectionType.INTRODUCTION

    def test_section_ids_are_deterministic(self) -> None:
        pages = _pages("Abstract\nText.")
        first, _ = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )
        second, _ = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )
        assert first[0].section_id == second[0].section_id


class TestPageProvenance:
    def test_section_spanning_pages_has_correct_page_range(self) -> None:
        pages = _pages(
            "1 Introduction\nIntro starts here",
            "still introduction text\n\n2 Methodology\nMethod text.",
        )
        sections, _warnings = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )

        introduction = next(s for s in sections if s.section_type == SectionType.INTRODUCTION)
        methodology = next(s for s in sections if s.section_type == SectionType.METHODOLOGY)

        assert introduction.page_start == 1
        assert introduction.page_end == 2
        assert methodology.page_start == 2
        assert methodology.page_end == 2

    def test_single_page_section_has_equal_start_and_end(self) -> None:
        pages = _pages("Abstract\nJust one page of content.")
        sections, _warnings = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )
        assert sections[0].page_start == sections[0].page_end == 1


class TestNoHeadingsDetected:
    def test_whole_document_becomes_one_other_section(self) -> None:
        pages = _pages("Just plain text with no recognizable headings at all.")
        sections, warnings = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )
        assert len(sections) == 1
        assert sections[0].section_type == SectionType.OTHER
        assert ParseWarning.NO_SECTION_HEADINGS_DETECTED in warnings

    def test_empty_document_yields_no_sections(self) -> None:
        sections, warnings = recover_sections(
            [], paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )
        assert sections == []
        assert ParseWarning.NO_SECTION_HEADINGS_DETECTED in warnings


class TestMissingAbstractOrReferencesWarnings:
    def test_missing_abstract_produces_warning(self) -> None:
        pages = _pages("1 Introduction\nText.\n\nReferences\n[1] Ref.")
        _sections, warnings = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )
        assert ParseWarning.NO_ABSTRACT_DETECTED in warnings
        assert ParseWarning.NO_REFERENCES_DETECTED not in warnings

    def test_missing_references_produces_warning(self) -> None:
        pages = _pages("Abstract\nText.\n\n1 Introduction\nText.")
        _sections, warnings = recover_sections(
            pages, paper_id="paper:arxiv:x", paper_version_id="paper-version:arxiv:x:v1"
        )
        assert ParseWarning.NO_REFERENCES_DETECTED in warnings
        assert ParseWarning.NO_ABSTRACT_DETECTED not in warnings


class TestQualityWarnings:
    def test_empty_page_is_flagged(self) -> None:
        pages = _pages("Some real content here that is long enough.", "")
        warnings = compute_quality_warnings(pages)
        assert ParseWarning.EMPTY_PAGE_DETECTED in warnings

    def test_low_text_density_is_flagged(self) -> None:
        pages = _pages("x", "y")
        warnings = compute_quality_warnings(pages)
        assert ParseWarning.LOW_TEXT_DENSITY in warnings

    def test_substantial_text_is_not_flagged_as_low_density(self) -> None:
        pages = _pages("word " * 100, "word " * 100)
        warnings = compute_quality_warnings(pages)
        assert ParseWarning.LOW_TEXT_DENSITY not in warnings
