"""Unit tests for `app.ingestion.parsing.normalization`.

Pure functions, no PDF library, no network -- always run.
"""

from app.ingestion.parsing.models import ParsedPage
from app.ingestion.parsing.normalization import (
    clean_page_text,
    collapse_blank_lines,
    collapse_inline_spaces,
    detect_repeated_header_footer_lines,
    normalize_line_endings,
    normalize_unicode,
    remove_repeated_header_footer_lines,
    repair_hyphenated_linebreaks,
)


class TestUnicodeNormalization:
    def test_nfkc_normalizes_compatibility_characters(self) -> None:
        # U+FB01 LATIN SMALL LIGATURE FI -> "fi"
        assert normalize_unicode("ﬁrst") == "first"


class TestLineEndingNormalization:
    def test_crlf_becomes_lf(self) -> None:
        assert normalize_line_endings("a\r\nb\r\nc") == "a\nb\nc"

    def test_bare_cr_becomes_lf(self) -> None:
        assert normalize_line_endings("a\rb") == "a\nb"


class TestHyphenatedLinebreakRepair:
    def test_lowercase_hyphen_break_is_joined(self) -> None:
        assert repair_hyphenated_linebreaks("informa-\ntion retrieval") == "information retrieval"

    def test_uppercase_start_is_not_joined(self) -> None:
        # Conservative: only lowercase-to-lowercase is trusted.
        text = "End of sentence-\nNext sentence starts here."
        assert repair_hyphenated_linebreaks(text) == text


class TestBlankLineCollapse:
    def test_multiple_blank_lines_collapse_to_one(self) -> None:
        text = "line one\n\n\n\nline two"
        assert collapse_blank_lines(text) == "line one\n\nline two"

    def test_single_blank_line_is_preserved(self) -> None:
        text = "line one\n\nline two"
        assert collapse_blank_lines(text) == text


class TestInlineSpaceCollapse:
    def test_repeated_spaces_collapse(self) -> None:
        assert collapse_inline_spaces("a   b     c") == "a b c"

    def test_newlines_are_preserved(self) -> None:
        assert collapse_inline_spaces("a   b\nc     d") == "a b\nc d"

    def test_leading_trailing_whitespace_is_trimmed_per_line(self) -> None:
        assert collapse_inline_spaces("  a b  \n  c d  ") == "a b\nc d"


class TestCleanPageTextPipeline:
    def test_applies_all_steps_together(self) -> None:
        raw = "Informa-\ntion   about\r\n\n\n\nretrieval systems."
        cleaned = clean_page_text(raw)
        assert "Information" in cleaned
        assert "\r" not in cleaned
        assert "   " not in cleaned


class TestRepeatedHeaderFooterDetection:
    def test_repeated_first_line_across_pages_is_detected_as_header(self) -> None:
        pages = [
            ParsedPage(page_number=i, text=f"Running Title\nBody text on page {i}.")
            for i in range(1, 6)
        ]
        headers, _footers = detect_repeated_header_footer_lines(pages)
        assert "running title" in headers

    def test_repeated_last_line_with_page_numbers_is_detected_as_footer(self) -> None:
        pages = [
            ParsedPage(page_number=i, text=f"Body text on page {i}.\nPage {i} of 5")
            for i in range(1, 6)
        ]
        _headers, footers = detect_repeated_header_footer_lines(pages)
        assert "page # of #" in footers

    def test_too_few_pages_detects_nothing(self) -> None:
        pages = [ParsedPage(page_number=1, text="Title\nBody"), ParsedPage(page_number=2, text="Title\nBody")]
        headers, footers = detect_repeated_header_footer_lines(pages)
        assert headers == set()
        assert footers == set()

    def test_non_repeated_first_lines_are_not_flagged(self) -> None:
        distinct_titles = ["Alpha Section", "Beta Section", "Gamma Section", "Delta Section", "Epsilon Section"]
        pages = [
            ParsedPage(page_number=i, text=f"{title}\nBody text.")
            for i, title in enumerate(distinct_titles, start=1)
        ]
        headers, _footers = detect_repeated_header_footer_lines(pages)
        assert headers == set()


class TestRepeatedHeaderFooterRemoval:
    def test_removes_only_confirmed_header_lines(self) -> None:
        # Three lines per page: a repeated header, unique body content
        # (won't match any header/footer pattern), and a repeated footer.
        pages = [
            ParsedPage(
                page_number=i,
                text=f"Running Title\nDistinctive body content number {i} that is not repeated.\nPage {i} of 5",
            )
            for i in range(1, 6)
        ]
        headers, footers = detect_repeated_header_footer_lines(pages)
        cleaned = remove_repeated_header_footer_lines(pages, headers, footers)

        for page in cleaned:
            assert "Running Title" not in page.text
            assert "Page" not in page.text.split("\n")[-1]
            assert "Distinctive body content" in page.text

    def test_no_headers_or_footers_leaves_pages_unchanged(self) -> None:
        pages = [ParsedPage(page_number=1, text="Unique content")]
        assert remove_repeated_header_footer_lines(pages, set(), set()) == pages
