"""Unit tests for `app.ingestion.discovery.normalization`.

Pure functions, no network -- always run.
"""

from datetime import datetime, timezone

import pytest

from app.ingestion.discovery.models import ArxivPaperResult
from app.ingestion.discovery.normalization import (
    arxiv_result_to_paper,
    arxiv_result_to_paper_version,
    normalize_abstract,
    normalize_arxiv_id,
    normalize_authors,
    normalize_categories,
    normalize_search_query,
    normalize_title,
    parse_arxiv_datetime,
)


class TestNormalizeArxivId:
    def test_bare_id_has_no_version(self) -> None:
        assert normalize_arxiv_id("2401.12345") == ("2401.12345", None)

    def test_bare_id_with_version_suffix(self) -> None:
        assert normalize_arxiv_id("2401.12345v2") == ("2401.12345", "v2")

    def test_http_abs_url_with_version(self) -> None:
        assert normalize_arxiv_id("http://arxiv.org/abs/2401.12345v2") == ("2401.12345", "v2")

    def test_https_abs_url_without_version(self) -> None:
        assert normalize_arxiv_id("https://arxiv.org/abs/2401.12345") == ("2401.12345", None)

    def test_https_abs_url_with_v3(self) -> None:
        assert normalize_arxiv_id("https://arxiv.org/abs/2401.12345v3") == ("2401.12345", "v3")

    def test_pdf_link_form(self) -> None:
        assert normalize_arxiv_id("http://arxiv.org/pdf/2401.12345v1") == ("2401.12345", "v1")

    def test_pdf_link_with_pdf_extension(self) -> None:
        assert normalize_arxiv_id("http://arxiv.org/pdf/2401.12345v1.pdf") == ("2401.12345", "v1")

    def test_blank_id_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            normalize_arxiv_id("   ")

    def test_paper_id_derivation_is_stable_across_versions(self) -> None:
        # Logical paper identity must remain paper:arxiv:2401.12345
        # regardless of which version string was in the raw input.
        from app.domain.ids import build_paper_id

        id_v1, _ = normalize_arxiv_id("2401.12345v1")
        id_v3, _ = normalize_arxiv_id("https://arxiv.org/abs/2401.12345v3")
        assert build_paper_id("arxiv", id_v1) == build_paper_id("arxiv", id_v3)
        assert build_paper_id("arxiv", id_v1) == "paper:arxiv:2401.12345"


class TestNormalizeSearchQuery:
    def test_collapses_surrounding_and_internal_whitespace(self) -> None:
        assert normalize_search_query("   graph rag   ") == "graph rag"

    def test_blank_query_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            normalize_search_query("    ")


class TestTitleAndAbstractNormalization:
    def test_title_whitespace_is_collapsed(self) -> None:
        assert normalize_title("Graph\n  RAG:   A Survey") == "Graph RAG: A Survey"

    def test_abstract_whitespace_is_collapsed(self) -> None:
        raw = "This paper\n introduces  a hybrid\n\n retrieval method."
        assert normalize_abstract(raw) == "This paper introduces a hybrid retrieval method."


class TestAuthorNormalization:
    def test_duplicate_authors_are_removed_preserving_order(self) -> None:
        authors = ["Alice Smith", "Bob Jones", "Alice Smith", "  Bob Jones  "]
        assert normalize_authors(authors) == ["Alice Smith", "Bob Jones"]

    def test_blank_author_entries_are_dropped(self) -> None:
        assert normalize_authors(["Alice Smith", "   ", ""]) == ["Alice Smith"]


class TestCategoryNormalization:
    def test_duplicates_removed_and_sorted(self) -> None:
        assert normalize_categories(["cs.CL", "cs.AI", "cs.CL"]) == ["cs.AI", "cs.CL"]

    def test_blank_categories_dropped(self) -> None:
        assert normalize_categories(["cs.AI", "  "]) == ["cs.AI"]


class TestDateParsing:
    def test_parses_zulu_suffix(self) -> None:
        parsed = parse_arxiv_datetime("2024-01-15T18:30:00Z")
        assert parsed == datetime(2024, 1, 15, 18, 30, 0, tzinfo=timezone.utc)

    def test_parses_explicit_offset(self) -> None:
        parsed = parse_arxiv_datetime("2024-01-15T18:30:00+00:00")
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 15


class TestArxivResultToPaper:
    def _result(self, **overrides) -> ArxivPaperResult:
        defaults = dict(
            source_id="2401.12345",
            version="v2",
            title="  Graph  RAG   ",
            abstract="An abstract\n with line   breaks.",
            authors=["Alice Smith", "Alice Smith"],
            categories=["cs.CL", "cs.AI"],
            published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        defaults.update(overrides)
        return ArxivPaperResult(**defaults)

    def test_produces_a_normalized_domain_paper(self) -> None:
        paper = arxiv_result_to_paper(self._result())
        assert paper.paper_id == "paper:arxiv:2401.12345"
        assert paper.title == "Graph RAG"
        assert paper.abstract == "An abstract with line breaks."
        assert paper.authors == ["Alice Smith"]
        assert paper.categories == ["cs.AI", "cs.CL"]

    def test_version_produces_a_paper_version(self) -> None:
        result = self._result()
        paper = arxiv_result_to_paper(result)
        version = arxiv_result_to_paper_version(result, paper_id=paper.paper_id)
        assert version is not None
        assert version.paper_id == paper.paper_id
        assert version.version == "v2"

    def test_missing_version_yields_no_paper_version(self) -> None:
        result = self._result(version=None)
        paper = arxiv_result_to_paper(result)
        version = arxiv_result_to_paper_version(result, paper_id=paper.paper_id)
        assert version is None
