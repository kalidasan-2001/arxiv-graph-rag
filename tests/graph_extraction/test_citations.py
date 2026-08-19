"""Unit tests for `citations.py` -- deterministic reference-entry
segmentation and arXiv-id resolution (prompt #16/#17/#59). Pure functions,
no DB, no LLM.
"""

from app.ingestion.graph_extraction.citations import (
    extract_arxiv_id,
    resolve_citations,
    split_reference_entries,
)


class TestSplitReferenceEntries:
    def test_numbered_bracket_entries_are_split(self) -> None:
        text = (
            "[1] Smith, J. Some Paper. 2020.\n"
            "[2] Doe, A. Another Paper. 2021.\n"
            "[3] Lee, K. A Third Paper. 2022."
        )
        entries = split_reference_entries(text)
        assert len(entries) == 3
        assert entries[0].startswith("Smith, J.")
        assert entries[2].startswith("Lee, K.")

    def test_numbered_dot_entries_are_split(self) -> None:
        text = "1. Smith, J. Paper One. 2020.\n2. Doe, A. Paper Two. 2021."
        entries = split_reference_entries(text)
        assert len(entries) == 2

    def test_falls_back_to_blank_line_blocks_when_unnumbered(self) -> None:
        text = "Smith, J. Some Paper. 2020.\n\nDoe, A. Another Paper. 2021."
        entries = split_reference_entries(text)
        assert len(entries) == 2

    def test_single_stray_marker_does_not_trigger_numbered_mode(self) -> None:
        # Only one "[1]"-like marker -- not trustworthy evidence of real
        # numbering, falls back to blank-line splitting instead.
        text = "[1] Smith, J. Some Paper. 2020.\n\nDoe, A. Another Paper without a marker. 2021."
        entries = split_reference_entries(text)
        assert len(entries) == 2


class TestExtractArxivId:
    def test_modern_id_with_prefix(self) -> None:
        assert extract_arxiv_id("Smith, J. Some Paper. arXiv:2401.12345, 2024.") == "2401.12345"

    def test_modern_id_without_prefix(self) -> None:
        assert extract_arxiv_id("Smith, J. Some Paper. 2401.12345, 2024.") == "2401.12345"

    def test_versioned_id_strips_version_suffix(self) -> None:
        assert extract_arxiv_id("arXiv:2401.12345v2") == "2401.12345"

    def test_legacy_style_id(self) -> None:
        assert extract_arxiv_id("Smith, J. Some Paper. cs/0501001, 2005.") == "cs/0501001"

    def test_no_id_present_returns_none(self) -> None:
        assert extract_arxiv_id("Smith, J. Some Paper. Conference Proceedings, 2024.") is None

    def test_case_insensitive_prefix(self) -> None:
        assert extract_arxiv_id("ARXIV: 2401.12345") == "2401.12345"


class TestResolveCitations:
    def test_explicit_arxiv_id_resolves(self) -> None:
        text = "[1] Smith, J. Some Paper. arXiv:2401.12345, 2024."
        resolved, unresolved = resolve_citations(text, source_chunk_id="chunk:refs")
        assert resolved == ["paper:arxiv:2401.12345"]
        assert unresolved == []

    def test_no_id_becomes_unresolved(self) -> None:
        text = "[1] Smith, J. Some Paper. Conference Proceedings, 2024."
        resolved, unresolved = resolve_citations(text, source_chunk_id="chunk:refs")
        assert resolved == []
        assert len(unresolved) == 1
        assert unresolved[0].source_chunk_id == "chunk:refs"
        assert unresolved[0].reason

    def test_mixed_entries_split_correctly(self) -> None:
        text = (
            "[1] Smith, J. Some Paper. arXiv:2401.12345, 2024.\n"
            "[2] Doe, A. Another Paper. Conference Proceedings, 2021.\n"
            "[3] Lee, K. Third Paper. arXiv:2205.09876v1, 2022."
        )
        resolved, unresolved = resolve_citations(text, source_chunk_id="chunk:refs")
        assert resolved == ["paper:arxiv:2401.12345", "paper:arxiv:2205.09876"]
        assert len(unresolved) == 1

    def test_never_resolves_from_title_alone(self) -> None:
        # Prompt #17: no fuzzy title matching implemented in V1 -- a
        # plausible-looking but ID-less entry must stay unresolved, never
        # guessed at.
        text = "[1] Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017."
        resolved, unresolved = resolve_citations(text, source_chunk_id="chunk:refs")
        assert resolved == []
        assert len(unresolved) == 1
