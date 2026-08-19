"""Unit tests for `WhitespaceTokenCounter`."""

from app.ingestion.chunking.tokenizer import WhitespaceTokenCounter


class TestWhitespaceTokenCounter:
    def test_counts_words_separated_by_single_spaces(self) -> None:
        assert WhitespaceTokenCounter().count("one two three") == 3

    def test_collapses_multiple_whitespace_runs(self) -> None:
        assert WhitespaceTokenCounter().count("one   two\n\nthree") == 3

    def test_empty_string_has_zero_tokens(self) -> None:
        assert WhitespaceTokenCounter().count("") == 0

    def test_whitespace_only_has_zero_tokens(self) -> None:
        assert WhitespaceTokenCounter().count("   \n  ") == 0

    def test_name_is_stable(self) -> None:
        assert WhitespaceTokenCounter().name == "whitespace-v1"
