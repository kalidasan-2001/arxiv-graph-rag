"""Conservative, deterministic text cleanup for extracted PDF pages.

Every operation here is a faithful-extraction concern, not an
interpretation one (prompt #10): no summarizing, no paraphrasing, no
grammar fixes, no LLM. If a heuristic isn't confident, it leaves the text
alone -- false "cleanup" that destroys real content is worse than leaving
some noise (prompt #11).
"""

import re
import unicodedata
from collections import Counter

from app.ingestion.parsing.models import ParsedPage

_INLINE_WHITESPACE = re.compile(r"[ \t]+")
# Conservative hyphenated-line-break repair: only joins when both the
# character before the hyphen and the character starting the next line are
# lowercase letters. This is a heuristic, not a certainty -- it will
# occasionally join a genuine compound word ("well-\nknown" -> "wellknown")
# and will just as often correctly leave an ambiguous case alone. Documented
# limitation, not a bug: true confidence requires layout/font information
# this stage doesn't use.
_HYPHENATED_LINEBREAK = re.compile(r"([a-z])-\n([a-z])")
# Used to compare header/footer candidate lines while ignoring page numbers.
_DIGIT_RUN = re.compile(r"\d+")


def normalize_unicode(text: str) -> str:
    """NFKC-normalize so visually-identical characters compare equal."""

    return unicodedata.normalize("NFKC", text)


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def repair_hyphenated_linebreaks(text: str) -> str:
    return _HYPHENATED_LINEBREAK.sub(r"\1\2", text)


def collapse_blank_lines(text: str, *, max_consecutive: int = 1) -> str:
    """Collapse runs of blank lines down to at most `max_consecutive`."""

    lines = text.split("\n")
    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_consecutive:
                result.append("")
        else:
            blank_run = 0
            result.append(line)
    return "\n".join(result)


def collapse_inline_spaces(text: str) -> str:
    """Collapse runs of horizontal whitespace to one space, per line --
    never touches newlines, so paragraph/line structure is preserved."""

    lines = text.split("\n")
    return "\n".join(_INLINE_WHITESPACE.sub(" ", line).strip() for line in lines)


def clean_page_text(text: str) -> str:
    """Apply the conservative cleanup pipeline, in order, to one page's text."""

    text = normalize_unicode(text)
    text = normalize_line_endings(text)
    text = repair_hyphenated_linebreaks(text)
    text = collapse_blank_lines(text)
    text = collapse_inline_spaces(text)
    return text


def _normalize_for_comparison(line: str) -> str:
    """Fold a line down to a comparable form: lowercase, digits collapsed
    to `#` (so "Page 3" and "Page 7" compare equal), whitespace trimmed."""

    return _DIGIT_RUN.sub("#", line.strip().lower())


def detect_repeated_header_footer_lines(
    pages: list[ParsedPage], *, min_page_count: int = 3, min_repeat_ratio: float = 0.6
) -> tuple[set[str], set[str]]:
    """Identify first/last-line patterns repeated across most pages.

    Conservative by design (prompt #11): requires at least `min_page_count`
    pages and the pattern to appear on at least `min_repeat_ratio` of them
    before it's trusted as a real header/footer rather than coincidence.
    Returns normalized-comparison-form strings, not the original text --
    use `remove_repeated_header_footer_lines` to actually strip them.
    """

    if len(pages) < min_page_count:
        return set(), set()

    first_lines: Counter[str] = Counter()
    last_lines: Counter[str] = Counter()
    for page in pages:
        lines = [line for line in page.text.split("\n") if line.strip()]
        if not lines:
            continue
        first_lines[_normalize_for_comparison(lines[0])] += 1
        last_lines[_normalize_for_comparison(lines[-1])] += 1

    threshold = max(min_page_count, round(len(pages) * min_repeat_ratio))
    headers = {line for line, count in first_lines.items() if line and count >= threshold}
    footers = {line for line, count in last_lines.items() if line and count >= threshold}
    return headers, footers


def remove_repeated_header_footer_lines(
    pages: list[ParsedPage], headers: set[str], footers: set[str]
) -> list[ParsedPage]:
    """Strip only lines matching a confirmed repeated header/footer pattern."""

    if not headers and not footers:
        return pages

    cleaned: list[ParsedPage] = []
    for page in pages:
        lines = page.text.split("\n")
        non_blank_indices = [i for i, line in enumerate(lines) if line.strip()]
        if non_blank_indices:
            first_idx = non_blank_indices[0]
            if _normalize_for_comparison(lines[first_idx]) in headers:
                lines[first_idx] = ""
            last_idx = non_blank_indices[-1]
            # `lines[last_idx].strip()` re-checks liveness rather than
            # comparing indices, so a single-line page already cleared as a
            # header isn't double-processed, but a single line that only
            # matches a footer pattern still gets removed correctly.
            if lines[last_idx].strip() and _normalize_for_comparison(lines[last_idx]) in footers:
                lines[last_idx] = ""
        cleaned.append(ParsedPage(page_number=page.page_number, text="\n".join(lines)))
    return cleaned
