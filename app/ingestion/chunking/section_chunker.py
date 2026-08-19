"""Deterministic, section-aware chunking (prompt #20).

Core rule: a chunk is never assembled from more than one recovered
section's text (prompt #4). Within a section, natural breakpoints are
preferred in this priority order:

    paragraph boundary -> sentence boundary -> word-boundary fallback

No LLM, no NLP library -- paragraphs are blank-line-separated blocks
(matching how Prompt 5's normalization already collapses runs of blank
lines to at most one), and sentences are split on a simple, deterministic
`. `/`! `/`? ` boundary regex. If even a single sentence exceeds the
configured chunk size (a giant run-on sentence, an equation block, ...),
it is split at word boundaries as a last resort -- chunks are never split
mid-word, and this guarantees no single generated chunk runs away
unboundedly large.
"""

import re

from app.domain.enums import SectionType
from app.domain.papers import PaperChunk, PaperSection
from app.ingestion.chunking.fingerprint import build_chunk_config_fingerprint
from app.ingestion.chunking.models import (
    ChunkDiagnostics,
    ChunkedPaperDocument,
    ChunkingConfig,
    ChunkWarning,
)
from app.ingestion.chunking.tokenizer import TokenCounter, WhitespaceTokenCounter
from app.ingestion.parsing.models import ParsedPaperDocument
from app.core.config import Settings
from app.core.exceptions import ChunkingError

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# A final chunk is flagged OVERSIZED_CHUNK only once it exceeds the
# configured size by this margin -- the greedy packer's overlap-prepending
# can legitimately push a chunk slightly past the exact target, and that
# alone shouldn't read as a quality problem.
_OVERSIZED_TOLERANCE = 1.2


class SectionAwareChunker:
    """`ScientificChunker` implementation: paragraph/sentence-aware,
    section-scoped, deterministic."""

    def __init__(self, settings: Settings, token_counter: TokenCounter | None = None) -> None:
        self._chunking_version = settings.CHUNKING_VERSION
        self._chunk_size = settings.CHUNK_SIZE_TOKENS
        self._overlap = settings.CHUNK_OVERLAP_TOKENS
        self._min_tokens = settings.MIN_CHUNK_TOKENS
        self._counter = token_counter or WhitespaceTokenCounter()
        # Computed once from configuration alone (prompt 6.1) -- doesn't
        # depend on any document, so it's available immediately for
        # `ChunkingService` to compare against a stored artifact without
        # first running `.chunk()`.
        self._config_fingerprint = build_chunk_config_fingerprint(
            chunking_version=self._chunking_version,
            chunk_size_tokens=self._chunk_size,
            chunk_overlap_tokens=self._overlap,
            min_chunk_tokens=self._min_tokens,
            tokenizer_name=self._counter.name,
            tokenizer_version=self._counter.version,
        )

    @property
    def chunking_version(self) -> str:
        return self._chunking_version

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    def chunk(self, document: ParsedPaperDocument) -> ChunkedPaperDocument:
        chunks: list[PaperChunk] = []
        warnings: set[ChunkWarning] = set()

        for section in sorted(document.sections, key=lambda s: s.order):
            if not section.text.strip():
                warnings.add(ChunkWarning.EMPTY_SECTION_SKIPPED)
                continue

            section_chunks, section_warnings = self._chunk_section(
                section, paper_id=document.paper_id, paper_version_id=document.paper_version_id
            )
            chunks.extend(section_chunks)
            warnings.update(section_warnings)

            if section.section_type == SectionType.REFERENCES and section_chunks:
                warnings.add(ChunkWarning.REFERENCES_PRESERVED)

        if chunks:
            # Every chunk's page range is its containing section's page
            # range -- Prompt 5 doesn't preserve finer (paragraph-level)
            # page provenance, so this is the smallest valid known range,
            # not fabricated precision (prompt #11). Always true whenever
            # there's at least one chunk, so always flagged.
            warnings.add(ChunkWarning.PAGE_PROVENANCE_APPROXIMATE)

        _validate_chunks(chunks, document)
        diagnostics = _compute_diagnostics(chunks, size=self._chunk_size, min_tokens=self._min_tokens)

        return ChunkedPaperDocument(
            paper_id=document.paper_id,
            paper_version_id=document.paper_version_id,
            source_pdf_checksum=document.source_pdf_checksum,
            chunking=ChunkingConfig(
                version=self._chunking_version,
                chunk_size_tokens=self._chunk_size,
                chunk_overlap_tokens=self._overlap,
                min_chunk_tokens=self._min_tokens,
                tokenizer=self._counter.name,
                tokenizer_version=self._counter.version,
                config_fingerprint=self._config_fingerprint,
            ),
            chunks=chunks,
            diagnostics=diagnostics,
            warnings=sorted(warnings, key=lambda w: w.value),
        )

    def _chunk_section(
        self, section: PaperSection, *, paper_id: str, paper_version_id: str
    ) -> tuple[list[PaperChunk], set[ChunkWarning]]:
        warnings: set[ChunkWarning] = set()
        texts = _split_section_into_chunks(
            section.text, size=self._chunk_size, overlap=self._overlap, counter=self._counter
        )
        texts, merged = _merge_tiny_trailing_chunk(
            texts, min_tokens=self._min_tokens, counter=self._counter
        )
        if merged:
            warnings.add(ChunkWarning.TINY_FRAGMENT_MERGED)

        chunks: list[PaperChunk] = []
        for index, text in enumerate(texts):
            token_count = self._counter.count(text)
            if token_count > self._chunk_size * _OVERSIZED_TOLERANCE:
                warnings.add(ChunkWarning.OVERSIZED_CHUNK)

            chunks.append(
                PaperChunk.create(
                    paper_id=paper_id,
                    paper_version_id=paper_version_id,
                    section_id=section.section_id,
                    section_type=section.section_type,
                    chunk_index=index,
                    text=text,
                    token_count=token_count,
                    chunk_config_fingerprint=self._config_fingerprint,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    metadata={
                        "section_title": section.title,
                        "chunking_version": self._chunking_version,
                        "overlap_tokens": self._overlap,
                    },
                )
            )
        return chunks, warnings


# --- Splitting -------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


def _split_by_words(text: str, size: int) -> list[str]:
    words = text.split()
    if not words or size <= 0:
        return [text] if text.strip() else []
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def _split_section_into_chunks(
    text: str, *, size: int, overlap: int, counter: TokenCounter
) -> list[str]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    # Small-section rule (prompt #14): if the whole section already fits,
    # it becomes exactly one chunk -- never split merely to make more of
    # them.
    if counter.count(text) <= size:
        return [text.strip()]

    units: list[str] = []
    for paragraph in paragraphs:
        if counter.count(paragraph) <= size:
            units.append(paragraph)
            continue
        for sentence in _split_sentences(paragraph):
            if counter.count(sentence) <= size:
                units.append(sentence)
            else:
                # Last resort: word-boundary split of a single oversized
                # sentence (prompt #12) -- never split mid-word.
                units.extend(_split_by_words(sentence, size))

    return _pack_units(units, size=size, overlap=overlap, counter=counter)


def _pack_units(units: list[str], *, size: int, overlap: int, counter: TokenCounter) -> list[str]:
    """Greedily pack `units` (paragraphs/sentences/word-groups, all already
    at or under `size` tokens individually) into chunks, carrying
    `overlap` tokens of trailing context into the next chunk within the
    same section only (prompt #13)."""

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = counter.count(unit)
        if current and current_tokens + unit_tokens > size:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap, counter)
            current_tokens = sum(counter.count(u) for u in current)

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _overlap_tail(units: list[str], overlap_tokens: int, counter: TokenCounter) -> list[str]:
    """The trailing whole units of `units` totaling ~`overlap_tokens` --
    never a partial-sentence slice, so overlapped context stays coherent.

    Stops *before* adding any unit that alone would push the tail past
    `overlap_tokens` -- including on the very first (most recent) unit
    considered, so a tail candidate too large to fit the overlap budget on
    its own is skipped entirely rather than included anyway. Without this,
    a single unit close to `chunk_size_tokens` (e.g. a word-boundary-split
    fallback piece) could be carried forward as "overlap" and immediately
    double the next chunk's size once a new unit is added on top of it. An
    empty return means "no overlap for this boundary" -- correct when
    nothing small enough is available, not a bug.
    """

    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = counter.count(unit)
        if total + unit_tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        total += unit_tokens
        if total >= overlap_tokens:
            break
    return tail


def _merge_tiny_trailing_chunk(
    chunk_texts: list[str], *, min_tokens: int, counter: TokenCounter
) -> tuple[list[str], bool]:
    """Merge a too-small trailing chunk into its predecessor within the
    same section (prompt #15) -- never into a different section, and never
    when the section produced only one chunk (nothing to merge into)."""

    if len(chunk_texts) <= 1:
        return chunk_texts, False
    if counter.count(chunk_texts[-1]) >= min_tokens:
        return chunk_texts, False

    merged = chunk_texts[:-2] + [chunk_texts[-2] + "\n\n" + chunk_texts[-1]]
    return merged, True


# --- Validation and diagnostics --------------------------------------------


def _validate_chunks(chunks: list[PaperChunk], document: ParsedPaperDocument) -> None:
    """Prompt #35's validation rules. A violation here is a chunker
    correctness bug, not a user-facing condition -- fails loudly."""

    valid_section_ids = {section.section_id for section in document.sections}
    seen_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            raise ChunkingError(f"duplicate chunk_id generated: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        if chunk.chunk_index < 0:
            raise ChunkingError(f"invalid chunk_index for {chunk.chunk_id}")
        if not chunk.text.strip():
            raise ChunkingError(f"empty chunk text for {chunk.chunk_id}")
        if chunk.token_count <= 0:
            raise ChunkingError(f"non-positive token_count for {chunk.chunk_id}")
        if chunk.paper_id != document.paper_id or chunk.paper_version_id != document.paper_version_id:
            raise ChunkingError(f"paper identity mismatch for {chunk.chunk_id}")
        if chunk.section_id not in valid_section_ids:
            raise ChunkingError(f"chunk {chunk.chunk_id} references an unknown section_id")


def _compute_diagnostics(chunks: list[PaperChunk], *, size: int, min_tokens: int) -> ChunkDiagnostics:
    if not chunks:
        return ChunkDiagnostics(
            chunk_count=0,
            min_tokens=0,
            max_tokens=0,
            average_tokens=0,
            median_tokens=0,
            small_chunk_count=0,
            oversized_chunk_count=0,
        )

    counts = sorted(chunk.token_count for chunk in chunks)
    n = len(counts)
    median = float(counts[n // 2]) if n % 2 == 1 else (counts[n // 2 - 1] + counts[n // 2]) / 2
    return ChunkDiagnostics(
        chunk_count=n,
        min_tokens=counts[0],
        max_tokens=counts[-1],
        average_tokens=sum(counts) / n,
        median_tokens=median,
        small_chunk_count=sum(1 for c in counts if c < min_tokens),
        oversized_chunk_count=sum(1 for c in counts if c > size * _OVERSIZED_TOLERANCE),
    )
