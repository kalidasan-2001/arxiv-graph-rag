"""The chunker abstraction the rest of the application depends on.

Application code depends only on this `Protocol` and `ChunkedPaperDocument`
-- never on a specific splitting algorithm or tokenizer implementation
(prompt #3-equivalent for chunking). The chunker consumes the internal
`ParsedPaperDocument` model, never a raw PDF-library object.
"""

from typing import Protocol

from app.ingestion.chunking.models import ChunkedPaperDocument
from app.ingestion.parsing.models import ParsedPaperDocument


class ScientificChunker(Protocol):
    """A deterministic, section-aware splitter."""

    @property
    def chunking_version(self) -> str:
        """Persisted as `chunking.version` -- an informational label, not
        the reuse/invalidation identity check (see `config_fingerprint`)."""
        ...

    @property
    def config_fingerprint(self) -> str:
        """Deterministic fingerprint of the complete effective chunking
        configuration (prompt 6.1) -- what `ChunkingService` actually
        compares to decide whether an existing `chunks.json` is still
        valid, and what gets baked into every `chunk_id`. Computable
        without calling `chunk()`, since it depends only on configuration,
        not on any particular document.
        """
        ...

    def chunk(self, document: ParsedPaperDocument) -> ChunkedPaperDocument:
        """Split `document`'s recovered sections into `PaperChunk`s.

        Never mixes text from two different sections into the same chunk
        (prompt #4, the core architecture rule). Raises `ChunkingError` if
        the generated chunks fail validation (prompt #35) -- a chunker
        bug, not user error.
        """
        ...
