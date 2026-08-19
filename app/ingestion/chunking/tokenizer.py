"""Token-counting abstraction.

The application depends only on `TokenCounter`, never a specific tokenizer
library or LLM provider (prompt #6, and consistent with CLAUDE.md's
provider-abstraction principle) -- chunk boundaries and identity must not
depend on which embedding/LLM provider is configured.
"""

from typing import Protocol


class TokenCounter(Protocol):
    """Counts "tokens" in a piece of text, by whatever definition `name` documents."""

    @property
    def name(self) -> str:
        """Identifier persisted as `chunking.tokenizer` in the chunk artifact."""
        ...

    @property
    def version(self) -> str | None:
        """A separate version identifier, if the implementation has one.

        `None` when the tokenizer's identity is already fully captured by
        `name` (e.g. `WhitespaceTokenCounter`'s "-v1" suffix) -- never
        fabricated just to fill this field (prompt #2, "do not invent a
        tokenizer version if one does not exist"). Feeds into
        `build_chunk_config_fingerprint` alongside `name` so a real future
        tokenizer swap (e.g. a specific BPE vocab version) is still
        distinguishable even if `name` were left unchanged by mistake.
        """
        ...

    def count(self, text: str) -> int:
        ...


class WhitespaceTokenCounter:
    """V1 tokenizer: a token is a maximal run of non-whitespace characters
    (i.e. ``len(text.split())`` -- word count).

    Deterministic, has no external dependency, and does not depend on any
    specific LLM provider's tokenizer. Documented approximation: this
    undercounts relative to typical BPE tokenizers (roughly 1.3x more real
    tokens per word for English text), which V1 accepts as a stable,
    reproducible baseline for chunk-size decisions -- not a claim of exact
    fidelity to any particular provider's token count.
    """

    @property
    def name(self) -> str:
        return "whitespace-v1"

    @property
    def version(self) -> str | None:
        # The "-v1" in `name` already *is* this tokenizer's version -- there
        # is no separate version identifier to report.
        return None

    def count(self, text: str) -> int:
        return len(text.split())
