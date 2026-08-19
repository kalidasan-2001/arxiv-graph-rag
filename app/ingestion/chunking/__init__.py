"""Section-aware scientific chunking: turns a `ParsedPaperDocument` into a
stable, deterministic `PaperChunk[]` corpus.

Boundary: ``ChunkingService -> ScientificChunker -> TokenCounter``. Chunks
are never embedded or indexed here -- that is Prompt 7's job. Chunking
always starts from `parsed.json` (Prompt 5's output), never from
`paper.pdf` directly, keeping every stage independently reproducible.
"""
