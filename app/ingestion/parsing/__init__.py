"""Scientific PDF parsing: page-aware text extraction and section recovery.

Boundary: ``PaperParsingService -> ScientificPaperParser -> PDF library``.
Nothing outside ``app/ingestion/parsing/pymupdf_parser.py`` should import
``pymupdf`` -- the rest of the application depends only on the
``ScientificPaperParser`` protocol and the ``ParsedPaperDocument`` DTO, so
the underlying library can be replaced later without touching the
ingestion pipeline.

Stops at a persisted, structured ``parsed.json`` (status ``PARSED``) --
no chunking, no embeddings, no entity extraction.
"""
