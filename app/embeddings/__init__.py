"""Embedding provider abstraction (prompt 7).

Application code depends only on `EmbeddingProvider` (`provider.py`) and
`EmbeddingConfig` (`config.py`) -- never directly on `sentence-transformers`,
Hugging Face classes, or any other vendor SDK object (CLAUDE.md #15).
"""
