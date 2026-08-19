"""Domain layer: stable internal models and identifiers.

These types are the contract every later subsystem (persistence, ingestion,
retrieval, reasoning) depends on. They intentionally have no knowledge of
PostgreSQL, Qdrant, Neo4j, or any provider SDK.
"""
