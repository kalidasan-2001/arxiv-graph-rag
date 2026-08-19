"""PostgreSQL persistence: the operational metadata and ingestion control plane.

Owns paper metadata, paper versions, ingestion jobs, and ingestion step
state (CLAUDE.md #3). Does not own chunk text, vectors, or the knowledge
graph -- those belong to Qdrant and Neo4j in later stages.
"""
