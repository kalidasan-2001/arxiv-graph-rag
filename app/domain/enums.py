"""Domain enumerations.

All enums subclass ``str`` so they serialize predictably (the JSON value is
simply the member's string value) and compare equal to plain strings where
convenient.
"""

from enum import Enum


class SectionType(str, Enum):
    """Structural role of a section within a scientific paper.

    Not every paper contains every section; ``OTHER`` covers anything that
    doesn't map to a recognized scientific-paper section.
    """

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    LIMITATIONS = "limitations"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    OTHER = "other"


class EntityType(str, Enum):
    """Supported scientific-knowledge-graph entity types (CLAUDE.md #5)."""

    PAPER = "paper"
    AUTHOR = "author"
    METHOD = "method"
    DATASET = "dataset"
    TASK = "task"


class RelationshipType(str, Enum):
    """Supported scientific-knowledge-graph relationship types (CLAUDE.md #5)."""

    AUTHORED_BY = "authored_by"
    CITES = "cites"
    USES_METHOD = "uses_method"
    EVALUATED_ON = "evaluated_on"
    ADDRESSES = "addresses"


class IngestionStatus(str, Enum):
    """Stage of the ingestion pipeline a paper version is currently in."""

    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PARSING = "parsing"
    PARSED = "parsed"
    CHUNKING = "chunking"
    CHUNKED = "chunked"
    VECTOR_INDEXING = "vector_indexing"
    VECTOR_INDEXED = "vector_indexed"
    GRAPH_INDEXING = "graph_indexing"
    GRAPH_INDEXED = "graph_indexed"
    READY = "ready"
    FAILED = "failed"


class RetrievalStrategy(str, Enum):
    """Independent retrieval strategies (CLAUDE.md #21)."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class EvidenceType(str, Enum):
    """Kind of source material an ``EvidenceItem`` represents."""

    TEXT = "text"
    GRAPH_RELATIONSHIP = "graph_relationship"
    GRAPH_PATH = "graph_path"
    METADATA = "metadata"


class EvidenceSourceStore(str, Enum):
    """Physical store or trusted source that produced an evidence item."""

    QDRANT = "qdrant"
    NEO4J = "neo4j"
    METADATA = "metadata"


class EvidenceScoreKind(str, Enum):
    """Meaning of ``EvidenceItem.score``.

    These values are intentionally not normalized into one comparable
    cross-store score. Fusion can decide how to interpret them later.
    """

    VECTOR_SIMILARITY = "vector_similarity"
    GRAPH_PATH_CONFIDENCE = "graph_path_confidence"


class ConfidenceLevel(str, Enum):
    """Coarse confidence rating for a synthesized research answer."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ProcessingStage(str, Enum):
    """A coarse, individually-attemptable ingestion processing stage.

    Distinct from ``IngestionStatus``: a job's ``status`` tracks overall
    pipeline progress (12 values, including *ING/*ED pairs), while
    ``ProcessingStage`` names the operation an ``ingestion_steps`` row is
    attempting. Added in the PostgreSQL control-plane stage per the
    ``ingestion_steps`` schema requirement.
    """

    DOWNLOAD = "download"
    PARSE = "parse"
    CHUNK = "chunk"
    VECTOR_INDEX = "vector_index"
    GRAPH_EXTRACT = "graph_extract"
    GRAPH_INDEX = "graph_index"


class StepStatus(str, Enum):
    """Lifecycle of a single ``ingestion_steps`` attempt row."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
