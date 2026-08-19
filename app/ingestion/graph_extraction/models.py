"""DTOs for the scientific knowledge extraction stage.

Two layers, deliberately kept separate (prompt #33 lists "call LLM" and
"validate candidates" as distinct steps):

* ``Raw*Candidate`` -- exactly what the LLM returned, loosely typed
  (``entity_type``/``relationship_type``/``usage`` are plain strings) so
  one malformed field never fails parsing of an otherwise-good chunk
  response. This is the ``response_model`` passed to
  ``LLMProvider.generate_structured``.
* ``Extracted*Candidate`` -- the same information *after* deterministic
  ontology validation (``ontology.py``): strict enum types, confidence
  bounds enforced, non-empty names, and ``source_chunk_id`` attached.
  Nothing downstream ever sees a ``Raw*Candidate``.

``GraphExtractionArtifact.entities``/``relationships`` reuse the existing
Prompt 1 domain models (``ScientificEntity``/``ScientificRelationship``)
directly -- no second, incompatible entity/relationship shape.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.domain.enums import EntityType, RelationshipType
from app.domain.ids import normalize_whitespace
from app.domain.knowledge import ScientificEntity, ScientificRelationship

UsageClassification = Literal["used_by_this_paper", "mentioned_only"]


class RawEntityCandidate(BaseModel):
    """As returned directly by the LLM, before ontology validation."""

    entity_type: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    evidence_quote: str | None = None
    confidence: float


class RawRelationshipCandidate(BaseModel):
    """As returned directly by the LLM, before ontology validation."""

    relationship_type: str
    source_name: str
    source_type: str
    target_name: str
    target_type: str
    usage: str | None = None
    evidence_quote: str | None = None
    confidence: float


class RawExtractionResponse(BaseModel):
    """The full structured LLM output for one chunk -- the schema passed
    as ``response_model`` to ``LLMProvider.generate_structured`` (prompt #7)."""

    entities: list[RawEntityCandidate] = Field(default_factory=list)
    relationships: list[RawRelationshipCandidate] = Field(default_factory=list)


class ExtractedEntityCandidate(BaseModel):
    """A ``RawEntityCandidate`` that passed ontology validation (prompt #8):
    ``entity_type`` is a real ``EntityType``, ``name`` is non-blank,
    ``confidence`` is in ``[0, 1]``."""

    entity_type: EntityType
    name: str
    aliases: list[str] = Field(default_factory=list)
    evidence_quote: str | None = None
    confidence: float
    source_chunk_id: str

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("name must not be blank")
        return normalized

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return value


class ExtractedRelationshipCandidate(BaseModel):
    """A ``RawRelationshipCandidate`` that passed ontology validation
    (prompt #8/#9): both types are real, the (relationship_type,
    source_type, target_type) triple is in the compatibility matrix, names
    are non-blank, confidence is in ``[0, 1]``."""

    relationship_type: RelationshipType
    source_name: str
    source_type: EntityType
    target_name: str
    target_type: EntityType
    usage: UsageClassification | None = None
    evidence_quote: str | None = None
    confidence: float
    source_chunk_id: str

    @field_validator("source_name", "target_name")
    @classmethod
    def _names_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("name must not be blank")
        return normalized

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return value


class GraphExtractionWarningCode(str, Enum):
    """Deterministic, non-fatal extraction quality signals (mirrors
    `ChunkWarning`/`ParseWarning`'s pattern)."""

    ENTITY_CANDIDATE_REJECTED = "entity_candidate_rejected"
    RELATIONSHIP_CANDIDATE_REJECTED = "relationship_candidate_rejected"
    EVIDENCE_QUOTE_DISCARDED = "evidence_quote_discarded"
    CITATION_UNRESOLVED = "citation_unresolved"


class GraphExtractionWarning(BaseModel):
    code: GraphExtractionWarningCode
    detail: str
    source_chunk_id: str | None = None


class UnresolvedCitation(BaseModel):
    """A reference-list entry whose target paper could not be confidently
    resolved (prompt #16/#17) -- preserved for future work, never turned
    into a trusted `CITES` edge."""

    raw_text: str
    source_chunk_id: str
    reason: str


class ExtractionConfig(BaseModel):
    """Exactly what produced this extraction artifact -- the identity
    surface reuse/invalidation compares against (prompt #21/#22)."""

    version: str
    llm_provider: str
    llm_model: str
    llm_provider_version: str | None = None
    prompt_version: str
    schema_version: str
    temperature: float
    config_fingerprint: str
    generation_fingerprint: str


class GraphExtractionArtifact(BaseModel):
    """The full structured output of extracting one paper version's
    scientific knowledge (prompt #27). Never sent to Neo4j -- this is the
    trustworthy structured-data boundary Prompt 9 will consume."""

    paper_id: str
    paper_version_id: str
    source_chunk_artifact_checksum: str
    extraction: ExtractionConfig
    entities: list[ScientificEntity] = Field(default_factory=list)
    relationships: list[ScientificRelationship] = Field(default_factory=list)
    unresolved_citations: list[UnresolvedCitation] = Field(default_factory=list)
    warnings: list[GraphExtractionWarning] = Field(default_factory=list)
