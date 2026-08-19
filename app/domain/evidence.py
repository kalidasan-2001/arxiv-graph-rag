"""Normalized evidence, citation, and research-answer domain models.

``EvidenceItem`` is the single shared representation for both Qdrant-derived
textual evidence and Neo4j-derived graph evidence (CLAUDE.md #22) -- the
reasoning layer must be able to consume one evidence pool regardless of
which retrieval strategy produced it.

The citation contract (CLAUDE.md #9) is enforced structurally here: an
``AnswerCitation`` can only point at an ``evidence_id``, never at a raw,
LLM-generated URL or paper name. Whether that ``evidence_id`` actually
exists in the retrieved pool is validated later, at generation time -- this
stage only defines the shape.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.enums import (
    ConfidenceLevel,
    EvidenceScoreKind,
    EvidenceSourceStore,
    EvidenceType,
    RetrievalStrategy,
)
from app.domain.ids import build_evidence_id, ensure_json_safe, normalize_whitespace


class EvidenceProvenance(BaseModel):
    """Structured provenance common to text and graph evidence."""

    provenance_type: str
    source_store: EvidenceSourceStore
    paper_id: str | None = None
    paper_version_id: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    graph_index_generation_fingerprint: str | None = None
    graph_extraction_generation_fingerprint: str | None = None
    vector_generation_fingerprint: str | None = None
    extraction_version: str | None = None
    provenance_complete: bool = True
    warnings: list[str] = Field(default_factory=list)

    @field_validator("provenance_type")
    @classmethod
    def _provenance_type_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("provenance_type must not be blank")
        return normalized


class EvidenceItem(BaseModel):
    """A single normalized piece of evidence, from either Qdrant or Neo4j."""

    evidence_id: str
    evidence_type: EvidenceType
    paper_id: str | None = None
    paper_version_id: str | None = None
    chunk_id: str | None = None
    section_id: str | None = None
    section_type: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    text: str | None = None
    score: float | None = None
    score_kind: EvidenceScoreKind | None = None
    source: str
    source_store: EvidenceSourceStore | None = None
    provenance: EvidenceProvenance | None = None
    supporting_text_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _source_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("source must not be blank")
        return normalized

    @field_validator("score")
    @classmethod
    def _score_bounds(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        return value

    @model_validator(mode="after")
    def _score_kind_matches_score(self) -> "EvidenceItem":
        if self.score is None and self.score_kind is not None:
            raise ValueError("score_kind requires score")
        if self.score is not None and self.score_kind is None:
            raise ValueError("score requires score_kind")
        return self

    @field_validator("metadata")
    @classmethod
    def _metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)

    @model_validator(mode="after")
    def _has_provenance_reference(self) -> "EvidenceItem":
        """Evidence with no chunk/entity/relationship reference has no
        traceable source, which violates the provenance rule (CLAUDE.md #6)."""

        if not (self.chunk_id or self.source_chunk_ids or self.entity_ids or self.relationship_ids):
            raise ValueError(
                "EvidenceItem must reference at least one chunk_id, source_chunk_ids, "
                "entity_id, or relationship_id"
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        evidence_type: EvidenceType,
        source: str,
        chunk_id: str | None = None,
        entity_ids: list[str] | None = None,
        relationship_ids: list[str] | None = None,
        source_chunk_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "EvidenceItem":
        """Construct an ``EvidenceItem`` with its id derived via `build_evidence_id`."""

        entity_ids = entity_ids or []
        relationship_ids = relationship_ids or []
        source_chunk_ids = source_chunk_ids or []
        reference_ids = [
            rid for rid in [chunk_id, *source_chunk_ids, *entity_ids, *relationship_ids] if rid
        ]
        return cls(
            evidence_id=build_evidence_id(evidence_type, *reference_ids),
            evidence_type=evidence_type,
            source=source,
            chunk_id=chunk_id,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            source_chunk_ids=source_chunk_ids,
            **kwargs,
        )


class EvidencePoolItem(BaseModel):
    """Runtime label assigned to a stable evidence item."""

    pool_id: str
    evidence: EvidenceItem


class EvidencePool(BaseModel):
    """Closed evidence pool prepared for future generation/citation checks."""

    items: list[EvidencePoolItem] = Field(default_factory=list)


def build_evidence_pool(evidence: list[EvidenceItem]) -> EvidencePool:
    """Deduplicate by stable evidence id and assign deterministic E1/E2 labels."""

    seen: set[str] = set()
    items: list[EvidencePoolItem] = []
    for item in evidence:
        if item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        items.append(EvidencePoolItem(pool_id=f"E{len(items) + 1}", evidence=item))
    return EvidencePool(items=items)


class AnswerCitation(BaseModel):
    """A citation attached to a ``ResearchAnswer``.

    The only path to a source is ``evidence_id`` -- there is intentionally
    no free-form URL/reference field, so an LLM cannot fabricate a citation
    that bypasses the retrieved evidence pool.
    """

    citation_id: str
    evidence_id: str
    paper_id: str
    chunk_id: str | None = None
    label: str
    citation_number: int | None = None
    evidence_label: str | None = None
    evidence_type: EvidenceType | None = None
    paper_version_id: str | None = None
    section_id: str | None = None
    section_type: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    provenance_complete: bool | None = None
    provenance_warnings: list[str] = Field(default_factory=list)
    source_store: str | None = None
    supporting_text_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _label_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("label must not be blank")
        return normalized

    @field_validator("metadata")
    @classmethod
    def _citation_metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)


class ResearchAnswer(BaseModel):
    """A synthesized, evidence-grounded answer.

    Generation is not implemented here -- this is only the output shape a
    future LangGraph ``synthesize`` node must produce.
    """

    answer: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    confidence: ConfidenceLevel
    retrieval_strategy: RetrievalStrategy
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def _answer_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("answer must not be blank")
        return normalized
