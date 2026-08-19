"""Deterministic validation of answer evidence markers into trusted citations."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings
from app.domain.evidence import AnswerCitation, EvidenceItem, EvidencePool
from app.domain.ids import ensure_json_safe, normalize_whitespace
from app.generation.answer import AnswerGenerationContext, GeneratedGroundedAnswer

_MARKER_LIKE_RE = re.compile(r"\[E[^\]]*\]")
_VALID_MARKER_RE = re.compile(r"\[E([1-9][0-9]*)\]")


class CitationValidationStatus(str, Enum):
    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"
    NO_CITATIONS = "no_citations"


class CitationMarkerRecord(BaseModel):
    marker: str
    evidence_label: str | None = None
    valid_syntax: bool
    valid: bool
    reason: str | None = None


class CitationValidationResult(BaseModel):
    validated_text: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    valid_markers: list[str] = Field(default_factory=list)
    invalid_markers: list[CitationMarkerRecord] = Field(default_factory=list)
    uncited_evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validation_status: CitationValidationStatus
    validation_fingerprint: str
    validation_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("validation_metadata")
    @classmethod
    def _metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)


class ValidatedGroundedAnswer(BaseModel):
    text: str
    raw_text: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    citation_validation: CitationValidationResult
    generation_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", "raw_text")
    @classmethod
    def _text_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("validated answer text must not be blank")
        return normalized

    @field_validator("generation_metadata")
    @classmethod
    def _generation_metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)


class CitationValidator:
    """Validate visible E-markers against the actual answer context and pool."""

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    def validate(
        self,
        *,
        generated_answer: GeneratedGroundedAnswer,
        evidence_pool: EvidencePool,
        answer_context: AnswerGenerationContext,
    ) -> ValidatedGroundedAnswer:
        pool_by_label = {item.pool_id: item.evidence for item in evidence_pool.items}
        context_by_label = {item.pool_id: item for item in answer_context.evidence_items}
        for label, context_item in context_by_label.items():
            pool_item = pool_by_label.get(label)
            if pool_item is None or pool_item.evidence_id != context_item.evidence_id:
                raise ValueError(f"answer context label {label} does not match final evidence pool")

        citations_by_label: dict[str, AnswerCitation] = {}
        valid_markers: list[str] = []
        invalid_markers: list[CitationMarkerRecord] = []
        warnings: list[str] = []

        def replace_marker(match: re.Match[str]) -> str:
            marker = match.group(0)
            syntax_match = _VALID_MARKER_RE.fullmatch(marker)
            if syntax_match is None:
                invalid_markers.append(
                    CitationMarkerRecord(marker=marker, valid_syntax=False, valid=False, reason="invalid_syntax")
                )
                warnings.append(f"removed invalid citation marker {marker}")
                return ""

            label = f"E{int(syntax_match.group(1))}"
            evidence = pool_by_label.get(label)
            context_item = context_by_label.get(label)
            reason: str | None = None
            if context_item is None:
                reason = "not_in_answer_context"
            elif evidence is None:
                reason = "not_in_evidence_pool"
            elif evidence.evidence_id != context_item.evidence_id:
                reason = "context_pool_mismatch"
            elif _has_fatal_provenance(evidence):
                reason = "fatal_provenance"

            if reason is not None:
                invalid_markers.append(
                    CitationMarkerRecord(
                        marker=marker,
                        evidence_label=label,
                        valid_syntax=True,
                        valid=False,
                        reason=reason,
                    )
                )
                warnings.append(f"removed unsupported citation marker {marker}: {reason}")
                return ""

            if label not in citations_by_label:
                citation_number = len(citations_by_label) + 1
                citations_by_label[label] = _build_citation(
                    citation_number=citation_number,
                    evidence_label=label,
                    evidence=evidence,
                )
            valid_markers.append(marker)
            return f"[{citations_by_label[label].citation_number}]"

        validated_text = _normalize_marker_spacing(_MARKER_LIKE_RE.sub(replace_marker, generated_answer.text))
        citations = list(citations_by_label.values())
        status = _validation_status(markers_found=bool(valid_markers or invalid_markers), citations=citations, invalid_markers=invalid_markers)
        context_labels = set(context_by_label)
        cited_labels = set(citations_by_label)
        uncited_evidence = [label for label in context_by_label if label in context_labels - cited_labels]
        fingerprint = citation_validation_fingerprint(
            settings=self._settings,
            context_fingerprint=answer_context.context_fingerprint,
            generation_config_fingerprint=answer_context.generation_config_fingerprint,
            raw_answer_text=generated_answer.text,
        )
        result = CitationValidationResult(
            validated_text=validated_text,
            citations=citations,
            valid_markers=valid_markers,
            invalid_markers=invalid_markers,
            uncited_evidence=uncited_evidence,
            warnings=warnings,
            validation_status=status,
            validation_fingerprint=fingerprint,
            validation_metadata={
                "validator_version": self._settings.CITATION_VALIDATOR_VERSION,
                "marker_schema_version": self._settings.CITATION_MARKER_SCHEMA_VERSION,
                "context_fingerprint": answer_context.context_fingerprint,
                "generation_config_fingerprint": answer_context.generation_config_fingerprint,
            },
        )
        return ValidatedGroundedAnswer(
            text=validated_text,
            raw_text=generated_answer.text,
            citations=citations,
            citation_validation=result,
            generation_metadata={
                **generated_answer.generation_metadata,
                "citation_validation": status.value,
                "citation_validation_fingerprint": fingerprint,
            },
        )


def citation_validation_fingerprint(
    *,
    settings: Settings,
    context_fingerprint: str,
    generation_config_fingerprint: str,
    raw_answer_text: str,
) -> str:
    canonical = {
        "validator_version": settings.CITATION_VALIDATOR_VERSION,
        "marker_schema_version": settings.CITATION_MARKER_SCHEMA_VERSION,
        "context_fingerprint": context_fingerprint,
        "generation_config_fingerprint": generation_config_fingerprint,
        "raw_answer_sha256": hashlib.sha256(raw_answer_text.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


def _build_citation(*, citation_number: int, evidence_label: str, evidence: EvidenceItem) -> AnswerCitation:
    provenance = evidence.provenance
    source_store = None
    if evidence.source_store is not None:
        source_store = evidence.source_store.value
    elif provenance and provenance.source_store is not None:
        source_store = provenance.source_store.value
    metadata: dict[str, Any] = {}
    if evidence.evidence_type.value == "graph_path":
        metadata["nodes"] = evidence.metadata.get("nodes", evidence.entity_ids)
        metadata["relationships"] = evidence.metadata.get("relationships", evidence.relationship_ids)
    return AnswerCitation(
        citation_id=f"C{citation_number}",
        citation_number=citation_number,
        evidence_label=evidence_label,
        evidence_id=evidence.evidence_id,
        evidence_type=evidence.evidence_type,
        paper_id=evidence.paper_id or "",
        paper_version_id=evidence.paper_version_id,
        chunk_id=evidence.chunk_id,
        section_id=evidence.section_id,
        section_type=evidence.section_type,
        page_start=evidence.page_start,
        page_end=evidence.page_end,
        entity_ids=evidence.entity_ids,
        relationship_ids=evidence.relationship_ids,
        source_chunk_ids=evidence.source_chunk_ids,
        provenance_complete=provenance.provenance_complete if provenance else None,
        provenance_warnings=provenance.warnings if provenance else [],
        source_store=source_store,
        supporting_text_evidence_ids=evidence.supporting_text_evidence_ids,
        label=f"[{citation_number}]",
        metadata=metadata,
    )


def _has_fatal_provenance(evidence: EvidenceItem) -> bool:
    if evidence.metadata.get("fatal_provenance") is True:
        return True
    provenance = evidence.provenance
    if provenance is None:
        return False
    return any("fatal" in warning.lower() for warning in provenance.warnings)


def _validation_status(
    *,
    markers_found: bool,
    citations: list[AnswerCitation],
    invalid_markers: list[CitationMarkerRecord],
) -> CitationValidationStatus:
    if not markers_found:
        return CitationValidationStatus.NO_CITATIONS
    if citations and invalid_markers:
        return CitationValidationStatus.PARTIALLY_VALID
    if citations:
        return CitationValidationStatus.VALID
    return CitationValidationStatus.INVALID


def _normalize_marker_spacing(text: str) -> str:
    normalized = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\s+\n", "\n", normalized)
    return normalize_whitespace(normalized)
