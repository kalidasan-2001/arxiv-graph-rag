"""Deterministic final grounding gates and confidence classification."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings
from app.domain.enums import ConfidenceLevel, EvidenceType
from app.domain.evidence import AnswerCitation, EvidenceItem
from app.domain.ids import ensure_json_safe, normalize_whitespace
from app.generation.citations import CitationValidationResult, CitationValidationStatus, ValidatedGroundedAnswer
from app.retrieval.critic import EvidenceAssessment


class FinalAnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    REQUIRES_DISAMBIGUATION = "requires_disambiguation"
    FAILED = "failed"


class GroundingReasonCode(str, Enum):
    STRONG_GROUNDED_SUPPORT = "strong_grounded_support"
    SUFFICIENT_AFTER_REFINEMENT = "sufficient_after_refinement"
    PARTIAL_CITATION_VALIDATION = "partial_citation_validation"
    PROVENANCE_INCOMPLETE = "provenance_incomplete"
    MISSING_INFORMATION = "missing_information"
    LIMITED_EVIDENCE_COVERAGE = "limited_evidence_coverage"
    NO_TRUSTED_CITATIONS = "no_trusted_citations"
    INSUFFICIENT_RETRIEVAL_EVIDENCE = "insufficient_retrieval_evidence"
    ENTITY_AMBIGUOUS = "entity_ambiguous"
    ENTITY_NOT_FOUND = "entity_not_found"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    RETRIEVAL_FAILED = "retrieval_failed"
    CRITIC_FAILED = "critic_failed"
    REFINEMENT_FAILED = "refinement_failed"
    GENERATION_FAILED = "generation_failed"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"
    EMPTY_EVIDENCE = "empty_evidence"
    PLANNING_FAILED = "planning_failed"
    FATAL_PROVENANCE = "fatal_provenance"


class GroundingDecision(BaseModel):
    allow_answer: bool
    confidence: ConfidenceLevel
    reason_codes: list[GroundingReasonCode] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    grounding_fingerprint: str

    @field_validator("diagnostics")
    @classmethod
    def _diagnostics_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)


class FinalResearchAnswer(BaseModel):
    query: str
    status: FinalAnswerStatus
    internal_status: str
    answer: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    confidence: ConfidenceLevel
    grounding: GroundingDecision
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    retrieval_rounds: int = 0
    evidence_summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", "answer")
    @classmethod
    def _text_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("final answer text must not be blank")
        return normalized

    @field_validator("evidence_summary")
    @classmethod
    def _summary_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)


class GroundingDecisionService:
    """Make the final answer/abstention decision without using an LLM."""

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    def decide(
        self,
        *,
        query: str,
        internal_status: str,
        evidence: list[EvidenceItem],
        evidence_assessment: EvidenceAssessment | None,
        citation_validation: CitationValidationResult | None,
        validated_answer: ValidatedGroundedAnswer | None,
        retrieval_round: int,
        warnings: list[str],
        analysis_intent: str | None = None,
    ) -> FinalResearchAnswer:
        reason_codes: list[GroundingReasonCode] = []
        decision_warnings = list(warnings)
        trusted_citations = list(validated_answer.citations) if validated_answer else []
        validation_status = citation_validation.validation_status if citation_validation else None
        citation_diagnostics = _citation_diagnostics(citation_validation)
        final_status = FinalAnswerStatus.ABSTAINED

        hard_reason = _hard_gate_reason(
            internal_status=internal_status,
            evidence=evidence,
            evidence_assessment=evidence_assessment,
            validation_status=validation_status,
            trusted_citations=trusted_citations,
        )
        if hard_reason is not None:
            reason_codes.append(hard_reason)
            answer = _abstention_text(hard_reason)
            confidence = ConfidenceLevel.INSUFFICIENT_EVIDENCE
            citations: list[AnswerCitation] = []
        else:
            confidence = ConfidenceLevel.HIGH
            final_status = FinalAnswerStatus.ANSWERED
            strong_support = _strong_support_requirements_met(
                analysis_intent=analysis_intent,
                evidence_assessment=evidence_assessment,
                validation_status=validation_status,
                trusted_citations=trusted_citations,
            )
            if retrieval_round > 1:
                reason_codes.append(GroundingReasonCode.SUFFICIENT_AFTER_REFINEMENT)
            if validation_status == CitationValidationStatus.PARTIALLY_VALID:
                confidence = _min_confidence(confidence, ConfidenceLevel.MEDIUM)
                reason_codes.append(GroundingReasonCode.PARTIAL_CITATION_VALIDATION)
            if _has_nonfatal_provenance_gap(trusted_citations):
                confidence = _min_confidence(confidence, ConfidenceLevel.MEDIUM)
                reason_codes.append(GroundingReasonCode.PROVENANCE_INCOMPLETE)
            if evidence_assessment and evidence_assessment.missing_information:
                confidence = _min_confidence(confidence, ConfidenceLevel.MEDIUM)
                reason_codes.append(GroundingReasonCode.MISSING_INFORMATION)
            if _limited_evidence_coverage(evidence_assessment):
                confidence = _min_confidence(confidence, ConfidenceLevel.LOW)
                reason_codes.append(GroundingReasonCode.LIMITED_EVIDENCE_COVERAGE)
            if not strong_support:
                confidence = _min_confidence(confidence, ConfidenceLevel.MEDIUM)
                if GroundingReasonCode.LIMITED_EVIDENCE_COVERAGE not in reason_codes:
                    reason_codes.append(GroundingReasonCode.LIMITED_EVIDENCE_COVERAGE)
            if strong_support and not reason_codes:
                reason_codes.append(GroundingReasonCode.STRONG_GROUNDED_SUPPORT)
            answer = validated_answer.text if validated_answer else _abstention_text(GroundingReasonCode.GENERATION_FAILED)
            citations = trusted_citations

        reason_codes = _ordered_reason_codes(reason_codes)
        if internal_status == "REQUIRES_DISAMBIGUATION":
            final_status = FinalAnswerStatus.REQUIRES_DISAMBIGUATION

        fingerprint = grounding_fingerprint(
            settings=self._settings,
            internal_status=internal_status,
            evidence_assessment=evidence_assessment,
            citation_validation=citation_validation,
            validated_answer_text=validated_answer.text if validated_answer else None,
            reason_codes=reason_codes,
            confidence=confidence,
        )
        grounding = GroundingDecision(
            allow_answer=final_status == FinalAnswerStatus.ANSWERED,
            confidence=confidence,
            reason_codes=reason_codes,
            warnings=decision_warnings,
            abstention_reason=None if final_status == FinalAnswerStatus.ANSWERED else answer,
            diagnostics={
                **citation_diagnostics,
                "grounding_rules_version": self._settings.GROUNDING_RULES_VERSION,
                "confidence_rules_version": self._settings.CONFIDENCE_RULES_VERSION,
                "abstention_template_version": self._settings.ABSTENTION_TEMPLATE_VERSION,
                "evidence_count": len(evidence),
                "analysis_intent": analysis_intent,
                "strong_support_met": False if hard_reason is not None else strong_support,
            },
            grounding_fingerprint=fingerprint,
        )
        return FinalResearchAnswer(
            query=query,
            status=final_status,
            internal_status=internal_status,
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounding=grounding,
            missing_information=evidence_assessment.missing_information if evidence_assessment else [],
            warnings=decision_warnings,
            retrieval_rounds=retrieval_round,
            evidence_summary=_evidence_summary(evidence),
        )


def grounding_fingerprint(
    *,
    settings: Settings,
    internal_status: str,
    evidence_assessment: EvidenceAssessment | None,
    citation_validation: CitationValidationResult | None,
    validated_answer_text: str | None,
    reason_codes: list[GroundingReasonCode],
    confidence: ConfidenceLevel,
) -> str:
    canonical = {
        "grounding_rules_version": settings.GROUNDING_RULES_VERSION,
        "confidence_rules_version": settings.CONFIDENCE_RULES_VERSION,
        "abstention_template_version": settings.ABSTENTION_TEMPLATE_VERSION,
        "citation_validation_fingerprint": citation_validation.validation_fingerprint if citation_validation else None,
        "evidence_assessment": evidence_assessment.model_dump(mode="json") if evidence_assessment else None,
        "workflow_final_status": internal_status,
        "validated_answer_sha256": hashlib.sha256((validated_answer_text or "").encode("utf-8")).hexdigest(),
        "reason_codes": [code.value for code in reason_codes],
        "confidence": confidence.value,
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


def _hard_gate_reason(
    *,
    internal_status: str,
    evidence: list[EvidenceItem],
    evidence_assessment: EvidenceAssessment | None,
    validation_status: CitationValidationStatus | None,
    trusted_citations: list[AnswerCitation],
) -> GroundingReasonCode | None:
    status_map = {
        "PLANNING_FAILED": GroundingReasonCode.PLANNING_FAILED,
        "REQUIRES_DISAMBIGUATION": GroundingReasonCode.ENTITY_AMBIGUOUS,
        "ENTITY_NOT_FOUND": GroundingReasonCode.ENTITY_NOT_FOUND,
        "UNSUPPORTED_OPERATION": GroundingReasonCode.UNSUPPORTED_OPERATION,
        "RETRIEVAL_FAILED": GroundingReasonCode.RETRIEVAL_FAILED,
        "EMPTY_EVIDENCE": GroundingReasonCode.EMPTY_EVIDENCE,
        "INSUFFICIENT_EVIDENCE": GroundingReasonCode.INSUFFICIENT_RETRIEVAL_EVIDENCE,
        "REFINEMENT_FAILED": GroundingReasonCode.REFINEMENT_FAILED,
        "CRITIC_FAILED": GroundingReasonCode.CRITIC_FAILED,
        "ANSWER_GENERATION_FAILED": GroundingReasonCode.GENERATION_FAILED,
        "CITATION_VALIDATION_FAILED": GroundingReasonCode.CITATION_VALIDATION_FAILED,
    }
    if internal_status in status_map:
        return status_map[internal_status]
    if evidence_assessment is None or evidence_assessment.sufficient is not True:
        return GroundingReasonCode.INSUFFICIENT_RETRIEVAL_EVIDENCE
    if not evidence:
        return GroundingReasonCode.EMPTY_EVIDENCE
    if validation_status in {CitationValidationStatus.INVALID, CitationValidationStatus.NO_CITATIONS}:
        return GroundingReasonCode.CITATION_VALIDATION_FAILED
    if not trusted_citations:
        return GroundingReasonCode.NO_TRUSTED_CITATIONS
    if evidence and all(_has_fatal_provenance(item) for item in evidence):
        return GroundingReasonCode.FATAL_PROVENANCE
    return None


def _abstention_text(reason: GroundingReasonCode) -> str:
    if reason == GroundingReasonCode.ENTITY_NOT_FOUND:
        return "The requested entity could not be resolved in the indexed research corpus."
    if reason == GroundingReasonCode.ENTITY_AMBIGUOUS:
        return "The requested entity is ambiguous. Please select one of the available candidates."
    if reason == GroundingReasonCode.UNSUPPORTED_OPERATION:
        return "The current graph retrieval capabilities do not support this question reliably."
    if reason == GroundingReasonCode.RETRIEVAL_FAILED:
        return "The retrieval system could not obtain reliable evidence for this question."
    if reason == GroundingReasonCode.CITATION_VALIDATION_FAILED:
        return "The generated response could not be verified against the retrieved evidence."
    return "The available evidence is insufficient to answer this question reliably."


def _citation_diagnostics(citation_validation: CitationValidationResult | None) -> dict[str, int | None]:
    if citation_validation is None:
        return {
            "trusted_citation_count": 0,
            "markers_found": None,
            "valid_marker_count": None,
            "invalid_marker_count": None,
        }
    return {
        "trusted_citation_count": len(citation_validation.citations),
        "markers_found": len(citation_validation.valid_markers) + len(citation_validation.invalid_markers),
        "valid_marker_count": len(citation_validation.valid_markers),
        "invalid_marker_count": len(citation_validation.invalid_markers),
    }


def _has_nonfatal_provenance_gap(citations: list[AnswerCitation]) -> bool:
    return any(citation.provenance_complete is False for citation in citations)


def _has_fatal_provenance(evidence: EvidenceItem) -> bool:
    if evidence.metadata.get("fatal_provenance") is True:
        return True
    if evidence.provenance is None:
        return False
    return any("fatal" in warning.lower() for warning in evidence.provenance.warnings)


def _limited_evidence_coverage(assessment: EvidenceAssessment | None) -> bool:
    if assessment is None:
        return True
    if assessment.coverage.value != "complete":
        return True
    return False


def _strong_support_requirements_met(
    *,
    analysis_intent: str | None,
    evidence_assessment: EvidenceAssessment | None,
    validation_status: CitationValidationStatus | None,
    trusted_citations: list[AnswerCitation],
) -> bool:
    if analysis_intent is None:
        return True
    if evidence_assessment is None or evidence_assessment.sufficient is not True:
        return False
    if evidence_assessment.coverage.value != "complete":
        return False
    if evidence_assessment.missing_information:
        return False
    if validation_status != CitationValidationStatus.VALID:
        return False
    if not trusted_citations or _has_nonfatal_provenance_gap(trusted_citations):
        return False

    cited_types = {citation.evidence_type for citation in trusted_citations}
    has_text = EvidenceType.TEXT in cited_types
    has_graph = bool(cited_types & {EvidenceType.GRAPH_RELATIONSHIP, EvidenceType.GRAPH_PATH})
    has_path = EvidenceType.GRAPH_PATH in cited_types
    intent = analysis_intent.lower()

    if intent == "semantic_explanation":
        return evidence_assessment.semantic_coverage is True and has_text
    if intent == "mixed_semantic_structural":
        return (
            evidence_assessment.semantic_coverage is True
            and evidence_assessment.structural_coverage is True
            and has_text
            and has_graph
        )
    if intent in {"datasets_from_citing_papers", "methods_for_dataset", "citation_neighborhood"}:
        return evidence_assessment.structural_coverage is True and has_path
    if intent.startswith("shared_"):
        return evidence_assessment.structural_coverage is True and has_graph
    return evidence_assessment.structural_coverage is True and has_graph


def _evidence_summary(evidence: list[EvidenceItem]) -> dict[str, Any]:
    return {
        "evidence_count": len(evidence),
        "text_evidence_count": sum(1 for item in evidence if item.evidence_type == EvidenceType.TEXT),
        "graph_evidence_count": sum(
            1
            for item in evidence
            if item.evidence_type in {EvidenceType.GRAPH_RELATIONSHIP, EvidenceType.GRAPH_PATH}
        ),
        "metadata_evidence_count": sum(1 for item in evidence if item.evidence_type == EvidenceType.METADATA),
    }


def _min_confidence(current: ConfidenceLevel, cap: ConfidenceLevel) -> ConfidenceLevel:
    order = {
        ConfidenceLevel.HIGH: 3,
        ConfidenceLevel.MEDIUM: 2,
        ConfidenceLevel.LOW: 1,
        ConfidenceLevel.INSUFFICIENT_EVIDENCE: 0,
    }
    return current if order[current] <= order[cap] else cap


def _ordered_reason_codes(reason_codes: list[GroundingReasonCode]) -> list[GroundingReasonCode]:
    seen: set[GroundingReasonCode] = set()
    ordered: list[GroundingReasonCode] = []
    for code in reason_codes:
        if code in seen:
            continue
        seen.add(code)
        ordered.append(code)
    return ordered
