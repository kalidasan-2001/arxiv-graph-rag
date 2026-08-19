from app.core.config import Settings
from app.domain.enums import ConfidenceLevel, EvidenceSourceStore, EvidenceType
from app.domain.evidence import EvidenceItem, EvidenceProvenance, build_evidence_pool
from app.generation.answer import AnswerContextBuilder, GeneratedGroundedAnswer
from app.generation.citations import CitationValidationResult, CitationValidator
from app.generation.grounding import FinalAnswerStatus, GroundingDecisionService
from app.retrieval.critic import EvidenceAssessment, EvidenceCoverage, RefinementType


def test_high_confidence_structural_answer() -> None:
    final = _decide(
        "Which datasets does Paper A evaluate on?",
        [_graph()],
        assessment=_assessment(structural_coverage=True),
        analysis_intent="paper_datasets",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.HIGH
    assert final.grounding.allow_answer is True
    assert [code.value for code in final.grounding.reason_codes] == ["strong_grounded_support"]


def test_high_confidence_semantic_answer() -> None:
    final = _decide(
        "Explain Paper A's methodology.",
        [_text()],
        assessment=_assessment(semantic_coverage=True),
        analysis_intent="semantic_explanation",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.HIGH
    assert final.citations[0].evidence_type == EvidenceType.TEXT


def test_high_confidence_multi_hop_graph_path_answer() -> None:
    final = _decide(
        "Which datasets are used by papers citing Paper A?",
        [_path()],
        assessment=_assessment(structural_coverage=True),
        analysis_intent="datasets_from_citing_papers",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.HIGH
    assert final.citations[0].evidence_type == EvidenceType.GRAPH_PATH


def test_mixed_answer_can_be_high_when_text_and_graph_are_cited() -> None:
    assessment = _assessment(semantic_coverage=True, structural_coverage=True)
    final = _decide(
        "Explain Paper A and list datasets.",
        [_text("chunk:mixed"), _graph("rel:mixed")],
        answer="Text support [E1]. Dataset support [E2].",
        assessment=assessment,
        analysis_intent="mixed_semantic_structural",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.HIGH
    assert len(final.citations) == 2


def test_shared_entity_exact_graph_answer_can_be_high() -> None:
    final = _decide(
        "Which papers share the same dataset as Paper A?",
        [_path()],
        assessment=_assessment(structural_coverage=True),
        analysis_intent="shared_datasets",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.HIGH


def test_semantic_weak_coverage_caps_confidence_at_medium() -> None:
    final = _decide(
        "Explain Paper A.",
        [_text()],
        assessment=_assessment(),
        analysis_intent="semantic_explanation",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.MEDIUM
    assert "limited_evidence_coverage" in [code.value for code in final.grounding.reason_codes]


def test_mixed_missing_semantic_citation_caps_confidence_at_medium() -> None:
    final = _decide(
        "Explain Paper A and list datasets.",
        [_graph()],
        assessment=_assessment(semantic_coverage=True, structural_coverage=True),
        analysis_intent="mixed_semantic_structural",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.MEDIUM


def test_mixed_missing_graph_citation_caps_confidence_at_medium() -> None:
    final = _decide(
        "Explain Paper A and list datasets.",
        [_text()],
        assessment=_assessment(semantic_coverage=True, structural_coverage=True),
        analysis_intent="mixed_semantic_structural",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.MEDIUM


def test_valid_irrelevant_semantic_citation_does_not_make_high() -> None:
    final = _decide(
        "Explain a missing dataset claim.",
        [_text()],
        assessment=_assessment(),
        analysis_intent="semantic_explanation",
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.MEDIUM


def test_mixed_only_semantic_assessment_abstains() -> None:
    final = _decide(
        "Explain Paper A and list datasets.",
        [_text()],
        assessment=_assessment(sufficient=False, semantic_coverage=True, structural_coverage=False),
        analysis_intent="mixed_semantic_structural",
    )

    assert final.status == FinalAnswerStatus.ABSTAINED
    assert final.confidence == ConfidenceLevel.INSUFFICIENT_EVIDENCE


def test_mixed_only_structural_assessment_abstains() -> None:
    final = _decide(
        "Explain Paper A and list datasets.",
        [_graph()],
        assessment=_assessment(sufficient=False, semantic_coverage=False, structural_coverage=True),
        analysis_intent="mixed_semantic_structural",
    )

    assert final.status == FinalAnswerStatus.ABSTAINED
    assert final.confidence == ConfidenceLevel.INSUFFICIENT_EVIDENCE


def test_partial_citation_validation_caps_confidence_at_medium() -> None:
    final = _decide("Explain Paper A.", [_text()], answer="Supported [E1]. Unsupported [E999].")

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.answer == "Supported [1]. Unsupported."
    assert final.confidence == ConfidenceLevel.MEDIUM
    assert "partial_citation_validation" in [code.value for code in final.grounding.reason_codes]
    assert final.grounding.diagnostics["invalid_marker_count"] == 1


def test_nonfatal_provenance_gap_caps_confidence_at_medium() -> None:
    final = _decide("Explain Paper A.", [_text(provenance_complete=False)])

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.MEDIUM
    assert "provenance_incomplete" in [code.value for code in final.grounding.reason_codes]


def test_missing_information_caps_confidence_at_medium() -> None:
    final = _decide(
        "Explain Paper A.",
        [_text()],
        assessment=_assessment(missing_information=["evaluation detail"]),
    )

    assert final.status == FinalAnswerStatus.ANSWERED
    assert final.confidence == ConfidenceLevel.MEDIUM
    assert "missing_information" in [code.value for code in final.grounding.reason_codes]


def test_zero_trusted_citations_abstains() -> None:
    validation = CitationValidationResult(
        validated_text="Supported.",
        citations=[],
        valid_markers=[],
        invalid_markers=[],
        validation_status="valid",
        validation_fingerprint="manual-zero",
    )
    final = GroundingDecisionService(settings=Settings(_env_file=None)).decide(
        query="Explain Paper A.",
        internal_status="SUCCESS",
        evidence=[_text()],
        evidence_assessment=_assessment(),
        citation_validation=validation,
        validated_answer=None,
        retrieval_round=1,
        warnings=[],
    )

    assert final.status == FinalAnswerStatus.ABSTAINED
    assert final.confidence == ConfidenceLevel.INSUFFICIENT_EVIDENCE
    assert [code.value for code in final.grounding.reason_codes] == ["no_trusted_citations"]


def test_no_markers_abstains_with_citation_template() -> None:
    final = _decide("Explain Paper A.", [_text()], answer="Supported without citations.")

    assert final.status == FinalAnswerStatus.ABSTAINED
    assert final.answer == "The generated response could not be verified against the retrieved evidence."
    assert final.confidence == ConfidenceLevel.INSUFFICIENT_EVIDENCE
    assert final.citations == []


def test_insufficient_retrieval_abstains() -> None:
    final = GroundingDecisionService(settings=Settings(_env_file=None)).decide(
        query="Explain Paper A.",
        internal_status="INSUFFICIENT_EVIDENCE",
        evidence=[_text()],
        evidence_assessment=_assessment(sufficient=False),
        citation_validation=None,
        validated_answer=None,
        retrieval_round=2,
        warnings=[],
    )

    assert final.status == FinalAnswerStatus.ABSTAINED
    assert final.answer == "The available evidence is insufficient to answer this question reliably."
    assert [code.value for code in final.grounding.reason_codes] == ["insufficient_retrieval_evidence"]


def test_ambiguous_entity_preserves_specific_final_status() -> None:
    final = GroundingDecisionService(settings=Settings(_env_file=None)).decide(
        query="Which Paper A?",
        internal_status="REQUIRES_DISAMBIGUATION",
        evidence=[],
        evidence_assessment=None,
        citation_validation=None,
        validated_answer=None,
        retrieval_round=0,
        warnings=[],
    )

    assert final.status == FinalAnswerStatus.REQUIRES_DISAMBIGUATION
    assert final.confidence == ConfidenceLevel.INSUFFICIENT_EVIDENCE
    assert final.answer == "The requested entity is ambiguous. Please select one of the available candidates."


def test_deterministic_reason_order_and_fingerprint() -> None:
    evidence = [_text(provenance_complete=False)]
    assessment = _assessment(missing_information=["detail"])
    first = _decide("Explain Paper A.", evidence, assessment=assessment)
    second = _decide("Explain Paper A.", evidence, assessment=assessment)
    changed = _decide("Explain Paper A.", evidence, assessment=assessment, answer="Changed [E1].")

    assert [code.value for code in first.grounding.reason_codes] == [
        "provenance_incomplete",
        "missing_information",
    ]
    assert first.grounding.grounding_fingerprint == second.grounding.grounding_fingerprint
    assert first.grounding.grounding_fingerprint != changed.grounding.grounding_fingerprint
    assert first.model_dump(mode="json")["grounding"]["allow_answer"] is True


def _decide(
    query: str,
    evidence: list[EvidenceItem],
    *,
    answer: str = "Supported answer [E1].",
    assessment: EvidenceAssessment | None = None,
    analysis_intent: str | None = None,
) -> object:
    settings = Settings(_env_file=None)
    pool = build_evidence_pool(evidence)
    context = AnswerContextBuilder(settings=settings).build(
        query=query,
        analysis=None,
        evidence_pool=pool,
        generation_config_fingerprint="generation-fp",
    )
    validated = CitationValidator(settings=settings).validate(
        generated_answer=GeneratedGroundedAnswer(text=answer),
        evidence_pool=pool,
        answer_context=context,
    )
    internal_status = "SUCCESS"
    if validated.citation_validation.validation_status.value in {"invalid", "no_citations"}:
        internal_status = "CITATION_VALIDATION_FAILED"
    return GroundingDecisionService(settings=settings).decide(
        query=query,
        internal_status=internal_status,
        evidence=evidence,
        evidence_assessment=assessment or _assessment(),
        citation_validation=validated.citation_validation,
        validated_answer=validated,
        retrieval_round=1,
        warnings=validated.citation_validation.warnings,
        analysis_intent=analysis_intent,
    )


def _assessment(
    *,
    sufficient: bool = True,
    missing_information: list[str] | None = None,
    semantic_coverage: bool | None = None,
    structural_coverage: bool | None = None,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=sufficient,
        coverage=EvidenceCoverage.COMPLETE if sufficient else EvidenceCoverage.INSUFFICIENT,
        missing_information=missing_information or [],
        recommended_refinement_type=RefinementType.NONE,
        semantic_coverage=semantic_coverage,
        structural_coverage=structural_coverage,
        deterministic=True,
    )


def _text(chunk_id: str = "chunk:1", *, provenance_complete: bool = True) -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.TEXT,
        source="qdrant",
        chunk_id=chunk_id,
        text=f"Evidence for {chunk_id}",
        provenance=_provenance(chunk_ids=[chunk_id], complete=provenance_complete),
    )


def _graph(relationship_id: str = "rel:1", *, provenance_complete: bool = True) -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.GRAPH_RELATIONSHIP,
        source="neo4j",
        entity_ids=["paper:arxiv:a", "entity:dataset:x"],
        relationship_ids=[relationship_id],
        source_chunk_ids=["chunk:graph"],
        provenance=_provenance(
            chunk_ids=["chunk:graph"],
            relationship_ids=[relationship_id],
            source_store=EvidenceSourceStore.NEO4J,
            complete=provenance_complete,
        ),
    )


def _path() -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.GRAPH_PATH,
        source="neo4j",
        entity_ids=["paper:arxiv:a", "paper:arxiv:b", "entity:dataset:x"],
        relationship_ids=["rel:cites", "rel:dataset"],
        source_chunk_ids=["chunk:path"],
        provenance=_provenance(
            chunk_ids=["chunk:path"],
            relationship_ids=["rel:cites", "rel:dataset"],
            source_store=EvidenceSourceStore.NEO4J,
        ),
    )


def _provenance(
    *,
    chunk_ids: list[str],
    relationship_ids: list[str] | None = None,
    source_store: EvidenceSourceStore = EvidenceSourceStore.QDRANT,
    complete: bool = True,
) -> EvidenceProvenance:
    return EvidenceProvenance(
        provenance_type="chunk",
        source_store=source_store,
        chunk_ids=chunk_ids,
        relationship_ids=relationship_ids or [],
        provenance_complete=complete,
        warnings=[] if complete else ["supporting chunk unavailable"],
    )
