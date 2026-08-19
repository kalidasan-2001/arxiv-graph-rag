from app.core.config import Settings
from app.domain.enums import EvidenceSourceStore, EvidenceType
from app.domain.evidence import EvidenceItem, EvidencePool, EvidencePoolItem, EvidenceProvenance
from app.generation.answer import AnswerContextBuilder, GeneratedGroundedAnswer, answer_generation_config_fingerprint
from app.generation.citations import (
    CitationValidationStatus,
    CitationValidator,
    citation_validation_fingerprint,
)


def _text(chunk_id: str = "chunk:1", *, provenance: EvidenceProvenance | None = None) -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.TEXT,
        source="qdrant",
        chunk_id=chunk_id,
        paper_id="paper:arxiv:a",
        paper_version_id="paper:arxiv:a:v1",
        section_id="section:method",
        section_type="methodology",
        page_start=4,
        page_end=5,
        text=f"Text evidence {chunk_id}.",
        source_store=EvidenceSourceStore.QDRANT,
        provenance=provenance,
    )


def _graph() -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.GRAPH_RELATIONSHIP,
        source="neo4j",
        entity_ids=["paper:arxiv:a", "entity:dataset:x"],
        relationship_ids=["rel:a-x"],
        source_chunk_ids=["chunk:graph"],
        supporting_text_evidence_ids=["evidence:text"],
        source_store=EvidenceSourceStore.NEO4J,
    )


def _path() -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.GRAPH_PATH,
        source="neo4j",
        entity_ids=["paper:arxiv:a", "paper:arxiv:b", "entity:dataset:x"],
        relationship_ids=["rel:b-a", "rel:b-x"],
        source_chunk_ids=["chunk:path"],
        metadata={"nodes": ["Paper A", "Paper B", "Dataset X"], "relationships": ["cites", "evaluated_on"]},
        source_store=EvidenceSourceStore.NEO4J,
    )


def _metadata() -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.METADATA,
        source="metadata",
        entity_ids=["paper:arxiv:a"],
        paper_id="paper:arxiv:a",
        text="Metadata evidence.",
        source_store=EvidenceSourceStore.METADATA,
    )


def _pool(*evidence: EvidenceItem) -> EvidencePool:
    return EvidencePool(
        items=[
            EvidencePoolItem(pool_id=f"E{index}", evidence=item)
            for index, item in enumerate(evidence, start=1)
        ]
    )


def _context(pool: EvidencePool, *, max_items: int = 10):
    settings = Settings(_env_file=None, ANSWER_MAX_EVIDENCE_ITEMS=max_items, ANSWER_MAX_CONTEXT_CHARS=30000)
    fingerprint = answer_generation_config_fingerprint(
        settings=settings,
        provider_name="fake",
        model_name="fake-answer",
        temperature=0.0,
    )
    return AnswerContextBuilder(settings=settings).build(
        query="Question?",
        analysis=None,
        evidence_pool=pool,
        generation_config_fingerprint=fingerprint,
    )


def _validate(text: str, pool: EvidencePool, *, max_items: int = 10):
    settings = Settings(_env_file=None, ANSWER_MAX_EVIDENCE_ITEMS=max_items)
    return CitationValidator(settings=settings).validate(
        generated_answer=GeneratedGroundedAnswer(
            text=text,
            used_evidence_markers=["E999"],
            generation_metadata={"paper_title": "Fake Paper", "page": 999},
        ),
        evidence_pool=pool,
        answer_context=_context(pool, max_items=max_items),
    )


def test_one_valid_citation_is_renumbered_and_trusted() -> None:
    result = _validate("Dataset X [E1].", _pool(_text()))

    assert result.text == "Dataset X [1]."
    assert result.citation_validation.validation_status == CitationValidationStatus.VALID
    assert result.citations[0].citation_number == 1
    assert result.citations[0].evidence_label == "E1"
    assert result.citations[0].chunk_id == "chunk:1"


def test_repeated_valid_citation_reuses_number() -> None:
    result = _validate("Dataset X [E1]. It is discussed again [E1].", _pool(_text()))

    assert result.text == "Dataset X [1]. It is discussed again [1]."
    assert len(result.citations) == 1


def test_multiple_valid_citations_use_first_appearance_numbering() -> None:
    pool = _pool(_text("chunk:1"), _text("chunk:2"), _text("chunk:3"))
    result = _validate("Method X [E3] and Dataset Y [E1].", pool)

    assert result.text == "Method X [1] and Dataset Y [2]."
    assert [citation.evidence_label for citation in result.citations] == ["E3", "E1"]


def test_unknown_malformed_e0_and_range_markers_are_rejected() -> None:
    result = _validate("Bad [E999] [E01] [E0] [E1-E3].", _pool(_text()))

    assert result.citation_validation.validation_status == CitationValidationStatus.INVALID
    assert result.citations == []
    assert result.text == "Bad."
    assert {marker.marker for marker in result.citation_validation.invalid_markers} == {
        "[E999]",
        "[E01]",
        "[E0]",
        "[E1-E3]",
    }


def test_mixed_valid_and_invalid_keeps_valid_and_strips_invalid() -> None:
    result = _validate("Dataset X [E1]. Method Y [E999].", _pool(_text()))

    assert result.citation_validation.validation_status == CitationValidationStatus.PARTIALLY_VALID
    assert result.text == "Dataset X [1]. Method Y."
    assert len(result.citations) == 1
    assert result.citation_validation.invalid_markers[0].reason == "not_in_answer_context"


def test_diagnostic_only_markers_and_llm_citation_lists_are_ignored() -> None:
    pool = _pool(_text())
    result = CitationValidator(settings=Settings(_env_file=None)).validate(
        generated_answer=GeneratedGroundedAnswer(text="Dataset X.", used_evidence_markers=["E1"]),
        evidence_pool=pool,
        answer_context=_context(pool),
    )

    assert result.citation_validation.validation_status == CitationValidationStatus.NO_CITATIONS
    assert result.citations == []


def test_marker_outside_answer_context_is_invalid_even_if_in_pool() -> None:
    result = _validate("Evidence outside context [E3].", _pool(_text("chunk:1"), _text("chunk:2"), _text("chunk:3")), max_items=2)

    assert result.citation_validation.validation_status == CitationValidationStatus.INVALID
    assert result.citation_validation.invalid_markers[0].reason == "not_in_answer_context"


def test_context_pool_mismatch_fails_safely() -> None:
    pool = _pool(_text("chunk:1"))
    context = _context(pool)
    mismatch_pool = _pool(_text("chunk:2"))

    try:
        CitationValidator(settings=Settings(_env_file=None)).validate(
            generated_answer=GeneratedGroundedAnswer(text="Dataset X [E1]."),
            evidence_pool=mismatch_pool,
            answer_context=context,
        )
    except ValueError as exc:
        assert "does not match final evidence pool" in str(exc)
    else:
        raise AssertionError("expected context/pool mismatch to fail")


def test_non_fatal_provenance_incomplete_creates_citation_with_warning() -> None:
    provenance = EvidenceProvenance(
        provenance_type="chunk",
        source_store=EvidenceSourceStore.QDRANT,
        chunk_ids=["chunk:1"],
        provenance_complete=False,
        warnings=["supporting chunk unavailable"],
    )
    result = _validate("Dataset X [E1].", _pool(_text(provenance=provenance)))

    assert result.citation_validation.validation_status == CitationValidationStatus.VALID
    assert result.citations[0].provenance_complete is False
    assert result.citations[0].provenance_warnings == ["supporting chunk unavailable"]


def test_fatal_provenance_rejects_citation() -> None:
    fatal = _text().model_copy(update={"metadata": {"fatal_provenance": True}})
    result = _validate("Dataset X [E1].", _pool(fatal))

    assert result.citation_validation.validation_status == CitationValidationStatus.INVALID
    assert result.citations == []
    assert result.citation_validation.invalid_markers[0].reason == "fatal_provenance"


def test_trusted_metadata_comes_from_evidence_not_model() -> None:
    result = _validate("Dataset X [E1].", _pool(_text()))

    citation = result.citations[0]
    assert citation.paper_id == "paper:arxiv:a"
    assert citation.page_start == 4
    assert citation.page_end == 5
    assert citation.metadata == {}


def test_graph_relationship_and_path_citation_metadata() -> None:
    result = _validate("Graph fact [E1]. Path fact [E2]. Metadata [E3].", _pool(_graph(), _path(), _metadata()))

    relationship, path, metadata = result.citations
    assert relationship.evidence_type == EvidenceType.GRAPH_RELATIONSHIP
    assert relationship.relationship_ids == ["rel:a-x"]
    assert relationship.entity_ids == ["paper:arxiv:a", "entity:dataset:x"]
    assert relationship.supporting_text_evidence_ids == ["evidence:text"]
    assert path.evidence_type == EvidenceType.GRAPH_PATH
    assert path.metadata["nodes"] == ["Paper A", "Paper B", "Dataset X"]
    assert metadata.evidence_type == EvidenceType.METADATA


def test_validation_fingerprint_is_deterministic_and_changes_with_raw_answer() -> None:
    settings = Settings(_env_file=None)
    context = _context(_pool(_text()))
    first = citation_validation_fingerprint(
        settings=settings,
        context_fingerprint=context.context_fingerprint,
        generation_config_fingerprint=context.generation_config_fingerprint,
        raw_answer_text="Dataset X [E1].",
    )
    second = citation_validation_fingerprint(
        settings=settings,
        context_fingerprint=context.context_fingerprint,
        generation_config_fingerprint=context.generation_config_fingerprint,
        raw_answer_text="Dataset Y [E1].",
    )

    assert first != second
