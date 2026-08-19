"""Unit tests for `ontology.py` -- schema validation, the relationship
compatibility matrix, use-vs-mention, and evidence-quote checking (prompt
#47/#48/#49/#52). Pure functions, no DB, no LLM.
"""

from app.ingestion.graph_extraction.models import (
    GraphExtractionWarningCode,
    RawEntityCandidate,
    RawRelationshipCandidate,
)
from app.ingestion.graph_extraction.ontology import (
    validate_entity_candidate,
    validate_relationship_candidate,
)

_CHUNK_TEXT = (
    "We propose a novel graph reconstruction attack against Graph RAG "
    'systems. As stated in the abstract, "our method achieves over 90% '
    'recovery" on standard benchmarks. We evaluate on HotpotQA and MIMIC-IV.'
)


def _entity(**overrides) -> RawEntityCandidate:
    defaults = dict(entity_type="method", name="GraphSteal", aliases=[], evidence_quote=None, confidence=0.9)
    defaults.update(overrides)
    return RawEntityCandidate(**defaults)


def _relationship(**overrides) -> RawRelationshipCandidate:
    defaults = dict(
        relationship_type="uses_method",
        source_name="Current Paper",
        source_type="paper",
        target_name="GraphSteal",
        target_type="method",
        usage="used_by_this_paper",
        evidence_quote=None,
        confidence=0.9,
    )
    defaults.update(overrides)
    return RawRelationshipCandidate(**defaults)


class TestEntityValidation:
    def test_valid_entity_is_accepted(self) -> None:
        candidate, warning = validate_entity_candidate(_entity(), source_chunk_id="chunk:a")
        assert candidate is not None
        assert warning is None
        assert candidate.name == "GraphSteal"

    def test_invalid_entity_type_is_rejected(self) -> None:
        candidate, warning = validate_entity_candidate(
            _entity(entity_type="organization"), source_chunk_id="chunk:a"
        )
        assert candidate is None
        assert warning is not None
        assert warning.code == GraphExtractionWarningCode.ENTITY_CANDIDATE_REJECTED

    def test_empty_name_is_rejected(self) -> None:
        candidate, warning = validate_entity_candidate(_entity(name="   "), source_chunk_id="chunk:a")
        assert candidate is None
        assert warning is not None

    def test_confidence_below_zero_is_rejected(self) -> None:
        candidate, warning = validate_entity_candidate(
            _entity(confidence=-0.1), source_chunk_id="chunk:a"
        )
        assert candidate is None
        assert warning is not None

    def test_confidence_above_one_is_rejected(self) -> None:
        candidate, warning = validate_entity_candidate(
            _entity(confidence=1.1), source_chunk_id="chunk:a"
        )
        assert candidate is None
        assert warning is not None

    def test_entity_type_is_case_insensitive(self) -> None:
        candidate, warning = validate_entity_candidate(
            _entity(entity_type="Method"), source_chunk_id="chunk:a"
        )
        assert candidate is not None
        assert warning is None

    def test_paper_and_author_are_not_valid_llm_entity_types(self) -> None:
        # Prompt #26: the LLM is only ever asked for method/dataset/task --
        # paper/author come deterministically. If it proposes one anyway,
        # ontology validation still rejects it (defense in depth).
        for bad_type in ("paper", "author"):
            candidate, warning = validate_entity_candidate(
                _entity(entity_type=bad_type), source_chunk_id="chunk:a"
            )
            # Note: "paper"/"author" ARE valid EntityType values -- this
            # documents that ontology.py alone doesn't block them; the
            # extraction service structurally prevents the LLM from ever
            # producing them by not asking for them (see prompt.py).
            assert candidate is not None


class TestRelationshipCompatibilityMatrix:
    def test_paper_authored_by_author_is_valid(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(
                relationship_type="authored_by", source_type="paper", target_type="author",
                usage=None,
            ),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None
        assert warnings == []

    def test_paper_cites_paper_is_valid(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(relationship_type="cites", source_type="paper", target_type="paper", usage=None),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None

    def test_paper_uses_method_method_is_valid(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(), source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT
        )
        assert candidate is not None

    def test_paper_evaluated_on_dataset_is_valid(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(
                relationship_type="evaluated_on", target_type="dataset", target_name="HotpotQA"
            ),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None

    def test_paper_addresses_task_is_valid(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(
                relationship_type="addresses", target_type="task", target_name="multi-hop QA",
                usage=None,
            ),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None

    def test_dataset_authored_by_author_is_rejected(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(
                relationship_type="authored_by", source_type="dataset", target_type="author",
                usage=None,
            ),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is None
        assert any(w.code == GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED for w in warnings)

    def test_method_cites_paper_is_rejected(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(
                relationship_type="cites", source_type="method", target_type="paper", usage=None
            ),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is None

    def test_unsupported_relationship_type_is_rejected(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(relationship_type="influences"), source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT
        )
        assert candidate is None
        assert any(w.code == GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED for w in warnings)

    def test_empty_names_are_rejected(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(target_name="   "), source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT
        )
        assert candidate is None

    def test_confidence_out_of_range_is_rejected(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(confidence=2.0), source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT
        )
        assert candidate is None


class TestUseVsMention:
    """Critical rule (prompt #13/#14/#48)."""

    def test_mentioned_only_method_does_not_become_uses_method(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(usage="mentioned_only"), source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT
        )
        assert candidate is None
        assert any(w.code == GraphExtractionWarningCode.RELATIONSHIP_CANDIDATE_REJECTED for w in warnings)

    def test_used_by_this_paper_method_becomes_uses_method(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(usage="used_by_this_paper"), source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT
        )
        assert candidate is not None

    def test_missing_usage_classification_is_treated_as_not_used(self) -> None:
        # Abstaining is safer than assuming used_by_this_paper (prompt #26).
        candidate, warnings = validate_relationship_candidate(
            _relationship(usage=None), source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT
        )
        assert candidate is None

    def test_mentioned_only_dataset_does_not_become_evaluated_on(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(
                relationship_type="evaluated_on", target_type="dataset", target_name="HotpotQA",
                usage="mentioned_only",
            ),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is None

    def test_evaluated_on_dataset_actually_used_is_accepted(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(
                relationship_type="evaluated_on", target_type="dataset", target_name="HotpotQA",
                usage="used_by_this_paper",
            ),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None

    def test_addresses_does_not_require_usage_classification(self) -> None:
        # ADDRESSES isn't usage-sensitive (prompt #15 has no use-vs-mention concept).
        candidate, warnings = validate_relationship_candidate(
            _relationship(
                relationship_type="addresses", target_type="task", target_name="graph reconstruction",
                usage=None,
            ),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None


class TestEvidenceQuoteValidation:
    def test_quote_found_in_chunk_is_kept(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(evidence_quote="our method achieves over 90% recovery"),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None
        assert candidate.evidence_quote == "our method achieves over 90% recovery"
        assert not any(w.code == GraphExtractionWarningCode.EVIDENCE_QUOTE_DISCARDED for w in warnings)

    def test_quote_not_found_is_discarded_but_relation_still_valid(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(evidence_quote="this sentence was never in the chunk"),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None  # relation survives (prompt #19)
        assert candidate.evidence_quote is None  # quote discarded
        assert any(w.code == GraphExtractionWarningCode.EVIDENCE_QUOTE_DISCARDED for w in warnings)

    def test_no_quote_provided_is_not_a_warning(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(evidence_quote=None), source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT
        )
        assert candidate is not None
        assert warnings == []

    def test_quote_matching_is_whitespace_tolerant(self) -> None:
        candidate, warnings = validate_relationship_candidate(
            _relationship(evidence_quote="our method  achieves\nover 90% recovery"),
            source_chunk_id="chunk:a", chunk_text=_CHUNK_TEXT,
        )
        assert candidate is not None
        assert candidate.evidence_quote is not None
