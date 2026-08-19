import math

import pytest

from app.domain.enums import EvidenceScoreKind, EvidenceSourceStore, EvidenceType
from app.domain.evidence import EvidenceItem, EvidenceProvenance
from evaluation.metrics import (
    aggregate_strategy_results,
    compute_ranking_metrics,
    compute_structural_metrics,
    matching_target_ids,
)
from evaluation.models import ExpectedTarget, TargetType


def _text(chunk_id: str, *, paper_id: str = "paper:a") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"evidence:{chunk_id}",
        evidence_type=EvidenceType.TEXT,
        paper_id=paper_id,
        paper_version_id=f"{paper_id}:v1",
        chunk_id=chunk_id,
        section_id="section:intro",
        section_type="introduction",
        text=f"text {chunk_id}",
        score=0.8,
        score_kind=EvidenceScoreKind.VECTOR_SIMILARITY,
        source="qdrant",
        source_store=EvidenceSourceStore.QDRANT,
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.QDRANT,
            paper_id=paper_id,
            paper_version_id=f"{paper_id}:v1",
            chunk_ids=[chunk_id],
        ),
    )


def _graph_path() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="evidence:path",
        evidence_type=EvidenceType.GRAPH_PATH,
        paper_id="paper:a",
        entity_ids=["paper:a", "paper:b", "entity:dataset:x"],
        relationship_ids=["rel:cites", "rel:dataset"],
        source_chunk_ids=["chunk:support"],
        text="path",
        score=0.9,
        score_kind=EvidenceScoreKind.GRAPH_PATH_CONFIDENCE,
        source="neo4j",
        source_store=EvidenceSourceStore.NEO4J,
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.NEO4J,
            paper_id="paper:a",
            chunk_ids=["chunk:support"],
            relationship_ids=["rel:cites", "rel:dataset"],
        ),
        metadata={
            "ordered_entity_ids": ["paper:a", "paper:b", "entity:dataset:x"],
            "ordered_relationship_ids": ["rel:cites", "rel:dataset"],
        },
    )


def test_hit_precision_recall_mrr_and_ndcg_are_hand_checkable() -> None:
    retrieved = [_text("chunk:noise"), _text("chunk:target"), _text("chunk:other")]
    expected = [ExpectedTarget(target_type=TargetType.CHUNK, target_id="chunk:target")]

    metrics, relevant_count, _matches = compute_ranking_metrics(retrieved, expected, k_values=[1, 3])

    assert relevant_count == 1
    assert metrics.hit_at_k[1] == 0.0
    assert metrics.hit_at_k[3] == 1.0
    assert metrics.precision_at_k[3] == pytest.approx(1 / 3)
    assert metrics.recall_at_k[3] == 1.0
    assert metrics.mrr == pytest.approx(1 / 2)
    assert metrics.ndcg_at_k[3] == pytest.approx(1 / math.log2(3))


def test_empty_results_return_zero_metrics() -> None:
    expected = [ExpectedTarget(target_type=TargetType.CHUNK, target_id="chunk:target")]

    metrics, relevant_count, _matches = compute_ranking_metrics([], expected, k_values=[5])

    assert relevant_count == 0
    assert metrics.hit_at_k[5] == 0.0
    assert metrics.precision_at_k[5] == 0.0
    assert metrics.recall_at_k[5] == 0.0
    assert metrics.mrr == 0.0


def test_duplicate_evidence_does_not_inflate_relevant_target_count() -> None:
    expected = [ExpectedTarget(target_type=TargetType.CHUNK, target_id="chunk:target")]
    retrieved = [_text("chunk:target"), _text("chunk:target")]

    metrics, relevant_count, _matches = compute_ranking_metrics(retrieved, expected, k_values=[2])

    assert relevant_count == 1
    assert metrics.recall_at_k[2] == 1.0
    assert metrics.precision_at_k[2] == 0.5


def test_relationship_entity_and_paper_matching_use_evidence_fields() -> None:
    path = _graph_path()

    assert matching_target_ids(path, [ExpectedTarget(target_type=TargetType.ENTITY, target_id="entity:dataset:x")])
    assert matching_target_ids(path, [ExpectedTarget(target_type=TargetType.RELATIONSHIP, target_id="rel:dataset")])
    assert matching_target_ids(_text("chunk:a", paper_id="paper:z"), [ExpectedTarget(target_type=TargetType.PAPER, target_id="paper:z")])


def test_path_exact_match_endpoint_accuracy_and_wrong_endpoint() -> None:
    path = _graph_path()
    exact_target = ExpectedTarget(
        target_type=TargetType.PATH,
        target_id="entity:dataset:x",
        path_signature=["paper:a", "rel:cites", "paper:b", "rel:dataset", "entity:dataset:x"],
    )
    wrong_path_same_endpoint = ExpectedTarget(
        target_type=TargetType.PATH,
        target_id="entity:dataset:x",
        path_signature=["paper:a", "rel:other", "paper:c", "rel:dataset", "entity:dataset:x"],
    )
    wrong_endpoint = ExpectedTarget(
        target_type=TargetType.PATH,
        target_id="entity:dataset:y",
        path_signature=["paper:a", "rel:cites", "paper:b", "rel:dataset", "entity:dataset:y"],
    )

    assert compute_structural_metrics([path], [exact_target]).path_exact_match == 1.0
    same_endpoint = compute_structural_metrics([path], [wrong_path_same_endpoint])
    assert same_endpoint.path_exact_match == 0.0
    assert same_endpoint.endpoint_accuracy == 1.0
    assert compute_structural_metrics([path], [wrong_endpoint]).endpoint_accuracy == 0.0


def test_aggregate_counts_only_applicable_results() -> None:
    from app.domain.enums import RetrievalStrategy
    from evaluation.models import EvaluationFailureCategory, StrategyCaseResult

    applicable = StrategyCaseResult(strategy=RetrievalStrategy.GRAPH, applicable=True)
    applicable.metrics.hit_at_k[5] = 1.0
    applicable.metrics.precision_at_k[5] = 0.5
    applicable.metrics.recall_at_k[5] = 1.0
    not_applicable = StrategyCaseResult(
        strategy=RetrievalStrategy.GRAPH,
        applicable=False,
        failure_reason=EvaluationFailureCategory.GRAPH_NOT_APPLICABLE,
    )

    aggregate = aggregate_strategy_results([applicable, not_applicable], k_values=[5])

    assert aggregate.applicable_cases == 1
    assert aggregate.not_applicable_cases == 1
    assert aggregate.hit_at_k[5] == 1.0
