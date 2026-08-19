"""Pure matching and metric functions for retrieval evaluation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

from app.domain.evidence import EvidenceItem

from evaluation.models import (
    AggregateMetrics,
    ExpectedTarget,
    RankingMetrics,
    StructuralMetrics,
    TargetType,
)


def path_signature_from_metadata(evidence: EvidenceItem) -> list[str]:
    raw_path = evidence.metadata.get("path")
    if isinstance(raw_path, list):
        return [str(item) for item in raw_path]
    ordered_entities = evidence.metadata.get("ordered_entity_ids")
    ordered_relationships = evidence.metadata.get("ordered_relationship_ids")
    if isinstance(ordered_entities, list) and isinstance(ordered_relationships, list):
        signature: list[str] = []
        for index, entity_id in enumerate(ordered_entities):
            signature.append(str(entity_id))
            if index < len(ordered_relationships):
                signature.append(str(ordered_relationships[index]))
        return signature
    path_nodes = evidence.metadata.get("path_node_ids")
    path_relationships = evidence.metadata.get("path_relationship_ids")
    if isinstance(path_nodes, list) and isinstance(path_relationships, list):
        signature: list[str] = []
        for index, node_id in enumerate(path_nodes):
            signature.append(str(node_id))
            if index < len(path_relationships):
                signature.append(str(path_relationships[index]))
        return signature
    return [*evidence.entity_ids, *evidence.relationship_ids]


def endpoint_id(evidence: EvidenceItem) -> str | None:
    ordered_entities = evidence.metadata.get("ordered_entity_ids")
    if isinstance(ordered_entities, list) and ordered_entities:
        return str(ordered_entities[-1])
    path_node_ids = evidence.metadata.get("path_node_ids")
    if isinstance(path_node_ids, list) and path_node_ids:
        return str(path_node_ids[-1])
    if evidence.entity_ids:
        return evidence.entity_ids[-1]
    return None


def matching_target_ids(evidence: EvidenceItem, expected_targets: Sequence[ExpectedTarget]) -> list[str]:
    matched: list[str] = []
    signature = path_signature_from_metadata(evidence)
    for target in expected_targets:
        if target.target_type == TargetType.CHUNK and evidence.chunk_id == target.target_id:
            matched.append(target.target_id)
        elif target.target_type == TargetType.PAPER and evidence.paper_id == target.target_id:
            matched.append(target.target_id)
        elif target.target_type == TargetType.ENTITY and target.target_id in evidence.entity_ids:
            matched.append(target.target_id)
        elif target.target_type == TargetType.RELATIONSHIP and target.target_id in evidence.relationship_ids:
            matched.append(target.target_id)
        elif target.target_type == TargetType.PATH and target.path_signature == signature:
            matched.append(target.target_id)
    return matched


def compute_ranking_metrics(
    retrieved: Sequence[EvidenceItem],
    expected_targets: Sequence[ExpectedTarget],
    *,
    k_values: Sequence[int],
) -> tuple[RankingMetrics, int, list[list[str]]]:
    expected_ids = {target.target_id for target in expected_targets}
    matched_by_rank = [matching_target_ids(item, expected_targets) for item in retrieved]
    unique_relevant: set[str] = set()
    first_relevant_rank: int | None = None

    for rank, matches in enumerate(matched_by_rank, start=1):
        if matches and first_relevant_rank is None:
            first_relevant_rank = rank
        unique_relevant.update(matches)

    metrics = RankingMetrics(mrr=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank)
    first_match_positions = _first_match_positions(matched_by_rank)
    for k in k_values:
        top_matches = matched_by_rank[:k]
        relevant_targets = {target_id for matches in top_matches for target_id in matches}
        relevant_positions = sum(1 for position in first_match_positions.values() if position <= k)
        returned_count = min(len(retrieved), k)
        metrics.hit_at_k[k] = 1.0 if relevant_targets else 0.0
        metrics.precision_at_k[k] = 0.0 if returned_count == 0 else relevant_positions / returned_count
        metrics.recall_at_k[k] = 0.0 if not expected_ids else len(relevant_targets) / len(expected_ids)
        metrics.ndcg_at_k[k] = _binary_ndcg(
            [
                1
                if any(first_match_positions.get(target_id) == rank for target_id in matches)
                else 0
                for rank, matches in enumerate(top_matches, start=1)
            ],
            min(k, len(expected_ids)),
        )

    return metrics, len(unique_relevant), matched_by_rank


def compute_structural_metrics(
    retrieved: Sequence[EvidenceItem],
    expected_targets: Sequence[ExpectedTarget],
) -> StructuralMetrics:
    provenance_count = sum(
        1 for item in retrieved if item.provenance is not None and item.provenance.provenance_complete
    )
    provenance_rate = 0.0 if not retrieved else provenance_count / len(retrieved)
    metrics = StructuralMetrics(provenance_completeness=provenance_rate)

    entity_targets = [target for target in expected_targets if target.target_type == TargetType.ENTITY]
    relationship_targets = [target for target in expected_targets if target.target_type == TargetType.RELATIONSHIP]
    path_targets = [target for target in expected_targets if target.target_type == TargetType.PATH]

    if entity_targets:
        found = {
            target.target_id
            for item in retrieved
            for target in entity_targets
            if target.target_id in item.entity_ids
        }
        metrics.entity_recall = len(found) / len(entity_targets)
    if relationship_targets:
        found = {
            target.target_id
            for item in retrieved
            for target in relationship_targets
            if target.target_id in item.relationship_ids
        }
        metrics.relationship_recall = len(found) / len(relationship_targets)
    if path_targets:
        exact = {
            target.target_id
            for item in retrieved
            for target in path_targets
            if target.path_signature == path_signature_from_metadata(item)
        }
        endpoints = {
            target.target_id
            for item in retrieved
            for target in path_targets
            if target.target_id == endpoint_id(item)
        }
        metrics.path_exact_match = len(exact) / len(path_targets)
        metrics.endpoint_accuracy = len(endpoints | exact) / len(path_targets)

    return metrics


def aggregate_strategy_results(results: Iterable[Any], *, k_values: Sequence[int]) -> AggregateMetrics:
    applicable = [result for result in results if result.applicable]
    not_applicable = [result for result in results if not result.applicable]
    aggregate = AggregateMetrics(
        applicable_cases=len(applicable),
        not_applicable_cases=len(not_applicable),
    )
    if not applicable:
        return aggregate

    for k in k_values:
        aggregate.hit_at_k[k] = statistics.fmean(result.metrics.hit_at_k.get(k, 0.0) for result in applicable)
        aggregate.precision_at_k[k] = statistics.fmean(
            result.metrics.precision_at_k.get(k, 0.0) for result in applicable
        )
        aggregate.recall_at_k[k] = statistics.fmean(
            result.metrics.recall_at_k.get(k, 0.0) for result in applicable
        )
    aggregate.mrr = statistics.fmean(result.metrics.mrr for result in applicable)
    latencies = sorted(float(result.latency_ms) for result in applicable)
    aggregate.mean_latency_ms = statistics.fmean(latencies)
    aggregate.median_latency_ms = statistics.median(latencies)
    aggregate.p95_latency_ms = percentile(latencies, 95)
    return aggregate


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (pct / 100)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _first_match_positions(matched_by_rank: Sequence[list[str]]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for rank, matches in enumerate(matched_by_rank, start=1):
        for target_id in matches:
            positions.setdefault(target_id, rank)
    return positions


def _binary_ndcg(relevance: Sequence[int], ideal_relevant_count: int) -> float:
    if ideal_relevant_count <= 0:
        return 0.0
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevance))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(relevance), ideal_relevant_count)))
    return 0.0 if ideal == 0.0 else dcg / ideal
