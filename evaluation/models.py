"""Models for deterministic retrieval evaluation.

The benchmark is intentionally explicit: graph cases name the graph
operation to run, and relevance is expressed as stable IDs from controlled
fixtures rather than judgments inferred from retriever output.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.enums import EntityType, RetrievalStrategy
from app.domain.ids import normalize_whitespace
from app.retrieval.graph_search import GraphSearchOperation


class RetrievalEvaluationCategory(str, Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    SHARED_ENTITY = "shared_entity"
    MULTI_HOP = "multi_hop"
    MIXED = "mixed"


class TargetType(str, Enum):
    CHUNK = "chunk"
    PAPER = "paper"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    PATH = "path"


class EvaluationFailureCategory(str, Enum):
    NO_RESULTS = "NO_RESULTS"
    RELEVANT_NOT_IN_TOP_K = "RELEVANT_NOT_IN_TOP_K"
    ENTITY_NOT_FOUND = "ENTITY_NOT_FOUND"
    GRAPH_NOT_APPLICABLE = "GRAPH_NOT_APPLICABLE"
    PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"


class BenchmarkGraphRequest(BaseModel):
    operation: GraphSearchOperation
    entity_id: str | None = None
    entity_type: EntityType | None = None
    canonical_name: str | None = None
    depth: int | None = None
    limit: int | None = None

    @model_validator(mode="after")
    def _has_entity_lookup(self) -> "BenchmarkGraphRequest":
        if not self.entity_id and not self.canonical_name:
            raise ValueError("graph_request requires entity_id or canonical_name")
        return self


class ExpectedTarget(BaseModel):
    target_type: TargetType
    target_id: str
    relevance: int = Field(default=1, ge=1)
    path_signature: list[str] | None = None

    @field_validator("target_id")
    @classmethod
    def _target_id_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("target_id must not be blank")
        return normalized

    @model_validator(mode="after")
    def _path_signature_required_for_path(self) -> "ExpectedTarget":
        if self.target_type == TargetType.PATH and not self.path_signature:
            raise ValueError("path targets require path_signature")
        return self


class RetrievalBenchmarkCase(BaseModel):
    id: str
    category: RetrievalEvaluationCategory
    question: str
    graph_request: BenchmarkGraphRequest | None = None
    expected_targets: list[ExpectedTarget]
    relevance_target_type: TargetType
    notes: str | None = None

    @field_validator("id", "question")
    @classmethod
    def _required_text(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("id/question must not be blank")
        return normalized

    @model_validator(mode="after")
    def _case_is_consistent(self) -> "RetrievalBenchmarkCase":
        if not self.expected_targets:
            raise ValueError("expected_targets must not be empty")
        for target in self.expected_targets:
            if target.target_type != self.relevance_target_type:
                raise ValueError("all expected targets must match relevance_target_type")
        return self


class RetrievalBenchmarkDataset(BaseModel):
    benchmark_version: str = "v1"
    cases: list[RetrievalBenchmarkCase]

    @model_validator(mode="after")
    def _unique_case_ids(self) -> "RetrievalBenchmarkDataset":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                duplicates.add(case.id)
            seen.add(case.id)
        if duplicates:
            raise ValueError(f"duplicate benchmark case ids: {sorted(duplicates)}")
        return self


class RankingMetrics(BaseModel):
    hit_at_k: dict[int, float] = Field(default_factory=dict)
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)


class StructuralMetrics(BaseModel):
    entity_recall: float | None = None
    relationship_recall: float | None = None
    path_exact_match: float | None = None
    endpoint_accuracy: float | None = None
    provenance_completeness: float = 0.0


class RetrievedEvidenceRecord(BaseModel):
    evidence_id: str
    evidence_type: str
    matched_target_ids: list[str] = Field(default_factory=list)
    rank: int
    branch_ranks: dict[str, int] = Field(default_factory=dict)
    fusion_score: float | None = None
    cross_store_supported: bool = False


class StrategyCaseResult(BaseModel):
    strategy: RetrievalStrategy
    applicable: bool
    retrieved_count: int = 0
    relevant_count: int = 0
    metrics: RankingMetrics = Field(default_factory=RankingMetrics)
    structural_metrics: StructuralMetrics = Field(default_factory=StructuralMetrics)
    latency_ms: float = 0.0
    failure_reason: EvaluationFailureCategory | None = None
    evidence: list[RetrievedEvidenceRecord] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RetrievalEvaluationCaseResult(BaseModel):
    case_id: str
    category: RetrievalEvaluationCategory
    strategy_results: dict[str, StrategyCaseResult]
    comparison: str


class AggregateMetrics(BaseModel):
    applicable_cases: int = 0
    not_applicable_cases: int = 0
    hit_at_k: dict[int, float] = Field(default_factory=dict)
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


class RetrievalEvaluationReport(BaseModel):
    benchmark_version: str
    benchmark_checksum: str
    run_fingerprint: str
    k_values: list[int]
    generated_at: str
    case_results: list[RetrievalEvaluationCaseResult]
    overall: dict[str, AggregateMetrics]
    by_category: dict[str, dict[str, AggregateMetrics]]
    hybrid_wins: list[str] = Field(default_factory=list)
    vector_wins: list[str] = Field(default_factory=list)
    graph_wins: list[str] = Field(default_factory=list)
    all_failures: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

