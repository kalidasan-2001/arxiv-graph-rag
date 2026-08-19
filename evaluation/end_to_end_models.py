"""Models for end-to-end answer evaluation across RAG variants."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.enums import ConfidenceLevel, EntityType, RetrievalStrategy
from app.domain.ids import ensure_json_safe, normalize_whitespace
from app.retrieval.graph_search import GraphSearchOperation
from evaluation.models import BenchmarkGraphRequest, ExpectedTarget, RetrievalEvaluationCategory


class EndToEndSystem(str, Enum):
    VECTOR_RAG = "VECTOR_RAG"
    GRAPH_RAG = "GRAPH_RAG"
    HYBRID_RAG = "HYBRID_RAG"
    AGENTIC_HYBRID_RAG = "AGENTIC_HYBRID_RAG"


class EndToEndCategory(str, Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    SHARED_ENTITY = "shared_entity"
    MULTI_HOP = "multi_hop"
    MIXED = "mixed"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"


class ExpectedAnswerFact(BaseModel):
    fact_id: str
    expected_text: str | None = None
    expected_target_id: str | None = None

    @field_validator("fact_id")
    @classmethod
    def _fact_id_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("fact_id must not be blank")
        return normalized


class ExpectedAgenticPlan(BaseModel):
    intent: str | None = None
    strategy: RetrievalStrategy | None = None
    graph_operation: GraphSearchOperation | None = None
    entity_type: EntityType | None = None


class EndToEndBenchmarkCase(BaseModel):
    id: str
    category: EndToEndCategory
    question: str
    graph_request: BenchmarkGraphRequest | None = None
    expected_targets: list[ExpectedTarget] = Field(default_factory=list)
    expected_answer_facts: list[ExpectedAnswerFact] = Field(default_factory=list)
    expected_abstention: bool = False
    expected_disambiguation: bool = False
    expected_agentic_plan: ExpectedAgenticPlan | None = None
    required_citation_targets: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("id", "question")
    @classmethod
    def _required_text(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("id/question must not be blank")
        return normalized

    @model_validator(mode="after")
    def _case_is_consistent(self) -> "EndToEndBenchmarkCase":
        if not self.expected_abstention and not self.expected_disambiguation:
            if not self.expected_targets and not self.expected_answer_facts:
                raise ValueError("answerable cases require expected targets or expected answer facts")
        if self.category == EndToEndCategory.AMBIGUOUS and not self.expected_disambiguation:
            raise ValueError("ambiguous category requires expected_disambiguation=true")
        return self


class EndToEndBenchmark(BaseModel):
    benchmark_version: str = "v1"
    cases: list[EndToEndBenchmarkCase]

    @model_validator(mode="after")
    def _unique_case_ids(self) -> "EndToEndBenchmark":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                duplicates.add(case.id)
            seen.add(case.id)
        if duplicates:
            raise ValueError(f"duplicate benchmark case ids: {sorted(duplicates)}")
        return self


class EndToEndCaseResult(BaseModel):
    case_id: str
    category: EndToEndCategory
    system: EndToEndSystem
    selected_strategy: str | None = None
    planner_intent: str | None = None
    graph_operation: str | None = None
    planning_correct: bool | None = None
    strategy_correct: bool | None = None
    retrieval_rounds: int = 0
    evidence_count: int = 0
    evidence_recall: float = 0.0
    answer_status: str
    answer_correct: bool = False
    abstained: bool = False
    abstention_correct: bool = False
    expected_disambiguation: bool = False
    disambiguation_correct: bool = False
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT_EVIDENCE
    trusted_citations: int = 0
    citation_status: str | None = None
    citation_validity_rate: float = 0.0
    trusted_citation_rate: float = 0.0
    provenance_completeness: float = 0.0
    latency_ms: float = 0.0
    llm_calls: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    refined: bool = False
    refinement_success: bool = False
    no_new_evidence_refinement: bool = False
    max_round_insufficient: bool = False
    failure_reason: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("diagnostics", "token_usage")
    @classmethod
    def _json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)


class EndToEndSystemMetrics(BaseModel):
    cases: int = 0
    answer_accuracy: float = 0.0
    correct_abstention_rate: float = 0.0
    false_answer_rate: float = 0.0
    grounded_answer_rate: float = 0.0
    evidence_recall: float = 0.0
    citation_validity_rate: float = 0.0
    trusted_citation_rate: float = 0.0
    planning_accuracy: float | None = None
    strategy_accuracy: float | None = None
    refinement_rate: float = 0.0
    refinement_success_rate: float = 0.0
    high_confidence_error_rate: float = 0.0
    provenance_completeness: float = 0.0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    llm_calls_per_query: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)
    confidence_distribution: dict[str, int] = Field(default_factory=dict)


class PairwiseComparison(BaseModel):
    left: EndToEndSystem
    right: EndToEndSystem
    wins: int = 0
    ties: int = 0
    losses: int = 0


class EndToEndEvaluationReport(BaseModel):
    benchmark_version: str
    benchmark_checksum: str
    run_fingerprint: str
    generated_at: str
    case_results: list[EndToEndCaseResult]
    overall: dict[str, EndToEndSystemMetrics]
    by_category: dict[str, dict[str, EndToEndSystemMetrics]]
    pairwise: dict[str, PairwiseComparison]
    agentic_wins: list[str] = Field(default_factory=list)
    agentic_losses: list[str] = Field(default_factory=list)
    failure_analysis: dict[str, dict[str, int]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)


def retrieval_category(category: EndToEndCategory) -> RetrievalEvaluationCategory | None:
    try:
        return RetrievalEvaluationCategory(category.value)
    except ValueError:
        return None
