from typing import Any

from app.core.config import Settings
from app.domain.enums import EvidenceSourceStore, EvidenceType, RetrievalStrategy
from app.domain.evidence import EvidenceItem, EvidenceProvenance, build_evidence_pool
from app.generation.answer import AnswerContextBuilder, GroundedAnswerGenerator
from app.generation.citations import CitationValidator
from app.generation.grounding import GroundingDecisionService
from app.retrieval.hybrid import FusedEvidenceItem, HybridRetrievalResult
from evaluation.end_to_end_models import EndToEndBenchmark, EndToEndCategory, EndToEndSystem
from evaluation.end_to_end_runner import (
    EndToEndEvaluationRunner,
    _aggregate,
    load_end_to_end_benchmark,
    validate_end_to_end_benchmark_file,
)
from evaluation.reporting import write_end_to_end_reports


def test_prompt20_benchmark_schema_and_category_counts() -> None:
    dataset = validate_end_to_end_benchmark_file("evaluation/end_to_end_benchmark.json")
    counts = {category.value: sum(1 for case in dataset.cases if case.category == category) for category in EndToEndCategory}

    assert len(dataset.cases) == 30
    assert counts == {
        "semantic": 5,
        "structural": 5,
        "shared_entity": 4,
        "multi_hop": 4,
        "mixed": 4,
        "unanswerable": 4,
        "ambiguous": 4,
    }


def test_metric_formulas_include_false_answers_and_high_confidence_errors() -> None:
    dataset = EndToEndBenchmark.model_validate(
        {
            "cases": [
                {
                    "id": "ok",
                    "category": "semantic",
                    "question": "Explain A",
                    "expected_targets": [{"target_type": "chunk", "target_id": "chunk:ok"}],
                    "expected_answer_facts": [{"fact_id": "fact:ok", "expected_text": "supported"}],
                    "required_citation_targets": ["chunk:ok"],
                },
                {"id": "bad", "category": "unanswerable", "question": "Missing", "expected_abstention": True},
            ]
        }
    )
    runner = _runner(_RetrievalService({"ok": [_text("chunk:ok")], "bad": [_text("chunk:bad")]}))

    report = runner.run_dataset(dataset)
    metrics = report.overall[EndToEndSystem.VECTOR_RAG.value]

    assert metrics.answer_accuracy == 1.0
    assert metrics.correct_abstention_rate == 0.0
    assert metrics.false_answer_rate == 1.0
    assert metrics.citation_validity_rate == 1.0
    assert metrics.grounded_answer_rate == 1.0
    assert metrics.high_confidence_error_rate == 0.5


def test_baseline_isolation_and_no_hidden_refinement() -> None:
    dataset = EndToEndBenchmark.model_validate(
        {
            "cases": [
                {
                    "id": "struct",
                    "category": "structural",
                    "question": "Which datasets?",
                    "graph_request": {"operation": "paper_datasets", "entity_id": "paper:a"},
                    "expected_targets": [{"target_type": "relationship", "target_id": "rel:a"}],
                    "required_citation_targets": ["rel:a"],
                }
            ]
        }
    )
    retrieval = _RetrievalService({"struct": [_graph("rel:a")]})
    report = _runner(retrieval).run_dataset(dataset)

    assert retrieval.calls == [RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID]
    assert all(
        result.retrieval_rounds <= 1
        for result in report.case_results
        if result.system in {EndToEndSystem.VECTOR_RAG, EndToEndSystem.GRAPH_RAG, EndToEndSystem.HYBRID_RAG}
    )
    assert report.case_results[-1].system == EndToEndSystem.AGENTIC_HYBRID_RAG


def test_report_generation_writes_required_files(tmp_path) -> None:
    dataset = EndToEndBenchmark.model_validate(
        {
            "cases": [
                {
                    "id": "ok",
                    "category": "semantic",
                    "question": "Explain A",
                    "expected_targets": [{"target_type": "chunk", "target_id": "chunk:ok"}],
                }
            ]
        }
    )
    report = _runner(_RetrievalService({"ok": [_text("chunk:ok")]})).run_dataset(dataset)

    json_path, markdown_path = write_end_to_end_reports(report, tmp_path)

    assert json_path.name == "end_to_end_report.json"
    assert markdown_path.name == "end_to_end_report.md"
    assert "# Executive Summary" in markdown_path.read_text(encoding="utf-8")
    assert "AGENTIC_HYBRID_RAG" in markdown_path.read_text(encoding="utf-8")


def test_refinement_and_planning_metrics_are_aggregated() -> None:
    results = [
        _result("a", refined=True, refinement_success=True, planning_correct=True, strategy_correct=True),
        _result("b", refined=True, refinement_success=False, planning_correct=False, strategy_correct=True),
    ]

    metrics = _aggregate(results)

    assert metrics.refinement_rate == 1.0
    assert metrics.refinement_success_rate == 0.5
    assert metrics.planning_accuracy == 0.5
    assert metrics.strategy_accuracy == 1.0


class _RetrievalService:
    def __init__(self, evidence_by_case: dict[str, list[EvidenceItem]]) -> None:
        self.evidence_by_case = evidence_by_case
        self.calls: list[RetrievalStrategy] = []

    def retrieve(self, **kwargs) -> HybridRetrievalResult:
        strategy = kwargs["strategy"]
        self.calls.append(strategy)
        case_id = "ok"
        if "Which datasets" in kwargs["query"]:
            case_id = "struct"
        elif "Missing" in kwargs["query"]:
            case_id = "bad"
        evidence = self.evidence_by_case.get(case_id, [])
        return HybridRetrievalResult(
            query=kwargs["query"],
            strategy=strategy,
            evidence=[
                FusedEvidenceItem(
                    evidence=item,
                    fusion_score=1.0,
                    branch_ranks={strategy.value: index},
                    branches=[strategy.value],
                )
                for index, item in enumerate(evidence, start=1)
            ],
            evidence_pool=build_evidence_pool(evidence),
            diagnostics={"strategy": strategy.value},
        )


class _AgenticWorkflow:
    def __init__(self, evidence: list[EvidenceItem] | None = None) -> None:
        self.evidence = evidence or [_text("chunk:ok")]

    def run(self, query: str):
        from app.generation.grounding import GroundingDecisionService
        from app.retrieval.critic import EvidenceAssessment, EvidenceCoverage, RefinementType
        from app.retrieval.workflow import RetrievalWorkflowStatus

        settings = Settings(_env_file=None)
        assessment = EvidenceAssessment(
            sufficient=True,
            coverage=EvidenceCoverage.COMPLETE,
            recommended_refinement_type=RefinementType.NONE,
        )
        pool = build_evidence_pool(self.evidence)
        context = AnswerContextBuilder(settings=settings).build(
            query=query,
            analysis=None,
            evidence_pool=pool,
            generation_config_fingerprint="fp",
        )
        validated = CitationValidator(settings=settings).validate(
            generated_answer=_Generated("supported [E1]."),
            evidence_pool=pool,
            answer_context=context,
        )
        final = GroundingDecisionService(settings=settings).decide(
            query=query,
            internal_status="SUCCESS",
            evidence=self.evidence,
            evidence_assessment=assessment,
            citation_validation=validated.citation_validation,
            validated_answer=validated,
            retrieval_round=1,
            warnings=[],
        )
        return type(
            "Result",
            (),
            {
                "final_answer": final,
                "final_status": final.status,
                "confidence": final.confidence,
                "retrieval_plan": type("Plan", (), {"strategy": RetrievalStrategy.VECTOR, "graph_operation": None})(),
                "analysis": type("Analysis", (), {"intent": type("Intent", (), {"value": "semantic_explanation"})()})(),
                "evidence": self.evidence,
                "answer": final.answer,
                "citations": final.citations,
                "citation_validation": validated.citation_validation,
                "retrieval_round": 1,
                "answer_generation_metadata": {},
                "evidence_sufficient": True,
                "trace": [],
                "status": RetrievalWorkflowStatus.SUCCESS,
                "grounding": final.grounding,
                "timings": {},
                "errors": [],
                "generated_answer": object(),
                "evidence_assessment": assessment,
            },
        )()


class _Generated:
    def __init__(self, text: str) -> None:
        self.text = text
        self.generation_metadata = {}


class _AnswerLLM:
    calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-answer"

    @property
    def provider_version(self) -> str:
        return "v1"

    @property
    def temperature(self) -> float:
        return 0.0

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model, max_output_tokens=None):
        self.calls += 1
        return response_model.model_validate({"text": "supported [E1].", "used_evidence_markers": ["E1"]})


def _runner(retrieval_service: _RetrievalService) -> EndToEndEvaluationRunner:
    settings = Settings(_env_file=None, VECTOR_SEARCH_DEFAULT_TOP_K=2, GRAPH_DEFAULT_LIMIT=2, HYBRID_DEFAULT_TOP_K=3)
    answer_llm = _AnswerLLM()
    return EndToEndEvaluationRunner(
        retrieval_service=retrieval_service,
        answer_context_builder=AnswerContextBuilder(settings=settings),
        answer_generator=GroundedAnswerGenerator(answer_llm, settings=settings),
        citation_validator=CitationValidator(settings=settings),
        grounding_service=GroundingDecisionService(settings=settings),
        agentic_workflow=_AgenticWorkflow(),
        settings=settings,
    )


def _result(
    case_id: str,
    *,
    refined: bool = False,
    refinement_success: bool = False,
    planning_correct: bool | None = None,
    strategy_correct: bool | None = None,
):
    from evaluation.end_to_end_models import EndToEndCaseResult

    return EndToEndCaseResult(
        case_id=case_id,
        category=EndToEndCategory.SEMANTIC,
        system=EndToEndSystem.AGENTIC_HYBRID_RAG,
        answer_status="answered",
        answer_correct=True,
        confidence="high",
        planning_correct=planning_correct,
        strategy_correct=strategy_correct,
        refined=refined,
        refinement_success=refinement_success,
    )


def _text(chunk_id: str) -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.TEXT,
        source="qdrant",
        chunk_id=chunk_id,
        text="supported",
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.QDRANT,
            chunk_ids=[chunk_id],
        ),
    )


def _graph(relationship_id: str) -> EvidenceItem:
    return EvidenceItem.create(
        evidence_type=EvidenceType.GRAPH_RELATIONSHIP,
        source="neo4j",
        entity_ids=["paper:a", "entity:dataset"],
        relationship_ids=[relationship_id],
        source_chunk_ids=["chunk:graph"],
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.NEO4J,
            chunk_ids=["chunk:graph"],
            relationship_ids=[relationship_id],
        ),
    )
