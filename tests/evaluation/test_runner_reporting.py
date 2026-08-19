from app.core.config import Settings
from app.domain.enums import EvidenceScoreKind, EvidenceSourceStore, EvidenceType, RetrievalStrategy
from app.domain.evidence import EvidenceItem, EvidencePool, EvidenceProvenance
from app.retrieval.hybrid import FusedEvidenceItem, HybridRetrievalResult, RetrievalBranchResult
from evaluation.models import BenchmarkGraphRequest, ExpectedTarget, RetrievalBenchmarkCase, RetrievalBenchmarkDataset
from evaluation.reporting import write_reports
from evaluation.runner import RetrievalEvaluationRunner


def _text(chunk_id: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"evidence:{chunk_id}",
        evidence_type=EvidenceType.TEXT,
        paper_id="paper:a",
        paper_version_id="paper:a:v1",
        chunk_id=chunk_id,
        section_id="section:intro",
        section_type="introduction",
        text=chunk_id,
        score=0.8,
        score_kind=EvidenceScoreKind.VECTOR_SIMILARITY,
        source="qdrant",
        source_store=EvidenceSourceStore.QDRANT,
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.QDRANT,
            paper_id="paper:a",
            paper_version_id="paper:a:v1",
            chunk_ids=[chunk_id],
        ),
    )


class _RetrievalService:
    def retrieve(self, *, strategy, **kwargs):
        evidence = [_text("chunk:target")] if strategy != RetrievalStrategy.GRAPH else []
        fused = [
            FusedEvidenceItem(evidence=item, fusion_score=1.0, branch_ranks={strategy.value: rank})
            for rank, item in enumerate(evidence, start=1)
        ]
        return HybridRetrievalResult(
            query=kwargs["query"],
            strategy=strategy,
            evidence=fused,
            evidence_pool=EvidencePool(),
            vector_result=RetrievalBranchResult(strategy=RetrievalStrategy.VECTOR, evidence=evidence),
            graph_result=None,
            diagnostics={"duration_ms": 0},
        )


def test_runner_marks_graph_not_applicable_without_lowering_metrics() -> None:
    dataset = RetrievalBenchmarkDataset(
        cases=[
            RetrievalBenchmarkCase(
                id="semantic",
                category="semantic",
                question="What does it say?",
                expected_targets=[ExpectedTarget(target_type="chunk", target_id="chunk:target")],
                relevance_target_type="chunk",
            )
        ]
    )

    report = RetrievalEvaluationRunner(_RetrievalService(), settings=Settings()).run_dataset(dataset)

    assert report.overall["vector"].hit_at_k[5] == 1.0
    assert report.overall["graph"].applicable_cases == 0
    assert report.overall["graph"].not_applicable_cases == 1
    assert report.vector_wins == ["semantic"]


def test_report_writers_emit_json_and_markdown(tmp_path) -> None:
    dataset = RetrievalBenchmarkDataset(
        cases=[
            RetrievalBenchmarkCase(
                id="case",
                category="structural",
                question="Which datasets?",
                graph_request=BenchmarkGraphRequest(operation="paper_datasets", entity_id="paper:a"),
                expected_targets=[ExpectedTarget(target_type="chunk", target_id="chunk:target")],
                relevance_target_type="chunk",
            )
        ]
    )
    report = RetrievalEvaluationRunner(_RetrievalService(), settings=Settings()).run_dataset(dataset)

    json_path, markdown_path = write_reports(report, tmp_path)

    assert json_path.exists()
    assert markdown_path.exists()
    assert "Retrieval Evaluation Report" in markdown_path.read_text(encoding="utf-8")
