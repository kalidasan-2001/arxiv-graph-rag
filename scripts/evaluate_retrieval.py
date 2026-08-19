"""Run deterministic retrieval evaluation and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import Settings
from app.embeddings.provider import get_embedding_provider
from app.graph.neo4j_repository import get_graph_repository
from app.retrieval.evidence import EvidenceProvenanceBridge
from app.retrieval.graph_search import GraphRetrievalService
from app.retrieval.hybrid import EvidenceFusionService, HybridRetrievalService
from app.retrieval.vector_search import VectorSearchService
from app.storage.qdrant.qdrant_repository import get_vector_repository
from evaluation.reporting import write_reports
from evaluation.runner import RetrievalEvaluationRunner, load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector, graph, and hybrid retrieval.")
    parser.add_argument("--benchmark", default="evaluation/retrieval_benchmark.json")
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    settings = Settings()
    vector_repository = get_vector_repository()
    graph_repository = get_graph_repository()
    service = HybridRetrievalService(
        vector_service=VectorSearchService(
            get_embedding_provider(),
            vector_repository,
            default_top_k=settings.VECTOR_SEARCH_DEFAULT_TOP_K,
            max_top_k=settings.VECTOR_SEARCH_MAX_TOP_K,
        ),
        graph_service=GraphRetrievalService(
            graph_repository,
            max_depth=settings.GRAPH_MAX_DEPTH,
            default_limit=settings.GRAPH_DEFAULT_LIMIT,
            max_limit=settings.GRAPH_MAX_LIMIT,
        ),
        provenance_bridge=EvidenceProvenanceBridge(
            vector_repository,
            max_supporting_chunks=settings.EVIDENCE_MAX_SUPPORTING_CHUNKS,
        ),
        fusion_service=EvidenceFusionService(rrf_k=settings.HYBRID_RRF_K),
        default_top_k=settings.HYBRID_DEFAULT_TOP_K,
        max_top_k=settings.HYBRID_MAX_TOP_K,
    )
    report = RetrievalEvaluationRunner(service, settings=settings).run_dataset(load_benchmark(Path(args.benchmark)))
    json_path, markdown_path = write_reports(report, args.output_dir)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
