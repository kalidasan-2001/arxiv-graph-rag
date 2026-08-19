"""Integration tests for `VectorSearchService` against a real Qdrant
instance, using deterministic fake embedding providers (prompt #66) --
avoids brittle real-model similarity assertions in normal CI, per the
prompt's own guidance. Requires a reachable Qdrant; skipped automatically
otherwise.
"""

import pytest

from app.core.exceptions import VectorSearchError
from app.retrieval.vector_search import VectorSearchService
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository
from tests.embeddings.fakes import BagOfWordsEmbeddingProvider, FakeEmbeddingProvider


def _payload(**overrides) -> VectorPointPayload:
    defaults = dict(
        chunk_id="chunk:aaaa",
        paper_id="paper:arxiv:2401.11111",
        paper_version_id="paper-version:arxiv:2401.11111:v1",
        section_id="section:bbbb",
        section_type="introduction",
        section_title="Introduction",
        chunk_index=0,
        page_start=1,
        page_end=1,
        source="arxiv",
        source_id="2401.11111",
        published_year=2024,
        categories=["cs.CL"],
        chunking_version="v1",
        chunk_config_fingerprint="chunk-fp-a",
        embedding_provider="fake-bow",
        embedding_model="bow-v1",
        embedding_config_fingerprint="embed-fp-a",
        vector_generation_fingerprint="gen-fp-a",
        text="Some chunk text.",
    )
    defaults.update(overrides)
    return VectorPointPayload(**defaults)


class TestSemanticRanking:
    def test_matching_chunk_ranks_above_unrelated_chunks(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        vocabulary = [
            "graph", "neural", "network", "attack", "surface", "privacy",
            "banana", "bread", "recipe", "weather", "forecast",
        ]
        provider = BagOfWordsEmbeddingProvider(vocabulary)
        repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=provider.dimension, distance="cosine")

        matching_text = "Graph neural network attack surface for privacy"
        unrelated_texts = ["banana bread recipe", "weather forecast tomorrow"]

        points = [
            VectorPoint(
                point_id=build_qdrant_point_id("chunk:matching"),
                vector=provider.embed_documents([matching_text])[0],
                payload=_payload(chunk_id="chunk:matching", text=matching_text),
            )
        ]
        for i, text in enumerate(unrelated_texts):
            points.append(
                VectorPoint(
                    point_id=build_qdrant_point_id(f"chunk:unrelated{i}"),
                    vector=provider.embed_documents([text])[0],
                    payload=_payload(chunk_id=f"chunk:unrelated{i}", text=text),
                )
            )
        repo.upsert_chunks(points)

        service = VectorSearchService(provider, repo, default_top_k=5, max_top_k=50)
        results = service.search("graph neural networks")

        assert results[0].chunk_id == "chunk:matching"


class TestFiltersAndTopK:
    def test_search_returns_ranked_results_with_filters(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8, normalize=True)
        repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=provider.dimension, distance="cosine")
        repo.upsert_chunks(
            [
                VectorPoint(
                    point_id=build_qdrant_point_id("chunk:a"),
                    vector=provider.embed_documents(["alpha content"])[0],
                    payload=_payload(chunk_id="chunk:a", paper_version_id="pv1"),
                ),
                VectorPoint(
                    point_id=build_qdrant_point_id("chunk:b"),
                    vector=provider.embed_documents(["beta content"])[0],
                    payload=_payload(chunk_id="chunk:b", paper_version_id="pv2"),
                ),
            ]
        )

        service = VectorSearchService(provider, repo, default_top_k=5, max_top_k=50)
        results = service.search("alpha content", paper_version_id="pv1")

        assert {hit.chunk_id for hit in results} == {"chunk:a"}
        assert provider.embed_query_calls == ["alpha content"]  # embed_query, not embed_documents

    def test_default_top_k_is_used_when_not_specified(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8, normalize=True)
        repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=provider.dimension, distance="cosine")
        repo.upsert_chunks(
            [
                VectorPoint(
                    point_id=build_qdrant_point_id(f"chunk:{i}"),
                    vector=provider.embed_documents([f"text {i}"])[0],
                    payload=_payload(chunk_id=f"chunk:{i}"),
                )
                for i in range(5)
            ]
        )

        service = VectorSearchService(provider, repo, default_top_k=2, max_top_k=50)
        results = service.search("text")

        assert len(results) == 2

    def test_top_k_above_max_is_rejected(self, qdrant_client, qdrant_collection_name) -> None:
        provider = FakeEmbeddingProvider(dimension=8, normalize=True)
        repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=provider.dimension, distance="cosine")
        service = VectorSearchService(provider, repo, default_top_k=5, max_top_k=10)

        with pytest.raises(VectorSearchError):
            service.search("some query", top_k=100)

    def test_blank_query_is_rejected(self, qdrant_client, qdrant_collection_name) -> None:
        provider = FakeEmbeddingProvider(dimension=8, normalize=True)
        repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=provider.dimension, distance="cosine")
        service = VectorSearchService(provider, repo, default_top_k=5, max_top_k=10)

        with pytest.raises(VectorSearchError):
            service.search("   ")
