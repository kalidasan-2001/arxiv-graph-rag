"""Integration tests for `QdrantVectorRepository` against a real Qdrant
instance (prompt #58) -- collection creation, upsert, filtering, count,
search, and scoped delete. Requires a reachable Qdrant (see
`tests/integration/conftest.py`); skipped automatically otherwise.
"""

import pytest

from app.core.exceptions import VectorCollectionIncompatibleError
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository


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
        embedding_provider="fake",
        embedding_model="fake-model",
        embedding_config_fingerprint="embed-fp-a",
        vector_generation_fingerprint="gen-fp-a",
        text="Some chunk text.",
    )
    defaults.update(overrides)
    return VectorPointPayload(**defaults)


def _point(chunk_id: str, vector: list[float], **payload_overrides) -> VectorPoint:
    return VectorPoint(
        point_id=build_qdrant_point_id(chunk_id),
        vector=vector,
        payload=_payload(chunk_id=chunk_id, **payload_overrides),
    )


def _repo(qdrant_client, qdrant_collection_name) -> QdrantVectorRepository:
    return QdrantVectorRepository(qdrant_client, qdrant_collection_name)


class TestEnsureCollection:
    def test_creates_a_missing_collection(self, qdrant_client, qdrant_collection_name) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        assert not qdrant_client.collection_exists(qdrant_collection_name)

        repo.ensure_collection(dimension=8, distance="cosine")

        assert qdrant_client.collection_exists(qdrant_collection_name)
        info = qdrant_client.get_collection(qdrant_collection_name)
        assert info.config.params.vectors.size == 8

    def test_is_a_no_op_when_collection_already_matches(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=8, distance="cosine")
        repo.ensure_collection(dimension=8, distance="cosine")  # must not raise

    def test_dimension_mismatch_raises_incompatible_error(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=384, distance="cosine")

        with pytest.raises(VectorCollectionIncompatibleError):
            repo.ensure_collection(dimension=768, distance="cosine")

        # Never silently destroyed (prompt #14) -- the original collection
        # and its dimension must still be exactly what it was.
        info = qdrant_client.get_collection(qdrant_collection_name)
        assert info.config.params.vectors.size == 384


class TestUpsertAndCount:
    def test_upsert_then_count_reflects_the_points(self, qdrant_client, qdrant_collection_name) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")

        repo.upsert_chunks(
            [
                _point("chunk:a", [0.1, 0.2, 0.3, 0.4], paper_version_id="pv1"),
                _point("chunk:b", [0.2, 0.3, 0.4, 0.5], paper_version_id="pv1"),
            ]
        )

        assert repo.count_for_paper_version("pv1") == 2

    def test_upsert_is_idempotent_by_point_id(self, qdrant_client, qdrant_collection_name) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")

        point = _point("chunk:a", [0.1, 0.2, 0.3, 0.4], paper_version_id="pv1")
        repo.upsert_chunks([point])
        repo.upsert_chunks([point])  # same point id -- overwrite, not duplicate

        assert repo.count_for_paper_version("pv1") == 1

    def test_count_can_be_filtered_by_generation_fingerprint(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        repo.upsert_chunks(
            [
                _point("chunk:a", [0.1, 0.2, 0.3, 0.4], paper_version_id="pv1", vector_generation_fingerprint="gen1"),
                _point("chunk:b", [0.2, 0.3, 0.4, 0.5], paper_version_id="pv1", vector_generation_fingerprint="gen2"),
            ]
        )

        assert repo.count_for_paper_version("pv1", generation_fingerprint="gen1") == 1
        assert repo.count_for_paper_version("pv1", generation_fingerprint="gen2") == 1
        assert repo.count_for_paper_version("pv1") == 2


class TestScopedDelete:
    def test_delete_removes_only_the_specified_paper_version(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        repo.upsert_chunks(
            [
                _point("chunk:a", [0.1, 0.2, 0.3, 0.4], paper_version_id="pv1"),
                _point("chunk:b", [0.2, 0.3, 0.4, 0.5], paper_version_id="pv2"),
            ]
        )

        deleted = repo.delete_paper_version("pv1")

        assert deleted == 1
        assert repo.count_for_paper_version("pv1") == 0
        assert repo.count_for_paper_version("pv2") == 1  # untouched (prompt #33)

    def test_delete_can_exclude_the_current_generation(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        repo.upsert_chunks(
            [
                _point("chunk:a", [0.1, 0.2, 0.3, 0.4], paper_version_id="pv1", vector_generation_fingerprint="stale"),
                _point("chunk:b", [0.2, 0.3, 0.4, 0.5], paper_version_id="pv1", vector_generation_fingerprint="current"),
            ]
        )

        deleted = repo.delete_paper_version("pv1", exclude_generation_fingerprint="current")

        assert deleted == 1
        assert repo.count_for_paper_version("pv1", generation_fingerprint="current") == 1
        assert repo.count_for_paper_version("pv1") == 1


class TestSearch:
    def test_search_ranks_the_closer_vector_first(self, qdrant_client, qdrant_collection_name) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        repo.upsert_chunks(
            [
                _point("chunk:close", [1.0, 0.0, 0.0, 0.0], paper_version_id="pv1", text="close chunk"),
                _point("chunk:far", [0.0, 1.0, 0.0, 0.0], paper_version_id="pv1", text="far chunk"),
            ]
        )

        results = repo.search([0.9, 0.1, 0.0, 0.0], top_k=5)

        assert results[0].chunk_id == "chunk:close"
        assert results[0].similarity_score > results[1].similarity_score

    def test_search_can_be_filtered_by_paper_version_id(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        repo.upsert_chunks(
            [
                _point("chunk:a", [1.0, 0.0, 0.0, 0.0], paper_version_id="pv1"),
                _point("chunk:b", [1.0, 0.0, 0.0, 0.0], paper_version_id="pv2"),
            ]
        )

        results = repo.search([1.0, 0.0, 0.0, 0.0], top_k=5, paper_version_id="pv1")

        assert {hit.chunk_id for hit in results} == {"chunk:a"}

    def test_search_can_be_filtered_by_section_type(self, qdrant_client, qdrant_collection_name) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        repo.upsert_chunks(
            [
                _point("chunk:a", [1.0, 0.0, 0.0, 0.0], paper_version_id="pv1", section_type="introduction"),
                _point("chunk:b", [1.0, 0.0, 0.0, 0.0], paper_version_id="pv1", section_type="references"),
            ]
        )

        results = repo.search([1.0, 0.0, 0.0, 0.0], top_k=5, section_type="references")

        assert {hit.chunk_id for hit in results} == {"chunk:b"}

    def test_search_respects_top_k(self, qdrant_client, qdrant_collection_name) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        repo.upsert_chunks(
            [
                _point(f"chunk:{i}", [1.0, float(i) * 0.01, 0.0, 0.0], paper_version_id="pv1")
                for i in range(5)
            ]
        )

        results = repo.search([1.0, 0.0, 0.0, 0.0], top_k=2)

        assert len(results) == 2


class TestExactChunkLookup:
    def test_get_by_chunk_ids_returns_exact_payload_records(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        repo = _repo(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        repo.upsert_chunks(
            [
                _point(
                    "chunk:a",
                    [1.0, 0.0, 0.0, 0.0],
                    paper_version_id="pv1",
                    section_id="section:a",
                    section_type="methodology",
                    page_start=2,
                    page_end=3,
                    vector_generation_fingerprint="gen-current",
                    text="Exact source chunk.",
                ),
                _point("chunk:b", [0.0, 1.0, 0.0, 0.0], paper_version_id="pv2"),
            ]
        )

        records = repo.get_by_chunk_ids(["chunk:missing", "chunk:a"])

        assert [record.chunk_id for record in records] == ["chunk:a"]
        assert records[0].paper_version_id == "pv1"
        assert records[0].section_id == "section:a"
        assert records[0].page_start == 2
        assert records[0].vector_generation_fingerprint == "gen-current"
