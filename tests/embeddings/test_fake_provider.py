"""Sanity tests for the deterministic fakes themselves -- these back every
other embedding-related unit/integration test, so their own determinism
and shape guarantees are worth verifying directly (prompt #56)."""

from tests.embeddings.fakes import BagOfWordsEmbeddingProvider, FakeEmbeddingProvider


class TestFakeEmbeddingProvider:
    def test_same_text_produces_the_same_vector(self) -> None:
        provider = FakeEmbeddingProvider()
        assert provider.embed_query("hello world") == provider.embed_query("hello world")

    def test_different_text_produces_a_different_vector(self) -> None:
        provider = FakeEmbeddingProvider()
        assert provider.embed_query("hello") != provider.embed_query("goodbye")

    def test_vectors_have_the_configured_dimension(self) -> None:
        provider = FakeEmbeddingProvider(dimension=16)
        vectors = provider.embed_documents(["a", "b", "c"])
        assert all(len(v) == 16 for v in vectors)

    def test_normalize_produces_unit_length_vectors(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8, normalize=True)
        vector = provider.embed_query("some text")
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_embed_documents_and_embed_query_are_tracked_separately(self) -> None:
        provider = FakeEmbeddingProvider()
        provider.embed_documents(["a", "b"])
        provider.embed_query("q")

        assert provider.embed_documents_calls == [["a", "b"]]
        assert provider.embed_query_calls == ["q"]
        assert provider.call_count == 1

    def test_config_fingerprint_is_deterministic_and_stable(self) -> None:
        first = FakeEmbeddingProvider(dimension=8, model_name="m", provider_name="p")
        second = FakeEmbeddingProvider(dimension=8, model_name="m", provider_name="p")
        assert first.config_fingerprint == second.config_fingerprint


class TestBagOfWordsEmbeddingProvider:
    def test_shared_words_increase_similarity(self) -> None:
        provider = BagOfWordsEmbeddingProvider(
            ["graph", "neural", "network", "attack", "privacy", "banana", "recipe"]
        )
        query = provider.embed_query("graph neural network")
        matching = provider.embed_documents(["Graph neural network attack surface"])[0]
        unrelated = provider.embed_documents(["banana bread recipe"])[0]

        matching_similarity = sum(a * b for a, b in zip(query, matching, strict=True))
        unrelated_similarity = sum(a * b for a, b in zip(query, unrelated, strict=True))
        assert matching_similarity > unrelated_similarity

    def test_vectors_are_unit_length(self) -> None:
        provider = BagOfWordsEmbeddingProvider(["a", "b", "c"])
        vector = provider.embed_query("a b")
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9
