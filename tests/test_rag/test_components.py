"""Sparse retrieval, embedding and reranking.

The reranker's most important behaviour is what it does when Cohere is
absent or broken: retrieval must degrade to unreranked results, never to
no results.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.exceptions import EmbeddingError
from cortex.rag.pipeline import CohereReranker, Document, Embedder, RetrievedChunk, SparseRetriever


def _doc(i, content):
    return Document(content=content, metadata={"i": i})


def _chunk(doc_id, content="body", score=0.5):
    return RetrievedChunk(doc_id=doc_id, content=content, score=score, metadata={}, source="dense")


class TestSparseRetriever:
    @pytest.fixture
    def indexed(self):
        r = SparseRetriever()
        r.index(
            [
                _doc(0, "the refund policy allows returns within thirty days"),
                _doc(1, "our shipping partner delivers on weekdays only"),
                _doc(2, "refunds are processed to the original payment method"),
            ]
        )
        return r

    def test_lexical_matches_are_found_and_ranked(self, indexed):
        """BM25 needs a corpus with contrast - a two-document index where
        the term appears in one gives it almost nothing to weight, which is
        why the first version of this test found nothing."""
        hits = indexed.search("refund", top_k=5)
        assert hits, "an indexed term must be retrievable"
        assert "refund" in hits[0].content.lower()

    def test_an_unindexed_retriever_returns_nothing_rather_than_raising(self):
        """Sparse search before any ingest is a normal cold-start state, not
        an error - hybrid retrieval should still work off the dense side."""
        assert SparseRetriever().search("anything", top_k=5) == []

    def test_zero_scoring_documents_are_excluded(self, indexed):
        """BM25 scores every document. Returning the zero-scoring ones would
        pad the fusion input with pure noise."""
        hits = indexed.search("refund", top_k=10)
        assert all(h.score > 0 for h in hits)
        assert len(hits) < 3, "the shipping document should not match 'refund'"

    def test_top_k_is_respected(self, indexed):
        assert len(indexed.search("refund policy returns", top_k=1)) == 1

    def test_results_are_tagged_with_their_source(self, indexed):
        """Fusion and debugging both need to know which retriever produced
        a chunk."""
        assert all(h.source == "sparse" for h in indexed.search("refund", top_k=5))

    def test_a_query_matching_nothing_returns_empty(self, indexed):
        assert indexed.search("quantum chromodynamics", top_k=5) == []


class TestEmbedder:
    @pytest.mark.asyncio
    async def test_a_batch_is_embedded_in_one_call(self):
        response = SimpleNamespace(data=[{"embedding": [0.1]}, {"embedding": [0.2]}])
        with patch(
            "cortex.rag.pipeline.litellm.aembedding", AsyncMock(return_value=response)
        ) as emb:
            out = await Embedder().embed_batch(["a", "b"])
        assert len(out) == 2
        assert emb.await_count == 1, "batching exists to avoid one round trip per chunk"

    @pytest.mark.asyncio
    async def test_a_provider_failure_becomes_a_typed_error(self):
        """A raw provider exception here surfaces during ingest with no clue
        that embedding was the failing step."""
        with patch(
            "cortex.rag.pipeline.litellm.aembedding", AsyncMock(side_effect=RuntimeError("429"))
        ):
            with pytest.raises(EmbeddingError):
                await Embedder().embed_batch(["a"])


class TestCohereReranker:
    @pytest.mark.asyncio
    async def test_without_an_api_key_retrieval_still_returns_results(self):
        """Degrade to unreranked, never to empty. A missing optional API key
        must not silently turn RAG off."""
        chunks = [_chunk(f"d{i}") for i in range(10)]
        with patch("cortex.rag.pipeline.settings") as cfg:
            cfg.cohere_api_key = None
            cfg.rag_top_k_rerank = 5
            out = await CohereReranker().rerank("q", chunks)
        assert len(out) == 5
        assert [c.doc_id for c in out] == [f"d{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_no_candidates_reranks_to_nothing(self):
        with patch("cortex.rag.pipeline.settings") as cfg:
            cfg.cohere_api_key = None
            cfg.rag_top_k_rerank = 5
            assert await CohereReranker().rerank("q", []) == []

    @pytest.mark.asyncio
    async def test_reranking_reorders_and_retags(self):
        chunks = [_chunk("first"), _chunk("second")]
        results = SimpleNamespace(
            results=[
                SimpleNamespace(index=1, relevance_score=0.99),
                SimpleNamespace(index=0, relevance_score=0.10),
            ]
        )
        client = MagicMock()
        client.rerank = AsyncMock(return_value=results)

        with patch("cortex.rag.pipeline.settings") as cfg:
            cfg.cohere_api_key = MagicMock(get_secret_value=lambda: "key")
            cfg.rag_top_k_rerank = 5
            with patch.dict(
                "sys.modules", {"cohere": MagicMock(AsyncClient=MagicMock(return_value=client))}
            ):
                out = await CohereReranker().rerank("q", chunks)

        assert [c.doc_id for c in out] == ["second", "first"], "cross-encoder order must win"
        assert all(c.source == "reranked" for c in out)

    @pytest.mark.asyncio
    async def test_a_reranker_outage_degrades_to_the_fused_order(self):
        """Cohere being down must cost relevance, not availability."""
        chunks = [_chunk(f"d{i}") for i in range(6)]
        with patch("cortex.rag.pipeline.settings") as cfg:
            cfg.cohere_api_key = MagicMock(get_secret_value=lambda: "key")
            cfg.rag_top_k_rerank = 3
            with patch.dict(
                "sys.modules",
                {
                    "cohere": MagicMock(
                        AsyncClient=MagicMock(side_effect=ConnectionError("cohere down"))
                    )
                },
            ):
                out = await CohereReranker().rerank("q", chunks)
        assert out, "an outage must not empty the result set"


class TestDocumentIdentity:
    def test_the_same_content_gets_the_same_id(self):
        """Ids were `uuid4()`, so re-ingesting an unchanged file wrote a
        whole new set of points instead of upserting over the old ones -
        silently duplicating the corpus and returning the same passage
        several times per query."""
        assert Document("identical text").id == Document("identical text").id

    def test_different_content_gets_a_different_id(self):
        assert Document("one").id != Document("two").id

    def test_identical_paragraphs_in_one_document_stay_distinct(self):
        """Content addressing alone would collapse a repeated paragraph into
        a single point and lose one of its positions."""
        a = Document("repeated", metadata={"chunk_index": 0})
        b = Document("repeated", metadata={"chunk_index": 1})
        assert a.id != b.id
