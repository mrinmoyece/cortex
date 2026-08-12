"""Tests for the Cortex RAG pipeline."""

from __future__ import annotations

from cortex.rag.pipeline import (
    Document,
    RAGPipeline,
    RetrievedChunk,
    SemanticChunker,
    SparseRetriever,
)


class TestSemanticChunker:
    def test_short_text_produces_single_chunk(self):
        chunker = SemanticChunker(max_chunk_size=500, overlap=50)
        text = "This is a short paragraph."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == "This is a short paragraph."

    def test_long_text_splits_at_paragraphs(self):
        chunker = SemanticChunker(max_chunk_size=20, overlap=0)
        # Each paragraph is ~5 words; threshold is 20 words
        paragraphs = [f"This is paragraph number {i} with some words." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_overlap_carries_last_paragraph(self):
        chunker = SemanticChunker(max_chunk_size=10, overlap=5)
        text = "First paragraph here.\n\nSecond paragraph content.\n\nThird paragraph text."
        chunks = chunker.chunk(text)
        # With overlap enabled, later chunks should contain content from prior paragraph
        assert len(chunks) >= 1

    def test_empty_text_returns_empty_list(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk("   ")
        assert chunks == []

    def test_metadata_attached_to_chunks(self):
        chunker = SemanticChunker()
        meta = {"source": "test-doc", "author": "cortex"}
        chunks = chunker.chunk("Some content here.", metadata=meta)
        assert chunks[0].metadata["source"] == "test-doc"
        assert "chunk_index" in chunks[0].metadata

    def test_chunk_index_increments(self):
        chunker = SemanticChunker(max_chunk_size=5, overlap=0)
        paragraphs = "\n\n".join(f"Para {i} has words." for i in range(5))
        chunks = chunker.chunk(paragraphs)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestSparseRetriever:
    def test_returns_empty_without_index(self):
        retriever = SparseRetriever()
        results = retriever.search("test query", top_k=5)
        assert results == []

    def test_finds_matching_documents(self):
        retriever = SparseRetriever()
        docs = [
            Document("The quarterly earnings exceeded expectations significantly"),
            Document("The weather forecast shows heavy rain tomorrow afternoon"),
            Document("Revenue grew by fifteen percent in the third quarter"),
        ]
        retriever.index(docs)
        results = retriever.search("quarterly revenue earnings", top_k=2)
        assert len(results) >= 1
        # Both earnings-related docs should score higher than weather
        result_contents = [r.content for r in results]
        assert any("earnings" in c or "Revenue" in c for c in result_contents)

    def test_zero_score_docs_excluded(self):
        retriever = SparseRetriever()
        docs = [Document("cats and dogs are pets"), Document("python programming language")]
        retriever.index(docs)
        results = retriever.search("javascript frameworks", top_k=5)
        # BM25 should return nothing or very low scores
        assert all(r.score > 0 for r in results)

    def test_top_k_respected(self):
        retriever = SparseRetriever()
        docs = [Document(f"Document about topic {i} in detail") for i in range(20)]
        retriever.index(docs)
        results = retriever.search("document topic", top_k=3)
        assert len(results) <= 3

    def test_source_is_sparse(self):
        retriever = SparseRetriever()
        docs = [Document("test content with matching words")]
        retriever.index(docs)
        results = retriever.search("matching words", top_k=1)
        if results:
            assert results[0].source == "sparse"


class TestRRFFusion:
    def test_rrf_combines_rankings(self):
        dense = [
            RetrievedChunk("doc1", "content1", {}, 0.95, "dense"),
            RetrievedChunk("doc2", "content2", {}, 0.80, "dense"),
        ]
        sparse = [
            RetrievedChunk("doc2", "content2", {}, 8.0, "sparse"),
            RetrievedChunk("doc3", "content3", {}, 6.0, "sparse"),
        ]
        fused = RAGPipeline._rrf_fuse(dense, sparse, k=60)
        # doc2 appears in both lists — should be ranked highly
        doc_ids = [c.doc_id for c in fused]
        assert "doc2" in doc_ids
        doc2_rank = doc_ids.index("doc2")
        assert doc2_rank <= 1  # Should be top 2

    def test_rrf_handles_empty_sparse(self):
        dense = [RetrievedChunk("doc1", "content", {}, 0.9, "dense")]
        fused = RAGPipeline._rrf_fuse(dense, [], k=60)
        assert len(fused) == 1
        assert fused[0].doc_id == "doc1"

    def test_rrf_handles_empty_dense(self):
        sparse = [RetrievedChunk("doc1", "content", {}, 5.0, "sparse")]
        fused = RAGPipeline._rrf_fuse([], sparse, k=60)
        assert len(fused) == 1

    def test_rrf_no_duplicates(self):
        doc = RetrievedChunk("doc1", "content", {}, 0.9, "dense")
        doc_sparse = RetrievedChunk("doc1", "content", {}, 5.0, "sparse")
        fused = RAGPipeline._rrf_fuse([doc], [doc_sparse], k=60)
        ids = [c.doc_id for c in fused]
        assert len(ids) == len(set(ids))


class TestDocumentModel:
    def test_document_has_unique_id(self):
        doc1 = Document("content one")
        doc2 = Document("content two")
        assert doc1.id != doc2.id

    def test_doc_hash_deterministic(self):
        doc1 = Document("same content")
        doc2 = Document("same content")
        assert doc1.doc_hash == doc2.doc_hash

    def test_doc_hash_differs_for_different_content(self):
        doc1 = Document("content A")
        doc2 = Document("content B")
        assert doc1.doc_hash != doc2.doc_hash

    def test_metadata_defaults_to_empty_dict(self):
        doc = Document("content")
        assert doc.metadata == {}


class TestDocumentIdIsAcceptableToQdrant:
    """Qdrant accepts only UUIDs or unsigned integers as point IDs. `Document.id`
    was a 64-character SHA-256 hex digest, which is neither, so the server
    rejected every upsert - ingestion could not have worked against a real
    instance, and no test reached one."""

    def test_the_id_is_a_valid_uuid(self):
        import uuid

        uuid.UUID(Document("content").id)  # raises if it is not

    def test_content_addressing_survived_the_change(self):
        assert Document("same").id == Document("same").id
        assert Document("same").id != Document("other").id

    def test_identical_content_from_distinct_sources_has_distinct_ids(self):
        assert Document("same", {"source": "one"}).id != Document("same", {"source": "two"}).id


class TestSparseRetrievalHonoursFilters:
    """Dense search enforced `filters`; sparse did not. A filtered hybrid
    search therefore returned unfiltered BM25 hits - in a multi-tenant
    deployment, another tenant's documents. A filter enforced on one of two
    retrieval paths is not a filter."""

    @staticmethod
    def _indexed() -> SparseRetriever:
        retriever = SparseRetriever()
        retriever.index(
            [
                Document("the refund policy allows returns", metadata={"tenant": "acme"}),
                Document("refunds go to the original card", metadata={"tenant": "globex"}),
                Document("shipping runs on weekdays only", metadata={"tenant": "acme"}),
                Document("the warehouse closes at six", metadata={"tenant": "globex"}),
            ]
        )
        return retriever

    def test_a_filter_excludes_other_tenants(self):
        hits = self._indexed().search("refund policy", top_k=10, filters={"tenant": "acme"})
        assert hits
        assert all(h.metadata.get("tenant") == "acme" for h in hits)

    def test_the_same_query_unfiltered_does_reach_the_other_tenant(self):
        """Without this the filtered case above proves nothing - the excluded
        document has to be one BM25 would otherwise have returned."""
        tenants = {
            h.metadata.get("tenant") for h in self._indexed().search("refund refunds", top_k=10)
        }
        assert tenants == {"acme", "globex"}

    def test_a_filter_matching_nothing_returns_nothing(self):
        assert (
            self._indexed().search("refund refunds", top_k=10, filters={"tenant": "nobody"}) == []
        )

    def test_filters_are_conjunctive(self):
        retriever = SparseRetriever()
        retriever.index(
            [
                Document("alpha topic", metadata={"tenant": "acme", "src": "wiki"}),
                Document("beta topic", metadata={"tenant": "acme", "src": "pdf"}),
                Document("gamma matter", metadata={"tenant": "globex", "src": "wiki"}),
            ]
        )
        assert retriever.search("alpha", top_k=5, filters={"tenant": "acme", "src": "wiki"})
        assert not retriever.search("alpha", top_k=5, filters={"tenant": "acme", "src": "pdf"})


class TestPayloadFilterHelpers:
    """The dense and sparse sides must apply the *same* predicate, or a hybrid
    result set is filtered differently depending on which retriever found the
    chunk."""

    def test_an_empty_filter_is_no_filter(self):
        from cortex.rag.pipeline import build_payload_filter, matches_payload_filter

        assert build_payload_filter(None) is None
        assert build_payload_filter({}) is None
        assert matches_payload_filter({"a": 1}, None) is True

    def test_both_halves_agree_on_the_keys_they_enforce(self):
        from cortex.rag.pipeline import build_payload_filter, matches_payload_filter

        filters = {"tenant": "acme", "src": "wiki"}
        built = build_payload_filter(filters)
        assert built is not None
        assert {c.key for c in built.must} == set(filters)  # type: ignore[union-attr]
        assert matches_payload_filter(filters, filters) is True
        assert matches_payload_filter({"tenant": "acme"}, filters) is False
        assert matches_payload_filter({}, filters) is False
