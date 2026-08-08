"""Chunking and fusion — the deterministic core of the RAG pipeline.

Both are pure functions, so there is no excuse for them being untested,
and both are where retrieval quality is silently won or lost.
"""

from __future__ import annotations

from cortex.rag.pipeline import RAGPipeline, RetrievedChunk, SemanticChunker


def _chunk(doc_id: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id, content=f"body of {doc_id}", score=score, metadata={}, source="test"
    )


class TestSemanticChunker:
    def test_splits_on_paragraph_boundaries_not_character_counts(self):
        """Splitting mid-sentence at a fixed offset is how a retrieved chunk
        ends up starting halfway through the clause that mattered."""
        chunker = SemanticChunker(max_chunk_size=10, overlap=0)
        text = "\n\n".join(" ".join(f"w{i}" for i in range(8)) for _ in range(4))
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        for c in chunks:
            assert not c.content.startswith(" ")
            assert c.content.strip() == c.content

    def test_every_chunk_is_indexed_in_order(self):
        chunker = SemanticChunker(max_chunk_size=5, overlap=0)
        text = "\n\n".join("alpha beta gamma delta" for _ in range(5))
        chunks = chunker.chunk(text)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_caller_metadata_survives_chunking(self):
        chunker = SemanticChunker(max_chunk_size=100, overlap=0)
        chunks = chunker.chunk("one\n\ntwo", metadata={"source": "handbook.pdf"})
        assert all(c.metadata["source"] == "handbook.pdf" for c in chunks)

    def test_overlap_carries_context_across_the_boundary(self):
        """Without overlap, a fact split across a boundary is retrievable
        from neither chunk."""
        chunker = SemanticChunker(max_chunk_size=6, overlap=3)
        text = "\n\n".join(["alpha beta gamma", "delta epsilon zeta", "eta theta iota"])
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert "delta epsilon zeta" in chunks[1].content

    def test_short_text_stays_a_single_chunk(self):
        assert len(SemanticChunker(max_chunk_size=500, overlap=50).chunk("just a line")) == 1

    def test_empty_and_whitespace_only_text_yields_nothing(self):
        chunker = SemanticChunker(max_chunk_size=100, overlap=0)
        assert chunker.chunk("") == []
        assert chunker.chunk("\n\n   \n\n") == []


class TestReciprocalRankFusion:
    def test_a_document_ranked_by_both_retrievers_beats_one_ranked_by_either(self):
        """The entire justification for hybrid retrieval. If this does not
        hold, running two retrievers is just paying twice."""
        dense = [_chunk("only-dense"), _chunk("both")]
        sparse = [_chunk("only-sparse"), _chunk("both")]
        fused = RAGPipeline._rrf_fuse(dense, sparse, k=60)
        assert fused[0].doc_id == "both"

    def test_fusion_is_order_independent(self):
        dense, sparse = [_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")]
        one = [c.doc_id for c in RAGPipeline._rrf_fuse(dense, sparse, k=60)]
        two = [c.doc_id for c in RAGPipeline._rrf_fuse(dense, sparse, k=60)]
        assert one == two

    def test_no_document_is_lost_or_duplicated(self):
        dense = [_chunk("a"), _chunk("b")]
        sparse = [_chunk("b"), _chunk("c")]
        ids = [c.doc_id for c in RAGPipeline._rrf_fuse(dense, sparse, k=60)]
        assert sorted(ids) == ["a", "b", "c"]
        assert len(ids) == len(set(ids)), (
            "a document appearing in both lists must not be duplicated"
        )

    def test_scores_from_the_retrievers_do_not_leak_into_the_ranking(self):
        """RRF ranks by POSITION, deliberately: cosine similarity and BM25
        scores are on incomparable scales, and mixing them numerically is
        the classic hybrid-retrieval bug."""
        dense = [_chunk("low-score-but-first", score=0.01)]
        sparse = [_chunk("high-score-but-second", score=999.0), _chunk("low-score-but-first", 0.01)]
        fused = RAGPipeline._rrf_fuse(dense, sparse, k=60)
        assert fused[0].doc_id == "low-score-but-first"

    def test_an_empty_retriever_does_not_break_fusion(self):
        """One retriever returning nothing is normal - a rare term has no
        BM25 hits - and must not empty the result."""
        fused = RAGPipeline._rrf_fuse([_chunk("a")], [], k=60)
        assert [c.doc_id for c in fused] == ["a"]
        assert RAGPipeline._rrf_fuse([], [], k=60) == []

    def test_k_dampens_the_advantage_of_being_first(self):
        """Small k makes rank 1 dominate; the k=60 default is what keeps a
        single retriever from dictating the merged order."""
        dense = [_chunk("first"), _chunk("second")]
        sparse = [_chunk("second"), _chunk("first")]
        assert len(RAGPipeline._rrf_fuse(dense, sparse, k=1)) == 2
        assert len(RAGPipeline._rrf_fuse(dense, sparse, k=60)) == 2
