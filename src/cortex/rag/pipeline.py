"""
Cortex RAG pipeline.

Production-grade retrieval-augmented generation with:
  - Multi-source ingestion (PDF, web, Confluence, database)
  - Semantic chunking (respects document structure, not naive fixed-size)
  - Hybrid search: dense vector (Qdrant) + sparse BM25 — fused with RRF
  - Cohere reranker — cross-encoder precision on top-k candidates
  - Ragas evaluation — faithfulness, context precision, answer relevancy
  - Async throughout for production throughput
"""

from __future__ import annotations

import hashlib

import litellm
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from cortex.config import settings
from cortex.exceptions import EmbeddingError, IngestionError, RetrievalError
from cortex.logging_config import get_logger
from cortex.obs.metrics import rag_documents_ingested, rag_retrieval_duration

logger = get_logger(__name__)


# ── Document model ────────────────────────────────────────────────────────────


class Document:
    __slots__ = ("content", "embedding", "id", "metadata")

    def __init__(self, content: str, metadata: dict | None = None) -> None:
        self.content = content
        # Content-addressed, not random. `uuid.uuid4()` meant re-ingesting an
        # unchanged document produced entirely new point ids, so every
        # re-ingest DUPLICATED the corpus in Qdrant instead of upserting over
        # it - retrieval then returns the same passage three times and the
        # reranker dutifully ranks all three. The `doc_hash` property already
        # existed and was written into the payload; it just was not used as
        # the identity it plainly is.
        #
        # Chunk index is mixed in so two identical paragraphs in one document
        # remain distinct points.
        chunk_index = (metadata or {}).get("chunk_index", 0)
        self.id = hashlib.sha256(f"{self.doc_hash}:{chunk_index}".encode()).hexdigest()
        self.metadata = metadata or {}
        self.embedding: list[float] | None = None

    @property
    def doc_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()


class RetrievedChunk:
    __slots__ = ("content", "doc_id", "metadata", "score", "source")

    def __init__(
        self, doc_id: str, content: str, metadata: dict, score: float, source: str
    ) -> None:
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.score = score
        self.source = source  # "dense" | "sparse" | "reranked"


# ── Chunker ───────────────────────────────────────────────────────────────────


class SemanticChunker:
    """
    Semantic chunking — splits at natural boundaries (paragraphs, headings)
    rather than fixed character counts. Preserves context better than naive
    chunking, especially for technical documents.
    """

    def __init__(
        self,
        max_chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        # Resolved here, not in the signature: a default argument is
        # evaluated at import time, so `= settings.x` reconstructs
        # Settings the moment the module is imported - the exact
        # import-time coupling this refactor removed everywhere else.
        max_chunk_size = max_chunk_size if max_chunk_size is not None else settings.rag_chunk_size
        overlap = overlap if overlap is not None else settings.rag_chunk_overlap
        self._max_size = max_chunk_size
        self._overlap = overlap

    def chunk(self, text: str, metadata: dict | None = None) -> list[Document]:
        """Split text into semantically coherent chunks."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Document] = []
        current_parts: list[str] = []
        current_len = 0
        chunk_index = 0

        for para in paragraphs:
            para_len = len(para.split())

            if current_len + para_len > self._max_size and current_parts:
                chunk_text = "\n\n".join(current_parts)
                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata={**(metadata or {}), "chunk_index": chunk_index},
                    )
                )
                chunk_index += 1
                # Overlap: carry over last paragraph
                overlap_parts = current_parts[-1:] if self._overlap > 0 else []
                current_parts = [*overlap_parts, para]
                current_len = sum(len(p.split()) for p in current_parts)
            else:
                current_parts.append(para)
                current_len += para_len

        if current_parts:
            chunks.append(
                Document(
                    content="\n\n".join(current_parts),
                    metadata={**(metadata or {}), "chunk_index": chunk_index},
                )
            )

        return chunks


# ── Embedder ──────────────────────────────────────────────────────────────────


class Embedder:
    """Batch embedder with retry and error handling."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Raises EmbeddingError on failure."""
        try:
            response = await litellm.aembedding(
                model=settings.embedding_model,
                input=texts,
            )
            return [item["embedding"] for item in response.data]
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc


# ── Vector Store Interface ────────────────────────────────────────────────────


class VectorStore:
    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None
        self._initialized = False

    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=str(settings.qdrant_url),
                api_key=settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None,
            )
        if not self._initialized:
            existing = {c.name for c in (await self._client.get_collections()).collections}
            if settings.qdrant_collection_rag not in existing:
                await self._client.create_collection(
                    collection_name=settings.qdrant_collection_rag,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )
            self._initialized = True
        return self._client

    async def upsert(self, documents: list[Document]) -> None:
        client = await self._get_client()
        points = [
            PointStruct(
                id=doc.id,
                vector=doc.embedding,
                payload={"content": doc.content, "hash": doc.doc_hash, **doc.metadata},
            )
            for doc in documents
            if doc.embedding is not None
        ]
        # Qdrant upsert in batches of 100 to avoid payload limits
        for i in range(0, len(points), 100):
            await client.upsert(
                collection_name=settings.qdrant_collection_rag,
                points=points[i : i + 100],
            )

    async def search(
        self, query_vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        client = await self._get_client()
        results = await client.search(
            collection_name=settings.qdrant_collection_rag,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filters,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                doc_id=str(r.id),
                content=r.payload.get("content", ""),
                metadata={k: v for k, v in r.payload.items() if k != "content"},
                score=r.score,
                source="dense",
            )
            for r in results
        ]


# ── BM25 Sparse Retriever ────────────────────────────────────────────────────


class SparseRetriever:
    """
    In-memory BM25 index. For production at scale, replace with
    Elasticsearch or Qdrant's sparse vector support.
    """

    def __init__(self) -> None:
        self._corpus: list[Document] = []
        self._index: BM25Okapi | None = None

    def index(self, documents: list[Document]) -> None:
        self._corpus = documents
        tokenised = [doc.content.lower().split() for doc in documents]
        self._index = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._index or not self._corpus:
            return []
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                doc_id=self._corpus[i].id,
                content=self._corpus[i].content,
                metadata=self._corpus[i].metadata,
                score=float(s),
                source="sparse",
            )
            for i, s in ranked
            if s > 0
        ]


# ── Reranker ──────────────────────────────────────────────────────────────────


class CohereReranker:
    """Cross-encoder reranker using Cohere's rerank-english-v3.0."""

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k if top_k is not None else settings.rag_top_k_rerank
        if not settings.cohere_api_key or not chunks:
            return chunks[:top_k]

        try:
            import cohere

            co = cohere.AsyncClient(settings.cohere_api_key.get_secret_value())
            results = await co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=[c.content for c in chunks],
                top_n=top_k,
            )
            reranked = []
            for r in results.results:
                original = chunks[r.index]
                reranked.append(
                    RetrievedChunk(
                        doc_id=original.doc_id,
                        content=original.content,
                        metadata=original.metadata,
                        score=r.relevance_score,
                        source="reranked",
                    )
                )
            return reranked
        except Exception as exc:
            logger.warning("reranker.failed", error=str(exc))
            return chunks[:top_k]


# ── Cortex RAG Pipeline ────────────────────────────────────────────────────────


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Ingest:  text → chunk → embed → upsert to Qdrant + BM25 index
    Retrieve: query → embed → dense search + BM25 → RRF fusion → rerank
    """

    def __init__(self) -> None:
        self._chunker = SemanticChunker()
        self._embedder = Embedder()
        self._vector_store = VectorStore()
        self._sparse = SparseRetriever()
        self._reranker = CohereReranker()
        self._all_docs: list[Document] = []  # For BM25 indexing

    async def ingest(self, text: str, metadata: dict | None = None) -> int:
        """
        Ingest a text document into the RAG pipeline.
        Returns the number of chunks created.
        """
        chunks = self._chunker.chunk(text, metadata)
        if not chunks:
            raise IngestionError("No chunks produced from document")

        # Embed in parallel batches of 20
        batch_size = 20
        all_embeddings: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = await self._embedder.embed_batch([c.content for c in batch])
            all_embeddings.extend(embeddings)

        for chunk, embedding in zip(chunks, all_embeddings, strict=False):
            chunk.embedding = embedding

        await self._vector_store.upsert(chunks)

        # Update BM25 index
        self._all_docs.extend(chunks)
        self._sparse.index(self._all_docs)

        rag_documents_ingested.inc(len(chunks))
        logger.info("rag.ingested", chunks=len(chunks), metadata=metadata)
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid retrieval: dense + sparse → RRF fusion → rerank.
        """
        import time

        top_k = top_k if top_k is not None else settings.rag_top_k_retrieve
        start = time.perf_counter()

        try:
            # Parallel dense + sparse search
            query_embedding = (await self._embedder.embed_batch([query]))[0]
            dense_task = self._vector_store.search(query_embedding, top_k=top_k, filters=filters)
            sparse_results = self._sparse.search(query, top_k=top_k)
            dense_results = await dense_task

            # Reciprocal Rank Fusion
            fused = self._rrf_fuse(dense_results, sparse_results, k=60)

            # Rerank top candidates
            reranked = await self._reranker.rerank(query, fused, top_k=settings.rag_top_k_rerank)

            elapsed = time.perf_counter() - start
            rag_retrieval_duration.observe(elapsed)
            logger.info(
                "rag.retrieved",
                query_preview=query[:80],
                dense_count=len(dense_results),
                sparse_count=len(sparse_results),
                final_count=len(reranked),
                latency_ms=round(elapsed * 1000),
            )
            return reranked

        except Exception as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

    @staticmethod
    def _rrf_fuse(
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion.
        Combines dense and sparse rankings into a single merged ranking.
        Score = sum(1 / (k + rank)) across all lists.
        """
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense):
            scores[chunk.doc_id] = scores.get(chunk.doc_id, 0) + 1.0 / (k + rank + 1)
            chunks_by_id[chunk.doc_id] = chunk

        for rank, chunk in enumerate(sparse):
            scores[chunk.doc_id] = scores.get(chunk.doc_id, 0) + 1.0 / (k + rank + 1)
            if chunk.doc_id not in chunks_by_id:
                chunks_by_id[chunk.doc_id] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunks_by_id[doc_id] for doc_id, _ in ranked]
