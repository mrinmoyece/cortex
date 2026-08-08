# RAG Pipeline

Cortex uses a production-grade retrieval pipeline that goes well beyond embedding + cosine similarity.

## Pipeline Overview

```
Document → SemanticChunker → Embedder → VectorStore (Qdrant)
                                    ↓
                              BM25 Index (in-memory)

Query ──► Embed query ──► Dense search (Qdrant) ─┐
      │                                            ├─► RRF Fusion ──► Cohere Rerank ──► Top-K chunks
      └──────────────► Sparse search (BM25) ──────┘
```

## Ingestion

### Semantic Chunking (`SemanticChunker`)

Splits documents at paragraph boundaries rather than fixed character counts. Why this matters:

- **Fixed-size chunking** (naive): splits mid-sentence, destroys context within paragraphs
- **Semantic chunking** (Cortex): respects natural document structure; each chunk is self-contained

Configuration in `.env`:
```
RAG_CHUNK_SIZE=512        # Max words per chunk
RAG_CHUNK_OVERLAP=64      # Words carried over to next chunk for continuity
```

Each chunk stores `chunk_index` and all source metadata in its Qdrant payload.

### Embedding

Uses `text-embedding-3-large` (3072 dimensions) by default. Batched in groups of 20 to avoid API rate limits. Change via `EMBEDDING_MODEL` in `.env`.

### Ingestion API

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Full document text here...",
    "metadata": {
      "source": "annual-report-2025",
      "author": "finance-team",
      "date": "2025-01-15"
    }
  }'
```

For large documents, use the Celery background task:
```python
from cortex.workers.celery_app import ingest_document_task
ingest_document_task.delay(text=long_document, metadata={"source": "large-report"})
```

## Retrieval

### Hybrid Search (Dense + Sparse)

Dense retrieval (Qdrant) captures semantic similarity. Sparse retrieval (BM25) captures exact keyword matches. Neither alone is sufficient:

| Query type | Dense | Sparse | Hybrid |
|-----------|-------|--------|--------|
| "revenue growth" (semantic) | ✅ finds "profit increase" | ❌ misses paraphrase | ✅ |
| "EBITDA Q3 2025" (exact term) | ❌ may miss exact acronym | ✅ exact match | ✅ |
| Named entities ("Project Apollo") | ⚠️ inconsistent | ✅ exact match | ✅ |

### Reciprocal Rank Fusion

Merges dense and sparse rankings without needing to normalise their different score scales.

```
Score(doc) = Σ  1 / (k + rank_in_list)
```

`k=60` is the standard value. Documents appearing in both lists get a significant boost.

### Cohere Reranker

A cross-encoder that jointly encodes the query and each candidate chunk. Much more accurate than bi-encoder similarity but too slow for the full corpus — applied only to the top 20 fused candidates.

Configure: `COHERE_API_KEY` in `.env`. If not set, the pipeline returns RRF-fused results directly.

### Retrieval Configuration

```
RAG_TOP_K_RETRIEVE=20     # Candidates sent to reranker
RAG_TOP_K_RERANK=5        # Final results returned
RAG_BM25_WEIGHT=0.3       # Sparse contribution in RRF (0.0–1.0)
RAG_DENSE_WEIGHT=0.7      # Dense contribution in RRF
```

## Evaluation

Cortex runs automated RAG evaluation using Ragas. Metrics:

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Faithfulness | Claims in the answer supported by retrieved context | ≥ 0.85 |
| Context Precision | Retrieved chunks are relevant (precision of the retrieval) | ≥ 0.75 |
| Answer Relevancy | Answer addresses the question | ≥ 0.80 |
| Context Recall | Ground-truth information was retrieved (requires labels) | ≥ 0.70 |

Run evaluation:
```bash
# On demand
python -c "
import asyncio
from cortex.eval.ragas_runner import run_regression_suite
result = asyncio.run(run_regression_suite())
print(result.to_dict())
"

# Via Celery (as scheduled)
celery -A cortex.workers.celery_app call cortex.workers.celery_app.run_eval_regression
```

Results are emitted as Prometheus metrics and visible in Grafana.

## Source Filtering

Filter retrieval to a specific document source:

```python
chunks = await rag.retrieve(
    query="quarterly results",
    filters={"must": [{"key": "source", "match": {"value": "q3-2025-report"}}]}
)
```

Or via MCP:
```python
await search_knowledge(query="quarterly results", source_filter="q3-2025-report")
```

## Multi-tenancy

For tenant isolation, add a `tenant_id` field to document metadata at ingestion time and include it in retrieval filters. Each tenant's documents are logically isolated within the same Qdrant collection.
