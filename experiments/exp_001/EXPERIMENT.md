# exp_001: Baseline Hybrid RAG

## Intent

Establish baseline performance for hybrid retrieval (BM25 + semantic) with structured answer generation. This experiment serves as the reference point for all future experiments and validates the core RAG pipeline.

## Pipeline

1. **Retrieval**: RRF fusion of BM25 (SQLite FTS5) and ChromaDB (text-embedding-ada-002)
   - RRF k=60
   - Retrieve top 10 per method, merge to top 3 for answer generation
   - BM25: Keyword-based search using SQLite FTS5 on title + transcript
   - ChromaDB: Semantic search using OpenAI embeddings (cosine distance)
   - Hybrid: Reciprocal Rank Fusion combining both rankings

2. **Context**: Top 3 videos (full title + transcript)
   - No chunking - full transcript per video
   - Context formatted as markdown sections

3. **Answer**: GPT-4.1-2025-04-14 with Pydantic structured output
   - System prompt enforces grounding in provided context
   - Structured output ensures citations are always included
   - Citations reference specific video IDs and titles

## Key Decisions

### No Chunking
Full transcript per video (simpler baseline). This means:
- Each video is treated as a single document
- Retrieval operates at video-level granularity
- Context can be very large for long videos
- Future experiments can explore chunking strategies

### Shared ChromaDB
Uses canonical `data/youtube/chroma/` index with text-embedding-ada-002. This is a **deviation from strict experiment isolation** but justified because:
- Embeddings are deterministic for a given model
- Rebuilding identical embeddings is wasteful
- Experiments with different embedding models will build their own indices
- This is clearly documented in run_receipt.json

### 3-Video Context
Retrieves top 3 videos for answer generation. This balances:
- Coverage: Enough context to answer most questions
- Token cost: Stays within reasonable context limits
- Quality: More focused than using 5+ videos

### RRF k=60
Standard Reciprocal Rank Fusion constant. Formula: `score = sum(1 / (k + rank))`
- Higher k gives more weight to lower-ranked items
- k=60 is a common default in hybrid search systems

## Query Set

- **ID**: qset_v01
- **Format**: CSV with splits (train/dev/test)
- **Location**: `data/queries/qset_v01/queries.csv`
- **Dev queries**: 105
- **Test queries**: 34
- **Total**: 139

Query types: exact, procedural, conceptual
Difficulty levels: grounded, medium, hard

## Runs

<!-- run.py will append results here -->

### r001 (dev)

**Date**: 2025-12-26
**Git SHA**: 0d7a161*
**Queries**: 5

**Retrieval (Hybrid)**:
- Recall@1: 0.2000
- Recall@3: 0.8000
- MRR@10: 0.4733

**Response**:
- Gold cited: 0.8000
- Citation precision: 1.0000
- Avg answer length: 1461 chars

### r001 (dev)

**Date**: 2025-12-26
**Git SHA**: 0d7a161*
**Queries**: 5

**Retrieval (Hybrid)**:
- Recall@1: 0.2000
- Recall@3: 0.8000
- MRR@10: 0.4733

**Response**:
- Gold cited: 0.8000
- Citation precision: 1.0000
- Avg answer length: 1185 chars

### r001 (dev)

**Date**: 2025-12-26
**Git SHA**: 0d7a161*
**Queries**: 5

**Retrieval (Hybrid)**:
- Recall@1: 0.2000
- Recall@3: 0.8000
- MRR@10: 0.4733

**Response**:
- Gold cited: 0.8000
- Citation precision: 1.0000
- Avg answer length: 1572 chars

### r001 (dev)

**Date**: 2025-12-26
**Git SHA**: 0d7a161*
**Queries**: 3

**Retrieval (Hybrid)**:
- Recall@1: 0.3333
- Recall@3: 0.6667
- MRR: 0.5000

**Response**:
- Gold cited: 0.6667
- Citation precision: 1.0000
- Avg answer length: 1269 chars
