# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YouTube answer engine that answers technical questions using video transcripts from a specific YouTube channel. It combines retrieval (BM25, semantic search via ChromaDB, and hybrid) with LLM-based answer generation.

## Environment Setup

1. Install dependencies using uv:
   ```bash
   uv sync
   ```

2. Environment variables required in `.env`:
   - `YOUTUBE_API_KEY` - for fetching video metadata and comments
   - `OPENAI_API_KEY` - for LLM-based query generation and answer synthesis
   - `PROXY_USERNAME`, `PROXY_PASSWORD`, `PROXY_URL` - for transcript fetching via proxy

3. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Data Architecture

### Storage Layers
- **SQLite** (`data/youtube/videos.db`): Primary store for video metadata and transcripts
  - `videos` table: video_id (PK), title, transcript
  - `videos_fts` table: FTS5 virtual table for BM25 full-text search
  - Triggers keep FTS5 synced with videos table automatically

- **ChromaDB** (`data/youtube/chroma/`): Vector store for semantic search
  - Uses cosine distance
  - Stores combined title+transcript as document
  - Default embedding model from ChromaDB (OpenAI text-embedding-ada-002)

### Search Methods
All search methods in `utils/` return `list[tuple[video_id, score]]`:
- `search_bm25.py`: SQLite FTS5 keyword search
- `search_chroma.py`: Semantic search via ChromaDB embeddings
- `search_hybrid.py`: Reciprocal Rank Fusion (RRF) combining BM25 and ChromaDB with k=60

## Core Workflows

### 1. Data Ingestion

**Initial ingestion** (fetch all videos from channel):
```bash
python -m data_ingestion.ingest
```
- Fetches all videos from hardcoded `CHANNEL_ID` in `data_ingestion/ingest.py:16`
- Filters videos shorter than 50 seconds
- Fetches transcripts via proxy (with retry logic and exponential backoff)
- Stores in SQLite `videos` table
- Rate limited to 1.05s between videos

**Incremental update** (fetch only new videos):
```bash
python -m data_ingestion.update
```

**Sync to ChromaDB** (after any database changes):
```bash
python -m data_ingestion.sync_chroma
```

### 2. Evaluation Pipeline

**Retrieval Evaluation** (comparing BM25, Chroma, Hybrid):
```bash
# Generate synthetic queries from video content + YouTube comments
python -m utils.query_gen.run

# Run retrieval eval on dev/test splits
python -m evals.evaluate

# View results with interactive Streamlit viewer
streamlit run evals/analysis.py
```

**Response Evaluation** (answer quality):
```bash
# Generate answers using hybrid retrieval results
python -m evals.response.generate [--limit N]

# View results with interactive Streamlit viewer
streamlit run evals/viewer.py
```

### 3. Answer Generation
```python
from utils.answer import generate_answer

response = generate_answer(
    query="How do I optimize React performance?",
    limit=3,  # number of videos to retrieve
    search_method="hybrid"  # or "bm25" or "chroma"
)

print(response.answer)  # str
print(response.citations)  # list[Citation(video_id, title)]
```

## Data Flow

1. **Ingestion**: YouTube API → SQLite + ChromaDB
2. **Query Generation**: Video content + YouTube comments → OpenAI (gpt-4.1-2025-04-14) → Synthetic queries with splits (dev/test) → `data/queries/qset_v01/queries.csv`
3. **Retrieval Eval**: Queries → Search methods → Metrics (MRR, Recall@K) → `evals/retrieval/data/eval_results.jsonl`
4. **Response Eval**: `eval_results.jsonl` (hybrid retrieval) → OpenAI (gpt-4.1-2025-04-14) → Answers + Citations → `evals/response/data/response_results.jsonl`

## Key Implementation Details

### Answer Generation
- System uses OpenAI's `client.responses.parse()` with Pydantic `text_format` for structured output
- Model: `gpt-4.1-2025-04-14`
- Prompts are in `prompts/answer_system.md` and `prompts/answer_user.md`
- Context formatting combines title + transcript for each retrieved video

### Query Generation
- Uses video title, transcript, and YouTube comments as input
- Generates 9 queries per video (3 easy, 3 medium, 3 hard) across multiple query types
- Queries stored in `data/queries/qset_v01/queries.csv` with train/development/test splits

### Hybrid Search (RRF)
- Combines BM25 and ChromaDB rankings using Reciprocal Rank Fusion
- Formula: `score = sum(1 / (k + rank))` where k=60
- Both retrieval systems fetch same limit, then RRF merges rankings
- Returns top results by RRF score (higher = better)

## Development Tools

### Jupyter
```bash
jupyter lab
```
Main notebook: `sandbox.ipynb`

### Streamlit Viewers
- `evals/analysis.py` - Compare retrieval methods, filter by difficulty/query type
- `evals/viewer.py` - Review generated answers and citations, tag responses, export error analysis
- `utils/query_gen/viewer.py` - Browse generated queries

## Evaluation Data Files

- **Query Generation**:
  - `data/queries/qset_v01/raw_queries.jsonl` - Raw LLM-generated queries with metadata
  - `data/queries/qset_v01/queries.csv` - Processed queries with train/development/test splits

- **Retrieval Evaluation**:
  - `evals/retrieval/data/eval_results.jsonl` - Results from all search methods with metrics

- **Response Evaluation**:
  - `evals/response/data/response_results.jsonl` - Generated answers with citations
  - `evals/response/data/error_analysis-*.csv` - Error analysis exports with timestamps
  - `evals/response/data/tags.json` - User-defined tags for categorizing response quality

- **Test Files**:
  - `evals/retrieval/tests.py` - Unit tests for retrieval metrics
  - `evals/response/tests.py` - Placeholder for response evaluation tests

## Important Notes

- All Python modules use relative imports from project root (e.g., `from data_ingestion.database import ...`)
- Many scripts add project root to `sys.path` for imports
- Database path is `data/youtube/videos.db` (absolute or relative from project root)
- ChromaDB path is `data/youtube/chroma/` (PersistentClient)
- `.bak` files are automatically created when regenerating evaluation results
