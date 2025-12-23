# yt-answer-engine

AI-powered question answering system that grounds responses in YouTube video transcripts using hybrid retrieval and LLM synthesis.

This system combines multiple retrieval methods (BM25 keyword search, semantic search via ChromaDB, and hybrid fusion) to find relevant video content, then uses OpenAI GPT-4.1 to generate accurate answers with inline citations. It includes a complete evaluation pipeline for benchmarking retrieval quality and answer accuracy.

## Features

- **Hybrid Retrieval**: Combines BM25 keyword search and ChromaDB semantic search using Reciprocal Rank Fusion (RRF)
- **LLM-Grounded Answers**: OpenAI GPT-4.1 generates answers strictly grounded in retrieved video transcripts
- **Citation System**: Automatic inline citations linking answers back to source videos
- **Evaluation Pipeline**: Synthetic query generation and comprehensive benchmarking for both retrieval and response quality
- **Interactive Analysis**: Streamlit-based viewers for exploring results, tagging responses, and exporting error analysis
- **Incremental Updates**: Support for both full channel ingestion and incremental updates

## Prerequisites

- **Python**: >= 3.13
- **Package Manager**: `uv` (recommended) or `pip`
- **API Keys**:
  - YouTube Data API v3 key
  - OpenAI API key
- **Proxy**: HTTP proxy credentials for transcript fetching (may be required due to geo-restrictions)

## Dependencies

Core dependencies (see `pyproject.toml` for complete list):

- `chromadb` >= 1.3.7 - Vector database for semantic search
- `openai` >= 2.14.0 - LLM-based answer generation
- `streamlit` >= 1.52.2 - Interactive evaluation viewers
- `youtube-transcript-api` >= 1.2.3 - Transcript fetching
- `pydantic` >= 2.12.5 - Structured output parsing
- `pandas` >= 2.3.3 - Data analysis and query management
- `jupyterlab` >= 4.5.1 - Development notebook environment

## Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd yt-answer-engine
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. Configure Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
YOUTUBE_API_KEY=your_youtube_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
PROXY_USERNAME=your_proxy_username
PROXY_PASSWORD=your_proxy_password
PROXY_URL=your_proxy_url
```

**Where to get API keys:**
- YouTube API: [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
- OpenAI API: [OpenAI Platform](https://platform.openai.com/api-keys)

### 4. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 5. Initialize Database

The database will be created automatically on first ingestion, but you can initialize it manually:

```bash
python -c "from data_ingestion.database import init_db; init_db()"
```

## System Overview

### Architecture

The system uses a two-stage architecture:

1. **Retrieval Stage**: Multi-method search across video transcripts
   - **BM25** (SQLite FTS5): Fast keyword-based search
   - **Semantic** (ChromaDB): Vector similarity search with embeddings
   - **Hybrid** (RRF): Reciprocal Rank Fusion combining both methods

2. **Generation Stage**: LLM-based answer synthesis
   - Retrieves top-k videos based on query
   - Formats video titles and transcripts as context
   - Uses OpenAI GPT-4.1 with structured output for answer generation
   - Enforces citation requirements through system prompts

### Data Storage

- **SQLite** (`data/videos.db`):
  - `videos` table: Primary storage for video metadata and transcripts
  - `videos_fts` table: FTS5 virtual table for BM25 search
  - Automatic triggers keep FTS5 synchronized with main table

- **ChromaDB** (`data/chroma/`):
  - Vector embeddings for semantic search
  - Uses OpenAI text-embedding-ada-002 (default)
  - Cosine distance metric

### Evaluation Framework

1. **Query Generation**: Synthetic queries generated from video content and YouTube comments using GPT-4.1
   - 9 queries per video (3 easy, 3 medium, 3 hard)
   - Train/validation/test splits for proper evaluation

2. **Retrieval Evaluation**: Benchmark all search methods
   - Metrics: Mean Reciprocal Rank (MRR), Recall@K
   - Compare BM25, ChromaDB, and Hybrid approaches

3. **Response Evaluation**: Assess answer quality
   - Manual tagging for response quality
   - Error analysis and export capabilities

## Core Workflows

### Data Ingestion

**Initial ingestion** (fetch all videos from channel):

```bash
python -m data_ingestion.ingest
```

This will:
- Fetch all video IDs from the hardcoded channel (see `data_ingestion/ingest.py:16` to modify)
- Filter out videos shorter than 50 seconds
- Download transcripts via proxy with retry logic
- Store in SQLite database
- Rate limit to 1.05s between videos

**Incremental update** (fetch only new videos):

```bash
python -m data_ingestion.update
```

**Sync to ChromaDB** (after database changes):

```bash
python -m data_ingestion.sync_chroma
```

### Retrieval Evaluation

**Generate synthetic queries**:

```bash
python -m retrieval_eval.query_gen.run
```

Output: `retrieval_eval/query_gen/data/queries.csv`

**Run retrieval benchmarks**:

```bash
python -m retrieval_eval.evaluate
```

Output: `retrieval_eval/data/eval_results.jsonl`

**View results interactively**:

```bash
streamlit run retrieval_eval/analysis.py
```

Features:
- Compare BM25, ChromaDB, and Hybrid methods
- Filter by query difficulty and type
- View detailed metrics (MRR, Recall@K)

### Response Evaluation

**Generate answers** (uses hybrid retrieval by default):

```bash
python -m response_eval.generate [--limit N]
```

- `--limit N`: Process only first N queries (useful for testing)
- Output: `response_eval/data/response_results.jsonl`

**View and analyze responses**:

```bash
streamlit run response_eval/viewer.py
```

Features:
- Review generated answers and citations
- Tag responses for quality assessment
- Mark retrieval failures
- Export error analysis to CSV

## Development Tools

### Jupyter Notebook

```bash
jupyter lab
```

Main development notebook: `sandbox.ipynb`

### Query Browser

```bash
streamlit run retrieval_eval/query_gen/viewer.py
```

Browse all generated queries with metadata, difficulty levels, and splits.

## Project Structure

```
yt-answer-engine/
├── data_ingestion/        # YouTube data fetching and storage
│   ├── ingest.py         # Main ingestion (full channel)
│   ├── update.py         # Incremental updates
│   ├── sync_chroma.py    # Vector DB synchronization
│   └── database.py       # SQLite operations
├── retrieval_eval/       # Search method benchmarking
│   ├── evaluate.py       # Run retrieval benchmarks
│   ├── analysis.py       # Streamlit retrieval viewer
│   └── query_gen/        # Synthetic query generation
│       ├── run.py        # Query generation workflow
│       ├── functions.py  # Query generation logic
│       └── viewer.py     # Query browser
├── response_eval/        # Answer quality evaluation
│   ├── generate.py       # Answer generation
│   └── viewer.py         # Streamlit response viewer
├── utils/                # Search implementations
│   ├── answer.py         # Answer generation API
│   ├── search_bm25.py    # BM25 keyword search
│   ├── search_chroma.py  # Semantic search
│   └── search_hybrid.py  # RRF hybrid search
├── prompts/              # LLM system prompts
│   ├── answer_system.md  # Grounding & citation requirements
│   └── answer_user.md    # User prompt template
├── data/                 # Storage (gitignored)
│   ├── videos.db         # SQLite with FTS5
│   └── chroma/           # ChromaDB vector store
├── sandbox.ipynb         # Main development notebook
├── CLAUDE.md             # Detailed implementation docs
└── README.md             # This file
```

## Configuration

### Environment Variables

Required in `.env` file:

- `YOUTUBE_API_KEY`: YouTube Data API v3 key for fetching video metadata and comments
- `OPENAI_API_KEY`: OpenAI API key for answer generation and query synthesis
- `PROXY_USERNAME`: HTTP proxy username for transcript fetching
- `PROXY_PASSWORD`: HTTP proxy password
- `PROXY_URL`: HTTP proxy URL (format: `http://host:port`)

### Channel Configuration

To ingest a different YouTube channel, edit the `CHANNEL_ID` constant in `data_ingestion/ingest.py:16`.

### Model Configuration

The system uses `gpt-4.1-2025-04-14` for both answer generation and query synthesis. To change the model, modify:
- Answer generation: `utils/answer.py` and `response_eval/generate.py`
- Query generation: `retrieval_eval/query_gen/functions.py`

## Additional Information

### Detailed Documentation

For implementation details, data architecture, and development guidelines, see [`CLAUDE.md`](./CLAUDE.md).

### License

This project is licensed under the Apache 2.0 License. See [`LICENSE`](./LICENSE) for details.

### Search Method Comparison

- **BM25**: Fast keyword-based search, good for exact term matches
- **ChromaDB**: Semantic similarity search, good for conceptual matches
- **Hybrid (RRF)**: Combines both methods using Reciprocal Rank Fusion (k=60), typically provides best results

### Data Files

Evaluation data is stored in JSONL and CSV formats:
- `retrieval_eval/query_gen/data/queries.csv` - Generated queries with splits
- `retrieval_eval/data/eval_results.jsonl` - Retrieval benchmark results
- `response_eval/data/response_results.jsonl` - Generated answers with citations
- `response_eval/data/error_analysis-*.csv` - Tagged error analysis exports

### Notes

- All Python modules use relative imports from project root
- Scripts automatically add project root to `sys.path` for imports
- `.bak` files are created when regenerating evaluation results
- Transcript fetching uses exponential backoff for retry logic
