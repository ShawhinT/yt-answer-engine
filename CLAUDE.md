# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**yt-answer-engine** is an AI system that answers technical questions using YouTube videos. The v0 implementation focuses exclusively on video-level retrieval using BM25 (lexical search), not answer generation.

### Core Architecture

The system follows a retrieval-first approach with these key components:

1. **Video Data Ingestion**: Collect video metadata (ID, title, full transcript) from YouTube
2. **Golden Dataset Generation**: Mine natural-language questions from video comments and generate synthetic questions grounded in video content (~450 total)
3. **Query Categorization**: Label questions by type (exact/concept/procedural)
4. **BM25 Retrieval Index**: SQLite FTS5 virtual table indexing video titles and transcripts
5. **Evaluation Pipeline**: Measure Recall@K and MRR using train/test split by video_id (80/20)
6. **Failure Analysis**: Identify BM25 failure modes by query type and source

### Design Principles

- **Video-level retrieval only** (no chunking in v0)
- **Lexical retrieval (BM25)** as baseline before embeddings
- **Separation of concerns**: problem definition, evaluation, and system are distinct
- **Debuggability over performance**: optimize for understanding failure modes

### Explicit Non-Goals (v0)

The v0 implementation deliberately excludes:
- Embeddings or hybrid retrieval
- Query rewriting
- Chunking
- Reranking
- Answer generation

## Development Setup

This project uses **uv** for Python dependency management.

### Prerequisites

- Python 3.13+ (specified in `.python-version`)
- uv package manager

### Environment Setup

```bash
# Sync dependencies (creates/updates .venv)
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
```

### Environment Variables

The project requires API keys in `.env`:
- `OPENAI_API_KEY`: For potential LLM-based question generation
- `YOUTUBE_API_KEY`: For fetching video data and transcripts

## Common Commands

### Dependency Management

```bash
# Add a new dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Update dependencies
uv sync

# Lock dependencies without syncing
uv lock
```

### JupyterLab

The project includes JupyterLab for exploratory work:

```bash
# Start JupyterLab
jupyter lab

# Or via uv:
uv run jupyter lab
```

## Key Dependencies

- **youtube-transcript-api**: Fetch transcripts from YouTube videos
- **pandas**: Data manipulation for eval logs and analysis
- **python-dotenv**: Load environment variables
- **jupyterlab/ipykernel**: Interactive development and analysis

## Project Status

This is an **early-stage project** currently in the planning/specification phase. The `vo.md` file contains the complete v0 specification with step-by-step implementation plan.

### Implementation Roadmap (from vo.md)

1. Video data ingestion (video_id, title, transcript)
2. Question mining from comments + synthetic generation
3. Post-hoc query categorization
4. Train/test split by video_id
5. Build BM25 index (SQLite FTS5)
6. Run retrieval evaluation (Recall@K, MRR)
7. Failure analysis by query type and source

### Success Criteria

- End-to-end retrieval works
- Metrics are reproducible
- Failure cases are explainable
- Clear signal on what to improve next (guides v1)

## Evaluation Methodology

- **Split strategy**: Split by video_id (not by question) to prevent leakage
- **Metrics**: Recall@K (coverage) and MRR (ranking quality)
- **Logging**: Per-question results in JSONL format with retrieved IDs, scores, and gold rank
- **Analysis**: Break down performance by query_type and source (comment vs synthetic)

## Future Directions (v1+)

Likely next steps after v0 baseline:
- Chunked retrieval (timestamps)
- Title-weighted search
- Embedding or hybrid retrieval
- Reranking layer
- Answer generation

## Retrieval Architecture Notes

- **Corpus**: YouTube lecture videos (5-40 min)
- **Retrieval unit**: Entire video (not chunks/segments)
- **Index**: SQLite FTS5 full-text search on title + transcript concatenation
- **Query processing**: Minimal preprocessing (normalize whitespace only)
- **Golden dataset size**: ~3 questions per video across ~150 videos = ~450 questions
