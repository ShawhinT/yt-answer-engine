# yt-answer-engine

AI-powered question answering system that grounds responses in YouTube video transcripts using hybrid retrieval and LLM synthesis. This system combines multiple retrieval methods (BM25 keyword search, semantic search via ChromaDB, and hybrid fusion) to find relevant video content, then uses OpenAI GPT-4.1 to generate accurate answers with inline citations. It includes a complete evaluation pipeline for benchmarking retrieval quality and answer accuracy.

[Video Explainer](https://youtu.be/2peE6mwoiXs?si=dI-bjuZZJ6s1P2vx)

## Prerequisites

- **Python**: >= 3.13
- **Package Manager**: `uv`
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

Using `uv`:

```bash
uv sync
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

## Core Workflows

### Data Ingestion

**Initial ingestion** (fetch all videos from channel):

```bash
python -m data_ingestion.ingest
```

This will:
- Fetch all video IDs from the hardcoded channel (see `data_ingestion/ingest.py` to modify)
- Filter out videos shorter than 50 seconds
- Download transcripts via proxy with retry logic
- Store in SQLite database
- Rate limit to 1.05s between videos

Note: Transcript syncing to ChromaDB is handled automatically by individual experiments.

### Query Generation

**Generate synthetic queries**:

```bash
python -m utils.query_gen.run
```

Output: `data/queries/qset_v01/queries.csv`

**Browse generated queries**:

```bash
streamlit run utils/query_gen/viewer.py
```

### Experiments

The project uses a standardized experiment framework for running and comparing different retrieval and answer generation approaches. Each experiment is self-contained with its own configuration and outputs.

**Run a specific experiment**:

```bash
python -m utils.experiments --exp exp_001 [--run-id r001] [--max-queries N]
```

**Run all experiments**:

```bash
python -m utils.experiments --all [--max-queries N]
```

**Recompute metrics without regenerating outputs**:

```bash
python -m utils.experiments --exp exp_001 --run-id r001 --no-outputs
```

Options:
- `--exp`: Experiment ID to run (e.g., `exp_001`)
- `--all`: Run all experiments sequentially
- `--run-id`: Specific run ID (e.g., `r001`). Auto-generated if not provided
- `--max-queries N`: Limit number of queries to process per split (useful for testing)
- `--no-outputs`: Skip output generation, only recompute metrics from existing files

**View experiment results**:

```bash
streamlit run experiments/viewer.py
```

Features:
- Compare multiple experiment runs side-by-side
- View retrieval metrics (MRR, Recall@K) for dev and test splits
- Track experiment metadata and configurations
- Filter and sort by performance metrics
- Results are automatically registered in `experiments/registry.yaml`

See `experiments/contract.md` for the experiment data contract and requirements.

### Evaluation Viewer

**View detailed evaluation results**:

```bash
streamlit run evals/viewer.py
```

Features:
- Review generated answers and citations
- Tag responses for quality assessment
- Mark retrieval failures
- Export error analysis to CSV
- Compare answer quality across experiments

### Interactive Chat UI

**Launch chat interface**:

```bash
streamlit run ui/chat.py
```

Features:
- Real-time question answering with citations
- Multi-turn conversations
- Label conversation traces with notes
- Save traces to JSONL for analysis

## Project Structure

```
yt-answer-engine/
├── data_ingestion/        # YouTube data fetching and storage
│   ├── ingest.py         # Main ingestion (full channel)
│   └── database.py       # SQLite operations
├── experiments/          # Experiment framework
│   ├── contract.md       # Experiment data contract
│   ├── registry.yaml     # Experiment run registry
│   ├── viewer.py         # Streamlit experiment comparison viewer
│   └── exp_XXX/          # Individual experiments
│       ├── experiment.yaml   # Experiment metadata
│       ├── src/
│       │   └── generate.py   # Main experiment script
│       └── runs/         # Experiment run outputs
│           └── {run_id}/
│               ├── retrieval.jsonl    # Retrieval results
│               ├── responses.jsonl    # Generated answers
│               ├── metrics.json       # Computed metrics
│               └── run_receipt.json   # Run metadata
├── evals/                # Evaluation tools
│   ├── viewer.py         # Streamlit evaluation viewer
│   ├── metrics.py        # Metrics computation utilities
│   └── tags.json         # Tag definitions for evaluation
├── ui/                   # User interfaces
│   └── chat.py           # Streamlit chat interface
├── utils/                # Shared utilities
│   ├── answer.py         # Answer generation API
│   ├── models.py         # Pydantic models for responses
│   ├── data.py           # Data loading utilities
│   ├── git.py            # Git metadata utilities
│   ├── experiments.py    # Experiment management utilities
│   └── query_gen/        # Synthetic query generation
│       ├── run.py        # Query generation workflow
│       ├── functions.py  # Query generation logic
│       └── viewer.py     # Query browser
├── data/                 # Storage (gitignored)
│   ├── youtube/          # YouTube-sourced data
│   │   └── videos.db     # SQLite with FTS5
│   └── queries/          # Query datasets
│       └── qset_v01/     # Query set version 1
├── sandbox.ipynb         # Main development notebook
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

To ingest a different YouTube channel, edit the `CHANNEL_ID` constant in `data_ingestion/ingest.py`.

### Model Configuration

The system uses `gpt-4.1-2025-04-14` for both answer generation and query synthesis. To change the model, modify:
- Answer generation: `utils/answer.py`
- Query generation: `utils/query_gen/functions.py`

## Additional Information

### License

This project is licensed under the Apache 2.0 License. See [`LICENSE`](./LICENSE) for details.

### Experiment Framework

The project uses a standardized experiment framework to ensure reproducibility and comparability:
- Each experiment follows a strict data contract (see `experiments/contract.md`)
- Experiments are self-contained with their own configurations and code
- All experiments output standardized formats for metrics and responses
- The `utils/experiments.py` module provides shared utilities for metrics computation

### Data Files

Key data files:
- `data/queries/qset_v01/queries.csv` - Generated queries with dev/test splits
- `experiments/*/runs/*/retrieval.jsonl` - Retrieval results per experiment run
- `experiments/*/runs/*/responses.jsonl` - Generated answers with citations
- `experiments/*/runs/*/metrics.json` - Computed evaluation metrics
- `experiments/registry.yaml` - Registry of all experiment runs
- `ui/data/chat_traces.jsonl` - Saved chat conversation traces

### Notes

- All Python modules use relative imports from project root
- Experiments automatically add project root to `sys.path` for imports
- Transcript fetching uses exponential backoff for retry logic
- The experiment viewer supports comparing multiple runs side-by-side
