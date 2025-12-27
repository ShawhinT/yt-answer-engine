"""
exp_002 Output Generation

Generates retrieval and response outputs for evaluation.
This script only produces outputs - evals are computed by utils/experiments.py.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.metrics import compute_retrieval_metrics
from utils.answer import generate_answer
from utils.data import load_queries
from utils.git import get_git_info

# Add src/utils to path for experiment-specific search methods
sys.path.insert(0, str(Path(__file__).parent / "utils"))

import search_bm25
import search_chroma
import search_hybrid

# Load environment variables
load_dotenv()

# Paths (relative to experiment root)
EXP_DIR = Path(__file__).parent.parent
SRC_DIR = EXP_DIR / "src"
QUERIES_PATH = PROJECT_ROOT / "data/queries/qset_v01/queries.csv"
PROMPTS_DIR = SRC_DIR / "prompts"

# Experiment configuration
CONTEXT_LIMIT = 3  # Number of videos to retrieve for context


def compute_extra_evals(retrieval_results: list[dict]) -> dict:
    """Compute experiment-specific metrics (bm25/chroma breakdowns).

    Args:
        retrieval_results: List of retrieval result dicts with method-specific results

    Returns:
        Dict with extra metrics not included in standard eval set
    """
    # Build per-method result lists for computing metrics
    methods = ["bm25", "chroma"]
    extra = {}

    for method in methods:
        # Build results in standard format for compute_retrieval_metrics
        method_results = []
        for r in retrieval_results:
            method_results.append({
                "gold_video_id": r["gold_video_id"],
                "retrieved_ids": [v["video_id"] for v in r["methods"][method]["retrieved"]],
            })
        extra[method] = compute_retrieval_metrics(method_results)

    return extra


def generate(run_id: str, max_queries: int | None = None) -> None:
    """
    Generate outputs for both dev and test splits.

    Produces retrieval.jsonl, responses.jsonl, extra.jsonl, and run_receipt.json.
    Evals are computed separately by utils/experiments.py.

    Args:
        run_id: Run identifier (e.g., r001)
        max_queries: Maximum number of queries to process per split (default: None = all)
    """
    # Create run directory and clear output files (allows re-running same run_id)
    run_dir = EXP_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ["retrieval.jsonl", "responses.jsonl", "extra.jsonl"]:
        (run_dir / f).unlink(missing_ok=True)

    # Generate outputs for both splits
    for split in ["dev", "test"]:
        _generate_split(split, run_id, max_queries)


def _generate_split(split: str, run_id: str, max_queries: int | None = None) -> None:
    """
    Generate outputs for a single split.

    Args:
        split: Dataset split to use (dev or test)
        run_id: Run identifier (e.g., r001)
        max_queries: Maximum number of queries to process (default: None = all)
    """
    print(f"{'=' * 72}")
    print(f"Generating exp_001 | split={split} | run_id={run_id}")
    print(f"{'=' * 72}")

    # Run directory (created by generate)
    run_dir = EXP_DIR / "runs" / run_id

    # Load queries
    queries = load_queries(QUERIES_PATH, split)
    if max_queries:
        queries = queries[:max_queries]
    print(f"\nLoaded {len(queries)} queries for split='{split}'")

    if not queries:
        print(f"ERROR: No queries found for split '{split}'")
        return

    # Get git info
    git_info = get_git_info(EXP_DIR)

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Track results
    methods = ["bm25", "chroma", "hybrid"]
    retrieval_results = []
    response_results = []

    # Process each query
    print("\nProcessing queries...")
    for i, q in enumerate(queries, 1):
        query_id = q["query_id"]
        query_text = q["query"]
        gold_video_id = q["video_id"]

        if i % 10 == 0 or i == 1:
            print(f"  [{i}/{len(queries)}] {query_id}")

        # Run all search methods
        bm25_results = search_bm25.search_with_scores(query_text, limit=CONTEXT_LIMIT)
        chroma_results = search_chroma.search_with_scores(query_text, limit=CONTEXT_LIMIT)
        hybrid_results = search_hybrid.search_with_scores(query_text, limit=CONTEXT_LIMIT)

        search_outputs = {
            "bm25": bm25_results,
            "chroma": chroma_results,
            "hybrid": hybrid_results,
        }

        # Build retrieval result with final (hybrid) results for standard metrics
        hybrid_retrieved_ids = [r[0] for r in hybrid_results]
        retrieval_result = {
            "query_id": query_id,
            "query": query_text,
            "gold_video_id": gold_video_id,
            "split": split,  # For grouping by split in metrics computation
            "retrieved_ids": hybrid_retrieved_ids,  # Final results for standard metrics
            "methods": {},  # Per-method details for extra evals
        }

        # Add per-method details for extra evals
        for method in methods:
            retrieval_result["methods"][method] = {
                "retrieved": [
                    {"video_id": vid, "rank": idx + 1, "score": score}
                    for idx, (vid, score) in enumerate(search_outputs[method])
                ]
            }

        retrieval_results.append(retrieval_result)

        # Generate answer using hybrid retrieval results
        answer_response = generate_answer(query_text, hybrid_retrieved_ids, client, PROMPTS_DIR)

        # Build response result
        response_result = {
            "query_id": query_id,
            "query": query_text,
            "gold_video_id": gold_video_id,
            "context_video_ids": hybrid_retrieved_ids,
            "answer": answer_response.answer,
            "citations": [c.model_dump() for c in answer_response.citations],
        }

        response_results.append(response_result)

    # Compute extra evals (experiment-specific: bm25/chroma breakdown)
    extra_evals = compute_extra_evals(retrieval_results)

    # Write outputs
    n = len(queries)
    print(f"\nWriting outputs to {run_dir}/")

    # 1. Run receipt
    run_receipt = {
        "exp_id": "exp_001",
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": git_info["sha"],
        "git_dirty": git_info["dirty"],
        "query_set_id": "qset_v01",
        "split": split,
        "query_count": n,
        "videos_db_path": "../../data/youtube/videos.db",
        "chroma_path": "../../data/youtube/chroma",
        "prompt_files": {
            "system": "prompts/answer_system.md",
            "user": "prompts/answer_user.md",
        },
        "models": {
            "embedding": "all-MiniLM-L6-v2",
            "answer": "gpt-4.1-2025-04-14",
        },
        "config": {
            "retrieval_methods": methods,
            "response_method": "hybrid",
            "response_limit": CONTEXT_LIMIT,
            "rrf_k": 60,
        },
    }

    with open(run_dir / "run_receipt.json", "w") as f:
        json.dump(run_receipt, f, indent=2)

    # 2. Retrieval results (append mode - files cleared at run start)
    with open(run_dir / "retrieval.jsonl", "a") as f:
        for result in retrieval_results:
            f.write(json.dumps(result) + "\n")

    # 3. Response results (append mode - files cleared at run start)
    with open(run_dir / "responses.jsonl", "a") as f:
        for result in response_results:
            f.write(json.dumps(result) + "\n")

    # 4. Extra evals (append mode - one line per split)
    extra_entry = {"split": split, **extra_evals}
    with open(run_dir / "extra.jsonl", "a") as f:
        f.write(json.dumps(extra_entry) + "\n")

    # Print summary
    print(f"\n{'=' * 72}")
    print("Generation Summary")
    print(f"{'=' * 72}")
    print(f"\nGenerated outputs for {n} queries (split='{split}')")
    print(f"Extra metrics (bm25/chroma breakdown) saved to extra.jsonl")

    print(f"\n{'=' * 72}")
    print(f"Generation complete! Outputs saved to: {run_dir}")
    print(f"{'=' * 72}\n")

