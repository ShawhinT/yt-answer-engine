"""
exp_001 Main Experiment Logic

Orchestrates the experiment pipeline:
1. Generates outputs (retrieval + responses)
2. Runs evaluation
3. Updates registry.md with results
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

from evals.metrics import recall_at_k, reciprocal_rank
from utils.answer import generate_answer
from utils.data import load_queries
from utils.git import get_git_info
from utils.models import AnswerResponse

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
REGISTRY_PATH = EXP_DIR.parent / "registry.md"

# Experiment configuration
CONTEXT_LIMIT = 3  # Number of videos to retrieve for context


def run_experiment(run_id: str, max_queries: int | None = None):
    """
    Execute full experiment pipeline on both dev and test splits.

    Args:
        run_id: Run identifier (e.g., r001)
        max_queries: Maximum number of queries to process per split (default: None = all)
    """
    for split in ["dev", "test"]:
        _run_split(split, run_id, max_queries)


def _run_split(split: str, run_id: str, max_queries: int | None = None):
    """
    Execute experiment pipeline for a single split.

    Args:
        split: Dataset split to use (dev or test)
        run_id: Run identifier (e.g., r001)
        max_queries: Maximum number of queries to process (default: None = all)
    """
    print(f"{'=' * 72}")
    print(f"Running exp_001 | split={split} | run_id={run_id}")
    print(f"{'=' * 72}")

    # Create run directory
    run_dir = EXP_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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

    # Track metrics
    methods = ["bm25", "chroma", "hybrid"]
    k_values = [1, 3]

    total_rr = {method: 0.0 for method in methods}
    recall_sums = {method: {k: 0 for k in k_values} for method in methods}

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
        max_k = max(k_values)
        bm25_results = search_bm25.search_with_scores(query_text, limit=max_k)
        chroma_results = search_chroma.search_with_scores(query_text, limit=max_k)
        hybrid_results = search_hybrid.search_with_scores(query_text, limit=max_k)

        search_outputs = {
            "bm25": bm25_results,
            "chroma": chroma_results,
            "hybrid": hybrid_results,
        }

        # Build retrieval result
        retrieval_result = {
            "query_id": query_id,
            "query": query_text,
            "gold_video_id": gold_video_id,
            "methods": {},
        }

        # Compute metrics for each method
        for method in methods:
            retrieved_ids = [r[0] for r in search_outputs[method]]

            rr = reciprocal_rank(gold_video_id, retrieved_ids)

            # Accumulate
            total_rr[method] += rr
            for k in k_values:
                recall_sums[method][k] += recall_at_k(gold_video_id, retrieved_ids, k)

            # Add to retrieval result
            retrieval_result["methods"][method] = {
                "retrieved": [
                    {"video_id": vid, "rank": idx + 1, "score": score}
                    for idx, (vid, score) in enumerate(search_outputs[method])
                ]
            }

        retrieval_results.append(retrieval_result)

        # Generate answer using hybrid retrieval results (top CONTEXT_LIMIT videos)
        hybrid_retrieved_ids = [r[0] for r in hybrid_results[:CONTEXT_LIMIT]]
        answer_response = generate_answer(query_text, hybrid_retrieved_ids, client, PROMPTS_DIR)

        # Build response result
        response_result = {
            "query_id": query_id,
            "query": query_text,
            "gold_video_id": gold_video_id,
            "context_video_ids": hybrid_retrieved_ids,
            "answer": answer_response.answer,
            "citations": [c.model_dump() for c in answer_response.citations],
            "metadata": {
                "answer_length": len(answer_response.answer),
                "citation_count": len(answer_response.citations),
                "contains_refusal": any(
                    phrase in answer_response.answer.lower()
                    for phrase in ["cannot answer", "don't have enough", "insufficient"]
                ),
            },
        }

        response_results.append(response_result)

    # Compute aggregate metrics
    n = len(queries)
    retrieval_metrics = {}
    for method in methods:
        retrieval_metrics[method] = {
            "recall@1": recall_sums[method][1] / n if n > 0 else 0,
            "recall@3": recall_sums[method][3] / n if n > 0 else 0,
            "mrr": total_rr[method] / n if n > 0 else 0,
        }

    # Compute response metrics
    gold_cited_count = sum(
        1
        for r in response_results
        if any(c["video_id"] == r["gold_video_id"] for c in r["citations"])
    )

    total_citations = sum(len(r["citations"]) for r in response_results)
    correct_citations = sum(
        len([c for c in r["citations"] if c["video_id"] in r["context_video_ids"]])
        for r in response_results
    )

    refusal_count = sum(r["metadata"]["contains_refusal"] for r in response_results)
    total_answer_length = sum(r["metadata"]["answer_length"] for r in response_results)

    response_metrics = {
        "gold_cited_rate": gold_cited_count / n if n > 0 else 0,
        "citation_precision": correct_citations / total_citations if total_citations > 0 else 0,
        "refusal_rate": refusal_count / n if n > 0 else 0,
        "avg_answer_length": total_answer_length / n if n > 0 else 0,
        "avg_citation_count": total_citations / n if n > 0 else 0,
    }

    # Write outputs
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

    # 2. Retrieval results
    with open(run_dir / "retrieval.jsonl", "w") as f:
        for result in retrieval_results:
            f.write(json.dumps(result) + "\n")

    # 3. Response results
    with open(run_dir / "responses.jsonl", "w") as f:
        for result in response_results:
            f.write(json.dumps(result) + "\n")

    # 4. Metrics
    metrics = {
        "exp_id": "exp_001",
        "run_id": run_id,
        "query_set_id": "qset_v01",
        "split": split,
        "query_count": n,
        "retrieval": retrieval_metrics,
        "response": response_metrics,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'=' * 72}")
    print("Results Summary")
    print(f"{'=' * 72}")
    print(f"\nRetrieval Metrics (n={n}):")
    print(f"  {'Metric':<20} {'BM25':>10} {'Chroma':>10} {'Hybrid':>10}")
    print(f"  {'-' * 50}")

    for metric_name in ["recall@1", "recall@3", "mrr"]:
        print(
            f"  {metric_name:<20} "
            f"{retrieval_metrics['bm25'][metric_name]:>10.4f} "
            f"{retrieval_metrics['chroma'][metric_name]:>10.4f} "
            f"{retrieval_metrics['hybrid'][metric_name]:>10.4f}"
        )

    print("\nResponse Metrics:")
    print(f"  Gold cited rate:      {response_metrics['gold_cited_rate']:.4f}")
    print(f"  Citation precision:   {response_metrics['citation_precision']:.4f}")
    print(f"  Refusal rate:         {response_metrics['refusal_rate']:.4f}")
    print(f"  Avg answer length:    {response_metrics['avg_answer_length']:.1f}")
    print(f"  Avg citation count:   {response_metrics['avg_citation_count']:.2f}")

    # Update registry.md
    update_registry_md(run_id, metrics)

    print(f"\n{'=' * 72}")
    print(f"Run complete! Results saved to: {run_dir}")
    print(f"{'=' * 72}\n")


def update_registry_md(run_id: str, metrics: dict):
    """Update registry.md with best metrics for exp_001."""
    if not REGISTRY_PATH.exists():
        # Create initial registry if it doesn't exist
        REGISTRY_PATH.write_text("""# Experiment Registry

| ID | Description | Query Set | Best Run | Recall@3 (Hybrid) | MRR (Hybrid) | Notes |
|----|-------------|-----------|----------|-------------------|--------------|-------|

---

## Future Experiments

Ideas for rapid experimentation:
- exp_002: Chunking strategies (500-token chunks with overlap)
- exp_003: Different embedding models (text-embedding-3-large)
- exp_004: Agentic retrieval (multi-hop reasoning)
- exp_005: Transcript rewriting for better retrieval
""")

    # Read current registry
    content = REGISTRY_PATH.read_text()

    # Check if exp_001 already exists in registry
    ret = metrics["retrieval"]["hybrid"]
    recall3 = ret["recall@3"]
    mrr = ret["mrr"]

    exp_001_line = f"| exp_001 | Baseline: Hybrid RAG (BM25+Chroma, RRF k=60) | qset_v01 | {run_id} | {recall3:.2f} | {mrr:.2f} | Initial baseline |"

    if "exp_001" in content:
        # Update existing entry (replace the line)
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("| exp_001"):
                lines[i] = exp_001_line
                break
        content = "\n".join(lines)
    else:
        # Add new entry (insert after header row)
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("|----|"):
                lines.insert(i + 1, exp_001_line)
                break
        content = "\n".join(lines)

    REGISTRY_PATH.write_text(content)
    print(f"Updated {REGISTRY_PATH}")

