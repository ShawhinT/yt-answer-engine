"""Retrieval evaluation pipeline for BM25 baseline."""

import csv
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.search import search_with_scores

QUERIES_PATH = Path("query_gen/data/queries.csv")
RESULTS_PATH = Path("retrieval_eval/data/eval_results.jsonl")


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────


def recall_at_k(gold_id: str, retrieved_ids: list[str], k: int) -> int:
    """1 if gold_id in top-k retrieved, else 0."""
    return 1 if gold_id in retrieved_ids[:k] else 0


def reciprocal_rank(gold_id: str, retrieved_ids: list[str]) -> float:
    """1/rank if gold found, else 0."""
    try:
        rank = retrieved_ids.index(gold_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def gold_rank(gold_id: str, retrieved_ids: list[str]) -> int:
    """Return 1-indexed rank of gold, or -1 if not found."""
    try:
        return retrieved_ids.index(gold_id) + 1
    except ValueError:
        return -1


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────


def load_queries(split: str = "test") -> list[dict]:
    """Load queries from CSV, filtered by split."""
    queries = []
    with open(QUERIES_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                queries.append(row)
    return queries


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────


def evaluate_split(split: str, k_values: list[int]) -> tuple[list[dict], dict]:
    """Evaluate a single split, return results and aggregate metrics."""
    queries = load_queries(split)
    print(f"Loaded {len(queries)} queries for split='{split}'")

    total_rr = 0.0
    recall_sums = {k: 0 for k in k_values}
    results = []

    for q in queries:
        query_id = q["query_id"]
        query_text = q["query"]
        gold_video_id = q["video_id"]

        # Run BM25 search
        search_results = search_with_scores(query_text, limit=max(k_values))
        retrieved_ids = [r[0] for r in search_results]
        scores = [r[1] for r in search_results]

        # Compute metrics
        rr = reciprocal_rank(gold_video_id, retrieved_ids)
        gr = gold_rank(gold_video_id, retrieved_ids)
        hits = {k: recall_at_k(gold_video_id, retrieved_ids, k) for k in k_values}

        # Accumulate
        total_rr += rr
        for k in k_values:
            recall_sums[k] += hits[k]

        # Log result
        result = {
            "query_id": query_id,
            "query": query_text,
            "gold_video_id": gold_video_id,
            "retrieved_ids": retrieved_ids,
            "scores": scores,
            "gold_rank": gr,
            "rr": rr,
            **{f"hit@{k}": hits[k] for k in k_values},
            # Metadata for analysis
            "split": split,
            "query_type": q["query_type"],
            "difficulty": q["difficulty"],
        }
        results.append(result)

    n = len(queries)
    metrics = {
        "n": n,
        "mrr": total_rr / n if n > 0 else 0,
        "recall": {k: recall_sums[k] / n if n > 0 else 0 for k in k_values},
    }
    return results, metrics


def evaluate():
    """Run evaluation on validation and test splits."""
    k_values = [1, 3]
    splits = ["validation", "test"]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_metrics = {}

    for split in splits:
        results, metrics = evaluate_split(split, k_values)
        all_results.extend(results)
        all_metrics[split] = metrics

    # Write all results
    with open(RESULTS_PATH, "w", encoding="utf-8") as out:
        for result in all_results:
            out.write(json.dumps(result) + "\n")

    # Print aggregate metrics
    print(f"\n{'='*40}")
    print("Results Summary")
    print(f"{'='*40}")
    for split in splits:
        m = all_metrics[split]
        print(f"\n{split.upper()} (n={m['n']})")
        print(f"  MRR:       {m['mrr']:.4f}")
        for k in k_values:
            print(f"  Recall@{k}: {m['recall'][k]:.4f}")

    print(f"\nPer-query results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()

