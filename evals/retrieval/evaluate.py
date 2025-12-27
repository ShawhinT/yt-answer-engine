"""Retrieval evaluation pipeline comparing BM25 and Chroma search."""

import csv
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.search_bm25 import search_with_scores as bm25_search
from utils.search_chroma import search_with_scores as chroma_search
from utils.search_hybrid import search_with_scores as hybrid_search

QUERIES_PATH = Path("evals/retrieval/query_gen/data/queries.csv")
RESULTS_PATH = Path("evals/retrieval/data/eval_results.jsonl")

METHODS = ["bm25", "chroma", "hybrid"]


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

    # Track metrics per method
    total_rr = {method: 0.0 for method in METHODS}
    recall_sums = {method: {k: 0 for k in k_values} for method in METHODS}
    results = []

    for q in queries:
        query_id = q["query_id"]
        query_text = q["query"]
        gold_video_id = q["video_id"]

        # Run all search methods
        bm25_results = bm25_search(query_text, limit=max(k_values))
        chroma_results = chroma_search(query_text, limit=max(k_values))
        hybrid_results = hybrid_search(query_text, limit=max(k_values))

        search_outputs = {
            "bm25": bm25_results,
            "chroma": chroma_results,
            "hybrid": hybrid_results,
        }

        # Build result dict
        result = {
            "query_id": query_id,
            "query": query_text,
            "gold_video_id": gold_video_id,
            # Metadata for analysis
            "split": split,
            "query_type": q["query_type"],
            "difficulty": q["difficulty"],
        }

        # Compute metrics for each method
        for method in METHODS:
            retrieved_ids = [r[0] for r in search_outputs[method]]
            scores = [r[1] for r in search_outputs[method]]

            rr = reciprocal_rank(gold_video_id, retrieved_ids)
            gr = gold_rank(gold_video_id, retrieved_ids)
            hits = {k: recall_at_k(gold_video_id, retrieved_ids, k) for k in k_values}

            # Accumulate
            total_rr[method] += rr
            for k in k_values:
                recall_sums[method][k] += hits[k]

            # Add method-prefixed fields to result
            result[f"{method}_retrieved_ids"] = retrieved_ids
            result[f"{method}_scores"] = scores
            result[f"{method}_gold_rank"] = gr
            result[f"{method}_rr"] = rr
            for k in k_values:
                result[f"{method}_hit@{k}"] = hits[k]

        results.append(result)

    n = len(queries)
    metrics = {}
    for method in METHODS:
        metrics[method] = {
            "n": n,
            "mrr": total_rr[method] / n if n > 0 else 0,
            "recall": {k: recall_sums[method][k] / n if n > 0 else 0 for k in k_values},
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
    print(f"\n{'='*72}")
    print("Results Summary (BM25 vs Chroma vs Hybrid)")
    print(f"{'='*72}")

    for split in splits:
        m = all_metrics[split]
        n = m["bm25"]["n"]
        print(f"\n{split.upper()} (n={n})")
        print(f"  {'Metric':<12} {'BM25':>10} {'Chroma':>10} {'Hybrid':>10}")
        print(f"  {'-'*42}")

        # MRR
        bm25_mrr = m["bm25"]["mrr"]
        chroma_mrr = m["chroma"]["mrr"]
        hybrid_mrr = m["hybrid"]["mrr"]
        print(f"  {'MRR':<12} {bm25_mrr:>10.4f} {chroma_mrr:>10.4f} {hybrid_mrr:>10.4f}")

        # Recall@k
        for k in k_values:
            bm25_r = m["bm25"]["recall"][k]
            chroma_r = m["chroma"]["recall"][k]
            hybrid_r = m["hybrid"]["recall"][k]
            print(f"  {f'Recall@{k}':<12} {bm25_r:>10.4f} {chroma_r:>10.4f} {hybrid_r:>10.4f}")

    print(f"\nPer-query results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()
