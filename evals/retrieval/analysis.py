"""Analyze retrieval results by query_type, difficulty - comparing BM25 vs Chroma vs Hybrid."""

import json
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("evals/retrieval/data/eval_results.jsonl")

METHODS = ["bm25", "chroma", "hybrid"]


def compute_metrics(group: list[dict], method: str) -> dict:
    """Compute MRR, R@1, R@3 for a group of results for a given method."""
    n = len(group)
    mrr = sum(r[f"{method}_rr"] for r in group) / n
    r1 = sum(r[f"{method}_hit@1"] for r in group) / n
    r3 = sum(r[f"{method}_hit@3"] for r in group) / n
    return {"mrr": mrr, "r1": r1, "r3": r3}


def print_comparison_header():
    """Print the comparison table header."""
    print(f"  {'':15} | {'n':>3} | {'BM25':>8} {'Chroma':>8} {'Hybrid':>8} | {'BM25':>8} {'Chroma':>8} {'Hybrid':>8} | {'BM25':>8} {'Chroma':>8} {'Hybrid':>8}")
    print(f"  {'':15} | {'':>3} | {'--- MRR ---':^26} | {'--- R@1 ---':^26} | {'--- R@3 ---':^26}")
    print(f"  {'-'*15}-+-{'-'*3}-+-{'-'*26}-+-{'-'*26}-+-{'-'*26}")


def print_comparison_row(key: str, n: int, bm25: dict, chroma: dict, hybrid: dict):
    """Print a single comparison row."""
    print(
        f"  {key:15} | {n:3} | "
        f"{bm25['mrr']:>8.3f} {chroma['mrr']:>8.3f} {hybrid['mrr']:>8.3f} | "
        f"{bm25['r1']:>8.3f} {chroma['r1']:>8.3f} {hybrid['r1']:>8.3f} | "
        f"{bm25['r3']:>8.3f} {chroma['r3']:>8.3f} {hybrid['r3']:>8.3f}"
    )


def analyze():
    """Print breakdown of metrics by different dimensions, comparing BM25 vs Chroma."""
    results = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    # First show breakdown by split
    print(f"\n{'='*120}")
    print("Breakdown by SPLIT (BM25 vs Chroma vs Hybrid)")
    print(f"{'='*120}")
    print_comparison_header()

    split_groups = defaultdict(list)
    for r in results:
        split_groups[r["split"]].append(r)

    for split, group in sorted(split_groups.items()):
        n = len(group)
        bm25_metrics = compute_metrics(group, "bm25")
        chroma_metrics = compute_metrics(group, "chroma")
        hybrid_metrics = compute_metrics(group, "hybrid")
        print_comparison_row(split, n, bm25_metrics, chroma_metrics, hybrid_metrics)

    # Then show breakdown by other dimensions
    dimensions = ["query_type", "difficulty"]

    for dim in dimensions:
        print(f"\n{'='*120}")
        print(f"Breakdown by {dim.upper()} (BM25 vs Chroma vs Hybrid)")
        print(f"{'='*120}")
        print_comparison_header()

        groups = defaultdict(list)
        for r in results:
            groups[r[dim]].append(r)

        for key, group in sorted(groups.items()):
            n = len(group)
            bm25_metrics = compute_metrics(group, "bm25")
            chroma_metrics = compute_metrics(group, "chroma")
            hybrid_metrics = compute_metrics(group, "hybrid")
            print_comparison_row(key, n, bm25_metrics, chroma_metrics, hybrid_metrics)

    # Query-level head-to-head comparison
    print(f"\n{'='*120}")
    print("Query-Level Head-to-Head (based on Reciprocal Rank)")
    print(f"{'='*120}")

    bm25_wins = []
    chroma_wins = []
    hybrid_wins = []
    ties = []

    for r in results:
        bm25_rr = r["bm25_rr"]
        chroma_rr = r["chroma_rr"]
        hybrid_rr = r["hybrid_rr"]
        query_info = {
            "query_id": r["query_id"],
            "query": r["query"],
            "bm25_rr": bm25_rr,
            "chroma_rr": chroma_rr,
            "hybrid_rr": hybrid_rr,
        }

        # Determine winner (highest RR)
        max_rr = max(bm25_rr, chroma_rr, hybrid_rr)
        winners = []
        if bm25_rr == max_rr:
            winners.append("bm25")
        if chroma_rr == max_rr:
            winners.append("chroma")
        if hybrid_rr == max_rr:
            winners.append("hybrid")

        if len(winners) > 1:
            ties.append(query_info)
        elif winners[0] == "bm25":
            bm25_wins.append(query_info)
        elif winners[0] == "chroma":
            chroma_wins.append(query_info)
        else:
            hybrid_wins.append(query_info)

    print(f"\n  BM25 wins: {len(bm25_wins)} | Chroma wins: {len(chroma_wins)} | Hybrid wins: {len(hybrid_wins)} | Ties: {len(ties)}")

    # Show examples where each method won
    if bm25_wins:
        print(f"\n  Sample queries where BM25 won (showing up to 5):")
        for q in sorted(bm25_wins, key=lambda x: x["bm25_rr"], reverse=True)[:5]:
            print(f"    - [{q['query_id']}] \"{q['query'][:50]}...\" (BM25: {q['bm25_rr']:.2f}, Chroma: {q['chroma_rr']:.2f}, Hybrid: {q['hybrid_rr']:.2f})")

    if chroma_wins:
        print(f"\n  Sample queries where Chroma won (showing up to 5):")
        for q in sorted(chroma_wins, key=lambda x: x["chroma_rr"], reverse=True)[:5]:
            print(f"    - [{q['query_id']}] \"{q['query'][:50]}...\" (BM25: {q['bm25_rr']:.2f}, Chroma: {q['chroma_rr']:.2f}, Hybrid: {q['hybrid_rr']:.2f})")

    if hybrid_wins:
        print(f"\n  Sample queries where Hybrid won (showing up to 5):")
        for q in sorted(hybrid_wins, key=lambda x: x["hybrid_rr"], reverse=True)[:5]:
            print(f"    - [{q['query_id']}] \"{q['query'][:50]}...\" (BM25: {q['bm25_rr']:.2f}, Chroma: {q['chroma_rr']:.2f}, Hybrid: {q['hybrid_rr']:.2f})")

    # Summary: which method wins overall
    print(f"\n{'='*120}")
    print("Overall Summary")
    print(f"{'='*120}")
    n = len(results)
    bm25_overall = compute_metrics(results, "bm25")
    chroma_overall = compute_metrics(results, "chroma")
    hybrid_overall = compute_metrics(results, "hybrid")

    print(f"\n  Total queries: {n}")
    print(f"\n  {'Metric':<10} {'BM25':>10} {'Chroma':>10} {'Hybrid':>10} {'Winner':>12}")
    print(f"  {'-'*54}")

    for metric, label in [("mrr", "MRR"), ("r1", "Recall@1"), ("r3", "Recall@3")]:
        bm25_val = bm25_overall[metric]
        chroma_val = chroma_overall[metric]
        hybrid_val = hybrid_overall[metric]
        max_val = max(bm25_val, chroma_val, hybrid_val)
        winners = []
        if bm25_val == max_val:
            winners.append("BM25")
        if chroma_val == max_val:
            winners.append("Chroma")
        if hybrid_val == max_val:
            winners.append("Hybrid")
        winner = "/".join(winners) if len(winners) > 1 else winners[0]
        print(f"  {label:<10} {bm25_val:>10.4f} {chroma_val:>10.4f} {hybrid_val:>10.4f} {winner:>12}")


if __name__ == "__main__":
    analyze()
