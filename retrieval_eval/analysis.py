"""Analyze retrieval results by query_type, difficulty."""

import json
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("retrieval_eval/data/eval_results.jsonl")


def analyze():
    """Print breakdown of metrics by different dimensions."""
    results = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    # First show breakdown by split
    print(f"\n{'='*50}")
    print("Breakdown by SPLIT")
    print(f"{'='*50}")

    split_groups = defaultdict(list)
    for r in results:
        split_groups[r["split"]].append(r)

    for split, group in sorted(split_groups.items()):
        n = len(group)
        mrr = sum(r["rr"] for r in group) / n
        r1 = sum(r["hit@1"] for r in group) / n
        r3 = sum(r["hit@3"] for r in group) / n
        print(f"  {split:15} | n={n:3} | MRR={mrr:.3f} | R@1={r1:.3f} | R@3={r3:.3f}")

    # Then show breakdown by other dimensions
    dimensions = ["query_type", "difficulty"]

    for dim in dimensions:
        print(f"\n{'='*50}")
        print(f"Breakdown by {dim.upper()}")
        print(f"{'='*50}")

        groups = defaultdict(list)
        for r in results:
            groups[r[dim]].append(r)

        for key, group in sorted(groups.items()):
            n = len(group)
            mrr = sum(r["rr"] for r in group) / n
            r1 = sum(r["hit@1"] for r in group) / n
            r3 = sum(r["hit@3"] for r in group) / n

            print(f"  {key:15} | n={n:3} | MRR={mrr:.3f} | R@1={r1:.3f} | R@3={r3:.3f}")


if __name__ == "__main__":
    analyze()

