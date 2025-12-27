"""Retrieval evaluation metrics.

Standardized metrics for evaluating retrieval quality:
- recall@k: Whether gold item appears in top-k results
- reciprocal_rank: 1/rank of gold item (MRR component)
"""


def recall_at_k(gold_id: str, retrieved_ids: list[str], k: int) -> int:
    """Check if gold_id appears in top-k retrieved results.

    Args:
        gold_id: The correct/gold video ID
        retrieved_ids: List of retrieved video IDs in ranked order
        k: Number of top results to check

    Returns:
        1 if gold_id in top-k, else 0
    """
    return 1 if gold_id in retrieved_ids[:k] else 0


def reciprocal_rank(gold_id: str, retrieved_ids: list[str]) -> float:
    """Compute reciprocal rank of gold item.

    Args:
        gold_id: The correct/gold video ID
        retrieved_ids: List of retrieved video IDs in ranked order

    Returns:
        1/rank if gold found, else 0.0
    """
    try:
        rank = retrieved_ids.index(gold_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def compute_recall(
    results: list[dict],
    k: int,
    gold_key: str = "gold_video_id",
    retrieved_key: str = "retrieved_ids",
) -> float:
    """Compute recall@k across a list of results.

    Args:
        results: List of result dicts
        k: Number of top results to check
        gold_key: Key for gold video ID in each result
        retrieved_key: Key for retrieved IDs list in each result

    Returns:
        Recall@k as a float between 0 and 1
    """
    if not results:
        return 0.0
    total = sum(
        recall_at_k(r[gold_key], r[retrieved_key], k)
        for r in results
    )
    return total / len(results)


def compute_mrr(
    results: list[dict],
    gold_key: str = "gold_video_id",
    retrieved_key: str = "retrieved_ids",
) -> float:
    """Compute Mean Reciprocal Rank across a list of results.

    Args:
        results: List of result dicts
        gold_key: Key for gold video ID in each result
        retrieved_key: Key for retrieved IDs list in each result

    Returns:
        MRR as a float between 0 and 1
    """
    if not results:
        return 0.0
    total_rr = sum(
        reciprocal_rank(r[gold_key], r[retrieved_key])
        for r in results
    )
    return total_rr / len(results)

