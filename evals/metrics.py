"""Evaluation metrics for experiments.

Standard eval set (used by all experiments):
- Retrieval: recall@1, recall@3, mrr
- Response: (placeholder for future metrics)

Low-level metrics:
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


# =============================================================================
# Standard Eval Set
# =============================================================================


def compute_retrieval_metrics(
    results: list[dict],
    gold_key: str = "gold_video_id",
    retrieved_key: str = "retrieved_ids",
) -> dict:
    """Compute standard retrieval metrics on final results.

    Args:
        results: List of result dicts with gold_video_id and retrieved_ids
        gold_key: Key for gold video ID in each result
        retrieved_key: Key for retrieved IDs list in each result

    Returns:
        Dict with recall@1, recall@3, mrr
    """
    return {
        "recall@1": compute_recall(results, k=1, gold_key=gold_key, retrieved_key=retrieved_key),
        "recall@3": compute_recall(results, k=3, gold_key=gold_key, retrieved_key=retrieved_key),
        "mrr": compute_mrr(results, gold_key=gold_key, retrieved_key=retrieved_key),
    }


def compute_response_metrics(results: list[dict]) -> dict:
    """Compute standard response metrics.

    Args:
        results: List of response result dicts

    Returns:
        Dict with response metrics (placeholder for now)
    """
    # TODO: implement response evals
    return {}


def compute_all_evals(
    retrieval_results: list[dict],
    response_results: list[dict] | None = None,
) -> dict:
    """Compute all standard evals for an experiment run.

    Args:
        retrieval_results: List of retrieval result dicts
        response_results: Optional list of response result dicts

    Returns:
        Dict with retrieval and response metric sub-dicts
    """
    evals = {"retrieval": compute_retrieval_metrics(retrieval_results)}
    if response_results:
        evals["response"] = compute_response_metrics(response_results)
    return evals

