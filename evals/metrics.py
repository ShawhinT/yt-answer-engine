"""Evaluation metrics for experiments.

Standard eval set (used by all experiments):
- Retrieval: recall@1, recall@3, mrr
- Response: {check_name}_rate for all checks in RESPONSE_CHECKS

Low-level metrics:
- recall@k: Whether gold item appears in top-k results
- reciprocal_rank: 1/rank of gold item (MRR component)
"""


# =============================================================================
# Response Quality Checks
# =============================================================================


def check_bad_framing(answer: str) -> bool:
    """Check if response contains bad framing patterns.

    Bad framing occurs when the model exposes internal retrieval architecture
    or frames transcripts/context as something the user provided.

    Args:
        answer: The LLM-generated response text to check

    Returns:
        True if bad framing detected (FAIL), False if clean (PASS)
    """
    # TODO: User will provide specific string checking logic
    # Placeholder implementation:
    answer_lower = answer.lower()

    bad_patterns = [
        # Add specific phrases to check for here
        # Examples:
        # "provided transcript",
        # "based on the context you",
        # "according to the video you provided",
    ]

    for pattern in bad_patterns:
        if pattern in answer_lower:
            return True  # Bad framing detected

    return False  # Clean response


# Registry of all response quality checks
# Add new checks here - they'll automatically be computed
RESPONSE_CHECKS = {
    "bad_framing": check_bad_framing,
    # Future checks:
    # "incomplete_answer": check_incomplete_answer,
    # "citation_mismatch": check_citation_mismatch,
}


# =============================================================================
# Retrieval Metrics
# =============================================================================


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

    Automatically computes failure rates for all checks defined in RESPONSE_CHECKS.

    Args:
        results: List of response result dicts with 'answer' field

    Returns:
        Dict with {check_name}_rate for each check (e.g., "bad_framing_rate")
    """
    if not results:
        return {f"{name}_rate": 0.0 for name in RESPONSE_CHECKS}

    metrics = {}
    for check_name, check_fn in RESPONSE_CHECKS.items():
        failure_count = sum(
            1 for r in results
            if check_fn(r.get("answer", ""))
        )
        metrics[f"{check_name}_rate"] = failure_count / len(results)

    return metrics


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

