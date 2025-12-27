"""Hybrid search combining BM25 and semantic search via RRF."""

import search_bm25
import search_chroma


def search_with_scores(query: str, limit: int = 10, k: int = 60) -> list[tuple[str, float]]:
    """Execute hybrid search using Reciprocal Rank Fusion.

    Args:
        query: Search query
        limit: Max results to return
        k: RRF constant (default 60)

    Returns:
        List of (video_id, rrf_score) tuples, sorted descending (higher = better).
    """
    # Get ranked results from both systems
    bm25_results = search_bm25.search_with_scores(query, limit=limit)
    chroma_results = search_chroma.search_with_scores(query, limit=limit)

    # Build rank lookup (1-indexed)
    bm25_ranks = {vid: rank for rank, (vid, _) in enumerate(bm25_results, 1)}
    chroma_ranks = {vid: rank for rank, (vid, _) in enumerate(chroma_results, 1)}

    # Calculate RRF scores for all videos
    all_videos = set(bm25_ranks) | set(chroma_ranks)
    scores = {}
    for vid in all_videos:
        score = 0
        if vid in bm25_ranks:
            score += 1 / (k + bm25_ranks[vid])
        if vid in chroma_ranks:
            score += 1 / (k + chroma_ranks[vid])
        scores[vid] = score

    # Sort by RRF score descending, return top limit
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:limit]

