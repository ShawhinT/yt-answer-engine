"""BM25 search utilities using SQLite FTS5."""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/youtube/videos.db")


def escape_fts5_query(query: str) -> str:
    """Escape special FTS5 characters by wrapping tokens in double quotes."""
    # Replace double quotes in the query, then wrap each word in quotes
    query = query.replace('"', "")
    tokens = query.split()
    return " ".join(f'"{token}"' for token in tokens)


def search_with_scores(query: str, limit: int = 10) -> list[tuple[str, float]]:
    """Execute BM25 search and return ranked results with scores.

    Args:
        query: Natural language search query
        limit: Max number of results to return

    Returns:
        List of (video_id, bm25_score) tuples, ordered by relevance.
        Note: Lower BM25 scores = better match in SQLite FTS5.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    escaped_query = escape_fts5_query(query)

    cursor.execute(
        """
        SELECT video_id, bm25(videos_fts)
        FROM videos_fts
        WHERE videos_fts MATCH ?
        ORDER BY bm25(videos_fts)
        LIMIT ?
    """,
        (escaped_query, limit),
    )

    results = cursor.fetchall()
    conn.close()
    return results

