"""Semantic search using ChromaDB."""

import chromadb
from pathlib import Path

CHROMA_PATH = Path("data/chroma")


def _get_collection():
    """Get or create the videos collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name="videos",
        metadata={"hnsw:space": "cosine"}
    )


def add_video(video_id: str, title: str, transcript: str):
    """Add/update a video in the ChromaDB collection.

    Args:
        video_id: YouTube video ID
        title: Video title
        transcript: Full transcript text
    """
    collection = _get_collection()

    # Combine title + transcript for richer embeddings
    document = f"{title}\n\n{transcript}"

    collection.upsert(
        ids=[video_id],
        documents=[document],
        metadatas=[{"title": title}]
    )


def search_with_scores(query: str, limit: int = 10) -> list[tuple[str, float]]:
    """Execute semantic search and return ranked results with scores.

    Args:
        query: Natural language search query
        limit: Max number of results to return

    Returns:
        List of (video_id, distance) tuples, ordered by relevance.
        Note: Lower distance = better match (cosine distance).
    """
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=limit
    )

    # Extract ids and distances
    ids = results["ids"][0]
    distances = results["distances"][0]

    return list(zip(ids, distances))


def get_video_count() -> int:
    """Return number of videos in ChromaDB."""
    collection = _get_collection()
    return collection.count()

