"""Database operations for video ingestion."""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/youtube/videos.db")


def init_db():
    """Create database and tables if they don't exist."""
    # Create data directory
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create videos table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            transcript TEXT NOT NULL
        )
    """)

    # Create FTS5 virtual table for BM25 search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts USING fts5(
            video_id UNINDEXED,
            title,
            transcript,
            content=videos
        )
    """)

    # Create triggers to keep FTS5 synced with videos table
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS videos_ai AFTER INSERT ON videos BEGIN
            INSERT INTO videos_fts(rowid, video_id, title, transcript)
            VALUES (new.rowid, new.video_id, new.title, new.transcript);
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS videos_ad AFTER DELETE ON videos BEGIN
            DELETE FROM videos_fts WHERE rowid = old.rowid;
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS videos_au AFTER UPDATE ON videos BEGIN
            UPDATE videos_fts
            SET title = new.title, transcript = new.transcript
            WHERE rowid = old.rowid;
        END
    """)

    conn.commit()
    conn.close()

    print(f"Database initialized at {DB_PATH}")


def insert_video(video_id: str, title: str, transcript: str):
    """Insert a video into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO videos (video_id, title, transcript)
        VALUES (?, ?, ?)
    """, (video_id, title, transcript))

    conn.commit()
    conn.close()


def get_video_count() -> int:
    """Return total number of videos in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM videos")
    count = cursor.fetchone()[0]

    conn.close()
    return count


def search_videos(query: str, limit: int = 5):
    """Test BM25 search (for validation)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT video_id, title
        FROM videos_fts
        WHERE videos_fts MATCH ?
        LIMIT ?
    """, (query, limit))

    results = cursor.fetchall()
    conn.close()

    return results


def get_all_video_ids() -> list[str]:
    """Get all video IDs from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT video_id FROM videos ORDER BY video_id")
    video_ids = [row[0] for row in cursor.fetchall()]

    conn.close()
    return video_ids


def get_video_by_id(video_id: str) -> dict[str, str] | None:
    """Get video data by ID.

    Args:
        video_id: YouTube video ID

    Returns:
        Dictionary with keys: video_id, title, transcript
        Returns None if video not found
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT video_id, title, transcript
        FROM videos
        WHERE video_id = ?
    """, (video_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "video_id": row[0],
        "title": row[1],
        "transcript": row[2]
    }
