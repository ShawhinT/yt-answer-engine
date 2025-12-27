"""Database maintenance utilities."""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/youtube/videos.db")


def rebuild_fts_index():
    """Rebuild the FTS5 index to sync with videos table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO videos_fts(videos_fts) VALUES('rebuild')")

    conn.commit()
    conn.close()
    print("FTS5 index rebuilt successfully")


if __name__ == "__main__":
    rebuild_fts_index()

