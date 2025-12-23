"""One-time sync script to populate ChromaDB from existing SQLite data."""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.database import get_all_video_ids, get_video_by_id
from utils.search_chroma import add_video, get_video_count


def sync():
    """Sync all videos from SQLite to ChromaDB."""
    video_ids = get_all_video_ids()
    print(f"Found {len(video_ids)} videos in SQLite")
    print()

    for i, vid in enumerate(video_ids, 1):
        video = get_video_by_id(vid)
        if video is None:
            print(f"[{i}/{len(video_ids)}] SKIP {vid}: Not found")
            continue

        add_video(video["video_id"], video["title"], video["transcript"])

        # Truncate title for display
        title = video["title"]
        display_title = title[:50] + "..." if len(title) > 50 else title
        print(f"[{i}/{len(video_ids)}] Synced: {display_title}")

    print()
    print("=" * 50)
    print(f"Sync complete! Total in ChromaDB: {get_video_count()}")
    print("=" * 50)


if __name__ == "__main__":
    sync()

