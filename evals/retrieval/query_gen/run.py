"""Simple workflow runner for query generation across all videos."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data_ingestion.database import get_all_video_ids, get_video_by_id
from evals.retrieval.query_gen.functions import (
    get_all_comments,
    generate_queries,
    save_queries_to_jsonl
)

OUTPUT_PATH = Path(__file__).parent / "data" / "raw_queries.jsonl"


def main():
    """
    Main workflow:
    1. Get all video IDs from database
    2. For each video:
       - Get title and transcript from DB
       - Get comments from YouTube API
       - Generate queries via OpenAI
       - Save to JSONL
    """
    print("Starting query generation workflow...")
    print("=" * 80)

    # Get all videos
    video_ids = get_all_video_ids()
    print(f"Found {len(video_ids)} videos in database")
    print()

    # Process each video
    success_count = 0
    failed_count = 0

    for i, video_id in enumerate(video_ids, 1):
        try:
            print(f"[{i}/{len(video_ids)}] Processing {video_id}...")

            # Get video data from database
            video_data = get_video_by_id(video_id)
            if video_data is None:
                print(f"  ERROR: Video not found in database")
                failed_count += 1
                continue

            title = video_data["title"]
            transcript = video_data["transcript"]

            # Truncate title for display
            display_title = title[:60] + "..." if len(title) > 60 else title
            print(f"  Title: {display_title}")

            # Get comments from YouTube
            print(f"  Fetching comments...")
            comments = get_all_comments(video_id)
            print(f"  Found {len(comments)} comments")

            # Generate queries (always 9 queries as per prompt)
            print(f"  Generating queries...")
            query_response = generate_queries(
                video_title=title,
                transcript=transcript,
                comments=comments
            )

            # Save to JSONL
            save_queries_to_jsonl(
                video_id=video_id,
                video_title=title,
                query_response=query_response,
                output_path=OUTPUT_PATH
            )

            print(f"  ✓ Generated and saved {len(query_response.queries)} queries")
            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed_count += 1

        print()

    # Summary
    print("=" * 80)
    print("Query generation complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
