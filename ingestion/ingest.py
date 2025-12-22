"""Main video ingestion script."""

import os
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from . import database

# Load environment variables
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = "UCa9gErQ9AE5jT2DZLjXBIdA"


def get_all_video_ids(channel_id: str) -> list[str]:
    """Get all video IDs from channel (with pagination)."""
    video_ids = []
    next_page_token = None

    url = "https://www.googleapis.com/youtube/v3/search"

    while True:
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "maxResults": 50,
            "pageToken": next_page_token,
            "key": API_KEY
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract video IDs from this page
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            video_ids.append(video_id)

        # Check for next page
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids


def get_video_title(video_id: str) -> str:
    """Get video title from video ID."""
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": API_KEY
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    items = data.get("items", [])

    if not items:
        raise ValueError(f"Video not found: {video_id}")

    return items[0]["snippet"]["title"]


def get_transcript(video_id: str) -> str | None:
    """Get transcript for video, return None if unavailable."""
    try:
        # Get transcript list
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to get manually created transcript first
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except NoTranscriptFound:
            # Fall back to auto-generated
            transcript = transcript_list.find_generated_transcript(["en"])

        # Fetch the transcript
        transcript_data = transcript.fetch()

        # Join all text segments with spaces
        full_transcript = " ".join([entry["text"] for entry in transcript_data])

        return full_transcript

    except (TranscriptsDisabled, NoTranscriptFound):
        # No transcript available
        return None
    except Exception as e:
        # Other errors
        print(f"    Error fetching transcript: {e}")
        return None


def main():
    """Main ingestion pipeline."""
    print("Starting ingestion...")

    # 1. Initialize database
    database.init_db()
    print()

    # 2. Get channel ID
    print(f"Fetching channel ID for @{CHANNEL_HANDLE}...")
    channel_id = get_channel_id(CHANNEL_HANDLE)
    print(f"Channel ID: {channel_id}")
    print()

    # 3. Get all video IDs
    print("Fetching all video IDs...")
    video_ids = get_all_video_ids(channel_id)
    print(f"Found {len(video_ids)} videos")
    print()

    # 4. For each video: fetch title, transcript, and insert
    print("Processing videos...")
    success_count = 0
    failed_count = 0

    for i, video_id in enumerate(video_ids, 1):
        try:
            # Get title
            title = get_video_title(video_id)

            # Get transcript
            transcript = get_transcript(video_id)

            if transcript is None:
                print(f"  [{i}/{len(video_ids)}] SKIP {video_id}: No transcript")
                failed_count += 1
                continue

            # Insert to database
            database.insert_video(video_id, title, transcript)

            # Truncate title for display
            display_title = title[:50] + "..." if len(title) > 50 else title
            print(f"  [{i}/{len(video_ids)}] ✓ {display_title}")
            success_count += 1

        except Exception as e:
            print(f"  [{i}/{len(video_ids)}] ERROR {video_id}: {e}")
            failed_count += 1

    # 5. Print summary
    print()
    print("=" * 50)
    print("Ingestion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total in DB: {database.get_video_count()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
