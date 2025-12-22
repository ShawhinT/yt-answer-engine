"""Main video ingestion script."""

import os
import re
import time
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.proxies import GenericProxyConfig
from . import database

# Load environment variables
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = "UCa9gErQ9AE5jT2DZLjXBIdA"


def parse_duration_to_seconds(duration: str) -> int:
    """Parse ISO 8601 duration (e.g., PT1M30S) to seconds."""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def get_video_durations(video_ids: list[str]) -> dict[str, int]:
    """Get durations for multiple videos. Returns dict of video_id -> seconds."""
    durations = {}
    url = "https://www.googleapis.com/youtube/v3/videos"

    # Process in batches of 50 (API limit)
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        params = {
            "part": "contentDetails",
            "id": ",".join(batch),
            "key": API_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        for item in response.json().get("items", []):
            duration_str = item["contentDetails"]["duration"]
            durations[item["id"]] = parse_duration_to_seconds(duration_str)

    return durations


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
            "key": API_KEY
        }

        # Only add pageToken if it exists (not None)
        if next_page_token:
            params["pageToken"] = next_page_token

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


def get_transcript(video_id: str, max_retries: int = 3) -> str | None:
    """Get transcript for video using proxy with retry logic.
    
    Args:
        video_id: YouTube video ID
        max_retries: Maximum number of retry attempts (default 3)
    
    Returns:
        Transcript text or None if unavailable
    """
    proxy_username = os.getenv("PROXY_USERNAME")
    proxy_password = os.getenv("PROXY_PASSWORD")
    proxy_url_base = os.getenv("PROXY_URL")

    if not (proxy_username and proxy_password and proxy_url_base):
        print(f"    Proxy credentials not available")
        return None

    # Construct proxy URLs with credentials
    http_proxy_url = f"http://{proxy_username}:{proxy_password}@{proxy_url_base}"
    https_proxy_url = f"https://{proxy_username}:{proxy_password}@{proxy_url_base}"

    proxy_config = GenericProxyConfig(
        http_url=http_proxy_url,
        https_url=https_proxy_url
    )

    ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)

    for attempt in range(max_retries):
        try:
            fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
            full_transcript = " ".join([entry.text for entry in fetched_transcript])
            return full_transcript

        except (TranscriptsDisabled, NoTranscriptFound):
            # No transcript available - don't retry
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Attempt {attempt + 1} failed, retrying in 1.5s...")
                time.sleep(1.5)
            else:
                print(f"    Error fetching transcript after {max_retries} attempts: {e}")
                return None

    return None


def main():
    """Main ingestion pipeline."""
    print("Starting ingestion...")

    # 1. Initialize database
    database.init_db()
    print()

    # 2. Use channel ID directly
    print(f"Using channel ID: {CHANNEL_ID}")
    print()

    # 3. Get all video IDs
    print("Fetching all video IDs...")
    video_ids = get_all_video_ids(CHANNEL_ID)
    print(f"Found {len(video_ids)} videos")
    print()

    # 4. Filter out videos shorter than 45 seconds
    print("Filtering short videos...")
    durations = get_video_durations(video_ids)
    video_ids = [vid for vid in video_ids if durations.get(vid, 0) >= 45]
    print(f"Kept {len(video_ids)} videos (>= 45 sec)")
    print()

    # 5. For each video: fetch title, transcript, and insert
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

        # Add delay between videos to avoid rate limiting
        time.sleep(1.05)

    # 6. Print summary
    print()
    print("=" * 50)
    print("Ingestion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total in DB: {database.get_video_count()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
