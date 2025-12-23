"""Functions for mining and generating queries from YouTube videos."""

import os
import time
import json
from pathlib import Path
from enum import Enum
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

# Load environment variables
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryType(str, Enum):
    """Category of query."""
    EXACT = "exact"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"


class DifficultyLevel(str, Enum):
    """Difficulty level of query."""
    GROUNDED = "grounded"
    MEDIUM = "medium"
    HARD = "hard"


class Query(BaseModel):
    """A single generated query."""
    query_type: QueryType = Field(description="The category: exact, conceptual, or procedural")
    difficulty: DifficultyLevel = Field(description="The difficulty: grounded, medium, or hard")
    grounding: str = Field(description="Section label + verbatim evidence quote (≤12 words)")
    query: str = Field(description="The natural language search query")


class QueryResponse(BaseModel):
    """Collection of generated queries (always 9: 3 types × 3 difficulties)."""
    queries: list[Query] = Field(description="List of exactly 9 generated queries")


# ============================================================================
# YouTube Comment Fetching
# ============================================================================

def get_all_comments(video_id: str) -> list[dict[str, str]]:
    """
    Get all top-level comments from a YouTube video.

    Args:
        video_id: YouTube video ID

    Returns:
        List of comment dictionaries.
        Each dict contains: comment_id, text, author, published_at, like_count
    """
    comments = []
    next_page_token = None
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    while True:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "order": "relevance",
            "key": API_KEY
        }

        # Only add pageToken if it exists (not None)
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract comments from this page
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "text": snippet["textDisplay"],
                "author": snippet["authorDisplayName"],
                "published_at": snippet["publishedAt"],
                "like_count": str(snippet["likeCount"])
            })

        # Check for next page
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        # Rate limiting between pagination requests
        time.sleep(1.05)

    return comments


# ============================================================================
# Prompt Loading
# ============================================================================

def load_system_prompt() -> str:
    """Load the system prompt from prompt.md."""
    prompt_path = Path(__file__).parent / "prompt.md"
    return prompt_path.read_text()


# ============================================================================
# Query Generation via OpenAI
# ============================================================================

def generate_queries(
    video_title: str,
    transcript: str,
    comments: list[dict[str, str]]
) -> QueryResponse:
    """
    Generate natural language queries from video content.

    Args:
        video_title: The title of the YouTube video
        transcript: The full transcript text
        comments: List of comment dictionaries from YouTube

    Returns:
        QueryResponse containing exactly 9 generated queries (3 types × 3 difficulties)
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load system prompt
    system_prompt = load_system_prompt()

    # Format comments for prompt
    if comments:
        # Take top 50 most relevant comments (already sorted by relevance)
        top_comments = comments[:50]
        comments_text = "\n".join([
            f"- {comment['text']}"
            for comment in top_comments
        ])
    else:
        comments_text = "(No comments available)"

    # Construct user message with all three inputs (one line)
    user_message = f"Video Title: {video_title}\n\nTranscript:\n{transcript}\n\nReal YouTube Comments:\n{comments_text}"

    # Call OpenAI API
    response = client.responses.parse(
        model="gpt-4.1-2025-04-14",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        text_format=QueryResponse,
    )

    return response.output_parsed


# ============================================================================
# Query Persistence
# ============================================================================

def save_queries_to_jsonl(
    video_id: str,
    video_title: str,
    query_response: QueryResponse,
    output_path: Path
):
    """
    Append generated queries to JSONL file.

    Args:
        video_id: YouTube video ID
        video_title: Video title for reference
        query_response: QueryResponse object from generate_queries
        output_path: Path to JSONL output file
    """
    # Create data directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append each query as a JSON line
    with open(output_path, 'a') as f:
        for query in query_response.queries:
            record = {
                "video_id": video_id,
                "video_title": video_title,
                "query": query.query,
                "query_type": query.query_type.value,
                "difficulty": query.difficulty.value,
                "grounding": query.grounding,
                "source": "synthetic"
            }
            f.write(json.dumps(record) + '\n')
