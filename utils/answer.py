"""Answer generation utilities."""

import sys
from pathlib import Path

from openai import OpenAI

# Add project root to path for data_ingestion access
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.database import get_video_by_id
from utils.data import load_prompt
from utils.models import AnswerResponse


def format_context(videos: list[dict[str, str]]) -> str:
    """Format video data as context string for the prompt.

    Args:
        videos: List of video dicts with 'title', 'video_id', and 'transcript' keys

    Returns:
        Formatted context string with videos separated by horizontal rules
    """
    parts = []
    for video in videos:
        part = f"### {video['title']} (ID: {video['video_id']})\n\n{video['transcript']}"
        parts.append(part)
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    video_ids: list[str],
    client: OpenAI,
    prompts_dir: Path | str,
    model: str = "gpt-4.1-2025-04-14",
) -> AnswerResponse:
    """Generate an answer using pre-retrieved video IDs.

    Args:
        query: User's question
        video_ids: List of video IDs to use as context
        client: OpenAI client instance
        prompts_dir: Path to directory containing prompt templates
        model: OpenAI model to use for answer generation

    Returns:
        AnswerResponse with answer text and citations
    """
    # Fetch video data for each ID
    videos = []
    for video_id in video_ids:
        video = get_video_by_id(video_id)
        if video:
            videos.append(video)

    if not videos:
        return AnswerResponse(
            answer="I couldn't find any relevant videos to answer your question.",
            citations=[],
        )

    # Load prompts
    system_prompt = load_prompt(prompts_dir, "system")
    user_template = load_prompt(prompts_dir, "user")

    # Format context and user message
    context = format_context(videos)
    user_message = user_template.format(query=query, context=context)

    # Generate answer
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        text_format=AnswerResponse,
    )

    return response.output_parsed

