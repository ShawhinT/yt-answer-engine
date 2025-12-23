"""Answer generation using retrieved video context."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

from ingestion.database import get_video_by_id
from utils import search_hybrid, search_bm25, search_chroma

# Load environment variables
load_dotenv()

# Path to prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ============================================================================
# Pydantic Models
# ============================================================================

class Citation(BaseModel):
    """A citation to a source video."""
    video_id: str = Field(description="YouTube video ID")
    title: str = Field(description="Title of the video")


class AnswerResponse(BaseModel):
    """Generated answer with citations."""
    answer: str = Field(description="The answer to the user's question")
    citations: list[Citation] = Field(description="List of videos cited in the answer")


# ============================================================================
# Prompt Loading
# ============================================================================

def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory.
    
    Args:
        name: Prompt name (e.g., "system" loads "answer_system.md")
    
    Returns:
        The prompt content as a string
    """
    prompt_path = PROMPTS_DIR / f"answer_{name}.md"
    return prompt_path.read_text()


# ============================================================================
# Context Formatting
# ============================================================================

def format_context(videos: list[dict[str, str]]) -> str:
    """Format video data as context string for the prompt.
    
    Args:
        videos: List of video dictionaries with video_id, title, transcript
    
    Returns:
        Formatted context string
    """
    parts = []
    for video in videos:
        part = f"### {video['title']} (ID: {video['video_id']})\n\n{video['transcript']}"
        parts.append(part)
    return "\n\n---\n\n".join(parts)


# ============================================================================
# Answer Generation
# ============================================================================

def generate_answer(
    query: str,
    limit: int = 3,
    search_method: str = "hybrid"
) -> AnswerResponse:
    """Generate an answer to a query using retrieved video context.
    
    Args:
        query: The user's question
        limit: Number of videos to retrieve for context (default: 3)
        search_method: Search method to use - "hybrid", "bm25", or "chroma" (default: "hybrid")
    
    Returns:
        AnswerResponse containing the answer and citations
    """
    # Select search function based on method
    search_functions = {
        "hybrid": search_hybrid.search_with_scores,
        "bm25": search_bm25.search_with_scores,
        "chroma": search_chroma.search_with_scores,
    }
    
    if search_method not in search_functions:
        raise ValueError(f"Unknown search method: {search_method}. Use 'hybrid', 'bm25', or 'chroma'.")
    
    search_fn = search_functions[search_method]
    
    # Retrieve relevant videos
    results = search_fn(query, limit=limit)
    video_ids = [video_id for video_id, _ in results]
    
    # Fetch full video data
    videos = []
    for video_id in video_ids:
        video = get_video_by_id(video_id)
        if video:
            videos.append(video)
    
    if not videos:
        return AnswerResponse(
            answer="I couldn't find any relevant videos to answer your question.",
            citations=[]
        )
    
    # Load prompts
    system_prompt = load_prompt("system")
    user_template = load_prompt("user")
    
    # Format context and user message
    context = format_context(videos)
    user_message = user_template.format(query=query, context=context)
    
    # Initialize OpenAI client and generate answer
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.responses.parse(
        model="gpt-4.1-2025-04-14",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        text_format=AnswerResponse,
    )
    
    return response.output_parsed

