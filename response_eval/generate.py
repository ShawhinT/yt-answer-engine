"""Generate answers using pre-retrieved video context from retrieval evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.database import get_video_by_id

# Load environment variables
load_dotenv()

# Paths
EVAL_RESULTS_PATH = Path("retrieval_eval/data/eval_results.jsonl")
OUTPUT_PATH = Path("response_eval/data/response_results.jsonl")
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
# Data Loading
# ============================================================================

def load_eval_results() -> list[dict]:
    """Load evaluation results from JSONL file."""
    results = []
    with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


# ============================================================================
# Answer Generation
# ============================================================================

def generate_answer_from_ids(
    query: str,
    video_ids: list[str],
    client: OpenAI
) -> AnswerResponse:
    """Generate an answer using pre-retrieved video IDs.
    
    Args:
        query: The user's question
        video_ids: List of video IDs to use as context
        client: OpenAI client instance
    
    Returns:
        AnswerResponse containing the answer and citations
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
            citations=[]
        )
    
    # Load prompts
    system_prompt = load_prompt("system")
    user_template = load_prompt("user")
    
    # Format context and user message
    context = format_context(videos)
    user_message = user_template.format(query=query, context=context)
    
    # Generate answer
    response = client.responses.parse(
        model="gpt-4.1-2025-04-14",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        text_format=AnswerResponse,
    )
    
    return response.output_parsed


# ============================================================================
# Main Generation Pipeline
# ============================================================================

def generate_responses(limit: int | None = None):
    """Generate answers for all queries in eval results.
    
    Args:
        limit: Optional limit on number of queries to process
    """
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Load eval results
    eval_results = load_eval_results()
    print(f"Loaded {len(eval_results)} queries from {EVAL_RESULTS_PATH}")
    
    if limit:
        eval_results = eval_results[:limit]
        print(f"Processing first {limit} queries")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Process each query
    results = []
    for i, eval_result in enumerate(eval_results, 1):
        query_id = eval_result["query_id"]
        query = eval_result["query"]
        gold_video_id = eval_result["gold_video_id"]
        hybrid_retrieved_ids = eval_result["hybrid_retrieved_ids"]
        
        print(f"[{i}/{len(eval_results)}] Processing {query_id}: {query[:50]}...")
        
        # Generate answer
        answer_response = generate_answer_from_ids(query, hybrid_retrieved_ids, client)
        
        # Build result
        result = {
            "query_id": query_id,
            "query": query,
            "gold_video_id": gold_video_id,
            "hybrid_retrieved_ids": hybrid_retrieved_ids,
            "answer": answer_response.answer,
            "citations": [c.model_dump() for c in answer_response.citations]
        }
        results.append(result)
    
    # Write results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nGenerated {len(results)} answers")
    print(f"Results saved to: {OUTPUT_PATH}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate answers using pre-retrieved video context"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process (useful for testing)"
    )
    
    args = parser.parse_args()
    generate_responses(limit=args.limit)


if __name__ == "__main__":
    main()

