"""Context formatting utilities for answer generation."""


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

