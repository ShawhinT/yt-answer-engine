"""Shared Pydantic models for answer generation."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation to a source video."""

    video_id: str = Field(description="YouTube video ID")
    title: str = Field(description="Title of the video")


class AnswerResponse(BaseModel):
    """Generated answer with citations."""

    answer: str = Field(description="The answer to the user's question")
    citations: list[Citation] = Field(description="List of videos cited in the answer")

