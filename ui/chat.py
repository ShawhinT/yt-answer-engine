"""
Minimal Streamlit chat UI for real-time answer generation with failure labeling.

This app provides a chat interface where users can:
- Ask questions and get answers with citations
- Have multi-turn conversations with the answer engine
- Label entire conversation traces with notes and failure checkboxes
- Save conversation traces to JSONL for later analysis
"""

# ============================================================================
# Imports & Configuration
# ============================================================================

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.answer import generate_answer, Citation
from data_ingestion.database import get_video_by_id

# Load environment variables
load_dotenv()


# ============================================================================
# Constants & Paths
# ============================================================================

JSONL_PATH = Path(__file__).parent / "data" / "chat_traces.jsonl"


# ============================================================================
# JSONL I/O Operations
# ============================================================================

def save_to_jsonl(
    trace_id: str,
    conversation: list[dict],
    timestamp: str,
    notes: str,
    retrieval_failure: bool,
    response_failure: bool,
) -> bool:
    """Save entire conversation trace to JSONL file (append-only).

    Args:
        trace_id: Unique identifier for this conversation trace
        conversation: List of message dicts from st.session_state.messages
        timestamp: ISO timestamp when trace is saved
        notes: User notes about performance
        retrieval_failure: Whether retrieval failed
        response_failure: Whether response quality was poor

    Returns:
        True if save successful, False otherwise
    """
    try:
        # Create directory if needed
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Build conversation array with serializable citations
        serialized_conversation = []
        for msg in conversation:
            serialized_msg = msg.copy()
            # Convert Citation objects to dicts for JSON serialization
            if "citations" in serialized_msg and serialized_msg["citations"]:
                serialized_msg["citations"] = [
                    {"video_id": c.video_id, "title": c.title}
                    for c in serialized_msg["citations"]
                ]
            serialized_conversation.append(serialized_msg)

        # Build complete trace record
        record = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "conversation": serialized_conversation,
            "notes": notes,
            "retrieval_failure": retrieval_failure,
            "response_failure": response_failure,
        }

        # Append to file
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return True
    except Exception as e:
        st.error(f"Failed to save trace: {str(e)}")
        return False


# ============================================================================
# Message Handlers
# ============================================================================

def handle_user_input(user_query: str):
    """Process user query and generate answer.

    Args:
        user_query: The user's question
    """
    # 1. Add user message to session state
    timestamp = datetime.now().isoformat()
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "timestamp": timestamp,
    })

    # 2. Generate answer
    search_method = st.session_state.search_method
    num_results = st.session_state.num_results

    try:
        with st.spinner("Thinking..."):
            answer_response = generate_answer(
                query=user_query,
                limit=num_results,
                search_method=search_method,
            )
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        # Remove the user message since we couldn't generate a response
        st.session_state.messages.pop()
        return

    # 3. Generate unique query_id
    query_id = f"chat_{timestamp}_{uuid4().hex[:8]}"

    # 4. Add assistant message to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_response.answer,
        "query_id": query_id,
        "timestamp": timestamp,
        "citations": answer_response.citations,
        "retrieved_ids": [c.video_id for c in answer_response.citations],
        "search_method": search_method,
        "num_results": num_results,
    })

    # 5. Mark trace as not saved (new messages added to conversation)
    st.session_state.trace_labels["saved"] = False


def save_trace():
    """Save the entire conversation trace to JSONL."""
    if not st.session_state.messages:
        st.warning("No conversation to save")
        return

    # Get notes and labels from sidebar inputs
    notes = st.session_state.get("notes_input", "")
    retrieval_failure = st.session_state.get("retrieval_failure_checkbox", False)
    response_failure = st.session_state.get("response_failure_checkbox", False)

    # Generate trace_id and timestamp
    timestamp = datetime.now().isoformat()
    trace_id = f"trace_{timestamp}_{uuid4().hex[:8]}"

    # Save entire conversation to JSONL
    success = save_to_jsonl(
        trace_id=trace_id,
        conversation=st.session_state.messages,
        timestamp=timestamp,
        notes=notes,
        retrieval_failure=retrieval_failure,
        response_failure=response_failure,
    )

    if success:
        # Mark as saved and update labels
        st.session_state.trace_labels["saved"] = True
        st.session_state.trace_labels["notes"] = notes
        st.session_state.trace_labels["retrieval_failure"] = retrieval_failure
        st.session_state.trace_labels["response_failure"] = response_failure
        st.toast("Conversation trace saved successfully!")


# ============================================================================
# Display Components
# ============================================================================

def format_citations(citations: list[Citation]) -> str:
    """Format citations as markdown numbered list with YouTube links.

    Args:
        citations: List of Citation objects

    Returns:
        Formatted markdown string
    """
    if not citations:
        return "*No citations available*"

    citation_lines = []
    for i, citation in enumerate(citations, 1):
        youtube_url = f"https://youtube.com/watch?v={citation.video_id}"
        citation_lines.append(f"{i}. [{citation.title}]({youtube_url})")

    return "\n".join(citation_lines)


def display_retrieved_context():
    """Display retrieved videos for the last query in sidebar."""
    if not st.session_state.messages:
        st.caption("No queries yet")
        return

    # Find last assistant message
    last_msg = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            last_msg = msg
            break

    if not last_msg:
        st.caption("No answer yet")
        return

    retrieved_ids = last_msg.get("retrieved_ids", [])

    if not retrieved_ids:
        st.caption("No videos retrieved")
        return

    for i, video_id in enumerate(retrieved_ids, 1):
        video = get_video_by_id(video_id)
        if video:
            title = video["title"]
            url = f"https://youtube.com/watch?v={video_id}"
            st.markdown(f"{i}. [{title}]({url})")
        else:
            st.caption(f"{i}. ID: {video_id} (not found)")


# ============================================================================
# Main App Layout
# ============================================================================

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Answer Engine Chat",
        page_icon="💬",
        layout="wide",
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.trace_labels = {
            "notes": "",
            "retrieval_failure": False,
            "response_failure": False,
            "saved": True,  # No conversation yet, so nothing to save
        }
        st.session_state.search_method = "hybrid"
        st.session_state.num_results = 3

    # Sidebar
    with st.sidebar:
        st.title("Settings")

        # Settings section
        st.subheader("Search Configuration")

        search_method = st.selectbox(
            "Search Method",
            options=["hybrid", "bm25", "chroma"],
            index=0,  # hybrid is default
            key="search_method_select",
            help="Method used to retrieve relevant videos",
        )
        st.session_state.search_method = search_method

        num_results = st.slider(
            "Number of Results",
            min_value=1,
            max_value=5,
            value=3,
            key="num_results_slider",
            help="Number of videos to retrieve for context",
        )
        st.session_state.num_results = num_results

        st.divider()

        # Label full trace section
        st.subheader("Label Trace")

        if st.session_state.messages:
            # Text area for notes
            notes = st.text_area(
                "Performance Notes",
                value=st.session_state.trace_labels.get("notes", ""),
                height=150,
                key="notes_input",
                placeholder="Add notes about the conversation quality, accuracy, or other observations...",
            )

            # Checkboxes for failure types
            retrieval_failure = st.checkbox(
                "Retrieval Failure",
                value=st.session_state.trace_labels.get("retrieval_failure", False),
                key="retrieval_failure_checkbox",
                help="Check if relevant videos were not retrieved in this conversation",
            )

            response_failure = st.checkbox(
                "Response Quality Issue",
                value=st.session_state.trace_labels.get("response_failure", False),
                key="response_failure_checkbox",
                help="Check if the answer quality was poor in this conversation",
            )

            # Save button
            saved = st.session_state.trace_labels.get("saved", False)
            save_button = st.button(
                "✓ Saved" if saved else "Save",
                key="save_button",
                disabled=saved,
                use_container_width=True,
                type="primary" if not saved else "secondary",
            )

            if save_button:
                save_trace()
        else:
            st.info("Start a conversation to label the trace")

        st.divider()

        # Retrieved context section
        st.subheader("Retrieved Context")
        st.caption("Videos used for the last answer:")
        display_retrieved_context()

    # Main chat area
    st.title("YouTube Answer Engine")
    st.caption("Ask questions and get answers from video transcripts")

    # Display conversation history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

            # For assistant messages, show citations
            if role == "assistant" and "citations" in message:
                citations = message["citations"]
                if citations:
                    st.markdown("**Citations:**")
                    st.markdown(format_citations(citations))

    # Chat input
    user_input = st.chat_input("Ask a question...")

    if user_input:
        handle_user_input(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
