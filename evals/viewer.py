"""Streamlit app for reviewing LLM responses and annotating failure modes."""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.database import get_video_by_id

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TAGS_PATH = Path(__file__).parent / "tags.json"


# ============================================================================
# Experiment/Run Discovery
# ============================================================================

def get_experiment_ids() -> list[str]:
    """Discover all experiment directories (exp_XXX pattern)."""
    exp_dirs = sorted(EXPERIMENTS_DIR.glob("exp_*"))
    return [d.name for d in exp_dirs if d.is_dir()]


def get_run_ids(exp_id: str) -> list[str]:
    """Get all run IDs for an experiment."""
    runs_dir = EXPERIMENTS_DIR / exp_id / "runs"
    if not runs_dir.exists():
        return []
    return sorted([d.name for d in runs_dir.glob("r*") if d.is_dir()])


def get_responses_path(exp_id: str, run_id: str) -> Path:
    """Get path to responses.jsonl for a specific experiment run."""
    return EXPERIMENTS_DIR / exp_id / "runs" / run_id / "responses.jsonl"


def get_retrieval_path(exp_id: str, run_id: str) -> Path:
    """Get path to retrieval.jsonl for a specific experiment run."""
    return EXPERIMENTS_DIR / exp_id / "runs" / run_id / "retrieval.jsonl"


def get_csv_output_dir(exp_id: str, run_id: str) -> Path:
    """Get CSV output directory for a specific experiment run."""
    return EXPERIMENTS_DIR / exp_id / "runs" / run_id


# ============================================================================
# Core I/O Functions
# ============================================================================

def load_all_query_ids(exp_id: str, run_id: str) -> list[tuple[str, str]]:
    """
    Load all query IDs with their query text for navigation.
    Now filters to only include 'dev' split queries.

    Returns:
        List of (query_id, query_text) tuples for dev split only.
    """
    queries = []
    jsonl_path = get_responses_path(exp_id, run_id)
    retrieval_path = get_retrieval_path(exp_id, run_id)

    if not jsonl_path.exists() or not retrieval_path.exists():
        return queries

    # Build a mapping of query_id -> split from retrieval.jsonl
    query_splits = {}
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                query_splits[record["query_id"]] = record.get("split")
            except (json.JSONDecodeError, KeyError):
                continue

    # Load queries from responses.jsonl, filtering by dev split
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                query_id = record["query_id"]
                # Only include if split is 'dev'
                if query_splits.get(query_id) == "dev":
                    queries.append((query_id, record["query"]))
            except (json.JSONDecodeError, KeyError):
                continue

    return queries


def load_query_result(exp_id: str, run_id: str, query_id: str) -> dict | None:
    """
    Load a single query result by ID.

    Args:
        exp_id: Experiment ID (e.g., "exp_001")
        run_id: Run ID (e.g., "r001")
        query_id: The query ID to load.

    Returns:
        Query result dictionary or None if not found.
    """
    jsonl_path = get_responses_path(exp_id, run_id)
    
    if not jsonl_path.exists():
        return None
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record["query_id"] == query_id:
                    # Ensure optional fields exist
                    record.setdefault("notes", "")
                    record.setdefault("tags", [])

                    # Load and merge evaluation metadata
                    eval_metadata = load_eval_metadata(exp_id, run_id, query_id)
                    if eval_metadata:
                        record.update(eval_metadata)

                    # Auto-add retrieval_failure tag if gold video not retrieved
                    gold_video_id = record.get("gold_video_id")
                    hybrid_retrieved = record.get("hybrid_retrieved_ids", [])

                    if gold_video_id and gold_video_id not in hybrid_retrieved:
                        if "retrieval_failure" not in record["tags"]:
                            record["tags"].append("retrieval_failure")

                    return record
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def load_eval_metadata(exp_id: str, run_id: str, query_id: str) -> dict | None:
    """
    Load evaluation metadata for a query from retrieval.jsonl.

    Args:
        exp_id: Experiment ID
        run_id: Run ID
        query_id: The query ID to load metadata for.

    Returns:
        Dictionary with query metadata and retrieval results, or None if not found.
    """
    eval_path = get_retrieval_path(exp_id, run_id)

    if not eval_path.exists():
        return None

    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record["query_id"] == query_id:
                    return {
                        "query_type": record.get("query_type"),
                        "difficulty": record.get("difficulty"),
                        "split": record.get("split"),
                        # BM25 results
                        "bm25_retrieved_ids": record.get("bm25_retrieved_ids", []),
                        "bm25_scores": record.get("bm25_scores", []),
                        "bm25_gold_rank": record.get("bm25_gold_rank", -1),
                        # Chroma results
                        "chroma_retrieved_ids": record.get("chroma_retrieved_ids", []),
                        "chroma_scores": record.get("chroma_scores", []),
                        "chroma_gold_rank": record.get("chroma_gold_rank", -1),
                        # Hybrid results
                        "hybrid_scores": record.get("hybrid_scores", []),
                        "hybrid_gold_rank": record.get("hybrid_gold_rank", -1),
                    }
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def update_query_data(exp_id: str, run_id: str, query_id: str, notes: str, tags: list[str]) -> bool:
    """
    Update notes and tags for a query in the JSONL file.
    Uses atomic write (temp file + rename) for safety.
    
    Args:
        exp_id: Experiment ID
        run_id: Run ID
        query_id: Query ID to update.
        notes: User notes to save.
        tags: List of active tag names.
    
    Returns:
        True if update was successful, False otherwise.
    """
    jsonl_path = get_responses_path(exp_id, run_id)
    
    if not jsonl_path.exists():
        return False
    
    # Create backup before modifying
    backup_path = jsonl_path.with_suffix('.jsonl.bak')
    shutil.copy2(jsonl_path, backup_path)
    
    lines = []
    updated = False
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            
            try:
                record = json.loads(stripped)
                if record["query_id"] == query_id:
                    record["notes"] = notes
                    record["tags"] = tags
                    lines.append(json.dumps(record) + '\n')
                    updated = True
                else:
                    lines.append(line)
            except (json.JSONDecodeError, KeyError):
                lines.append(line)
    
    if not updated:
        return False
    
    # Atomic write: write to temp file, then rename
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=jsonl_path.parent,
        delete=False,
        suffix='.tmp',
        encoding='utf-8'
    ) as tmp:
        tmp.writelines(lines)
        tmp_path = Path(tmp.name)
    
    tmp_path.replace(jsonl_path)
    return True


def load_tags() -> list[str]:
    """Load available tag definitions from tags.json."""
    if not TAGS_PATH.exists():
        return []
    
    try:
        with open(TAGS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_tags(tags: list[str]) -> None:
    """Save tag definitions to tags.json."""
    TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TAGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(tags, f, indent=2)


def count_annotated_queries(exp_id: str, run_id: str) -> tuple[int, int]:
    """
    Count total queries and queries with annotations (dev split only).

    Returns:
        Tuple of (total_queries, annotated_queries).
    """
    total = 0
    annotated = 0
    jsonl_path = get_responses_path(exp_id, run_id)
    retrieval_path = get_retrieval_path(exp_id, run_id)

    if not jsonl_path.exists() or not retrieval_path.exists():
        return 0, 0

    # Build query_id -> split mapping
    query_splits = {}
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                query_splits[record["query_id"]] = record.get("split")
            except (json.JSONDecodeError, KeyError):
                continue

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                query_id = record["query_id"]
                # Only count dev split queries
                if query_splits.get(query_id) == "dev":
                    total += 1
                    # Consider annotated if has notes or tags
                    if record.get("notes", "").strip() or record.get("tags", []):
                        annotated += 1
            except (json.JSONDecodeError, KeyError):
                continue

    return total, annotated


def get_unique_videos(exp_id: str, run_id: str) -> list[str]:
    """
    Get ordered list of unique gold_video_ids from dev split queries only.

    Returns:
        List of unique video IDs in order of first appearance.
    """
    videos = []
    seen = set()
    jsonl_path = get_responses_path(exp_id, run_id)
    retrieval_path = get_retrieval_path(exp_id, run_id)

    if not jsonl_path.exists() or not retrieval_path.exists():
        return videos

    # Build query_id -> split mapping
    query_splits = {}
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                query_splits[record["query_id"]] = record.get("split")
            except (json.JSONDecodeError, KeyError):
                continue

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                query_id = record["query_id"]
                # Only include dev split queries
                if query_splits.get(query_id) == "dev":
                    video_id = record.get("gold_video_id")
                    if video_id and video_id not in seen:
                        videos.append(video_id)
                        seen.add(video_id)
            except (json.JSONDecodeError, KeyError):
                continue

    return videos


def get_first_query_for_video(exp_id: str, run_id: str, gold_video_id: str) -> str | None:
    """
    Get the first query_id for a given gold_video_id.

    Args:
        exp_id: Experiment ID
        run_id: Run ID
        gold_video_id: The video ID to search for.

    Returns:
        The first query ID for this video, or None if not found.
    """
    jsonl_path = get_responses_path(exp_id, run_id)
    
    if not jsonl_path.exists():
        return None

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("gold_video_id") == gold_video_id:
                    return record["query_id"]
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def export_to_csv(exp_id: str, run_id: str) -> dict:
    """
    Export all annotated queries to CSV with timestamp in filename (dev split only).

    Returns:
        Stats dictionary with export counts and file path.
    """
    from datetime import datetime

    jsonl_path = get_responses_path(exp_id, run_id)
    csv_output_dir = get_csv_output_dir(exp_id, run_id)
    retrieval_path = get_retrieval_path(exp_id, run_id)

    # Load all tags for column headers
    all_tags = load_tags()

    # Load all query results
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record.setdefault("notes", "")
                record.setdefault("tags", [])
                results.append(record)
            except (json.JSONDecodeError, KeyError):
                continue

    # Load split information and filter to dev only
    query_splits = {}
    if retrieval_path.exists():
        with open(retrieval_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    query_splits[record["query_id"]] = record.get("split")
                except (json.JSONDecodeError, KeyError):
                    continue

    # Filter results to dev split only
    results = [r for r in results if query_splits.get(r["query_id"]) == "dev"]

    if not results:
        return {"total": 0, "annotated": 0, "path": None}

    # Build CSV rows
    rows = []
    annotated_count = 0

    for record in results:
        has_annotation = bool(record["notes"].strip() or record["tags"])
        if has_annotation:
            annotated_count += 1

        row = {
            "query_id": record["query_id"],
            "query": record["query"],
            "response": record.get("answer", ""),
            "notes": record["notes"],
            "tags": ",".join(record["tags"]),
        }

        # Add boolean column for each defined tag
        for tag in all_tags:
            row[f"tag_{tag}"] = 1 if tag in record["tags"] else 0

        rows.append(row)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"error_analysis-dev-{timestamp}.csv"
    csv_path = csv_output_dir / csv_filename

    # Write CSV
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["query_id", "query", "response", "notes", "tags"]
    fieldnames.extend([f"tag_{tag}" for tag in all_tags])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "total": len(results),
        "annotated": annotated_count,
        "path": str(csv_path)
    }


# ============================================================================
# Streamlit App
# ============================================================================

st.set_page_config(
    page_title="Response Evaluation",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_exp_id = None
    st.session_state.current_run_id = None
    st.session_state.current_query_id = None
    st.session_state.query_list = []
    st.session_state.available_tags = load_tags()

    # Ensure retrieval_failure tag exists
    if "retrieval_failure" not in st.session_state.available_tags:
        st.session_state.available_tags.append("retrieval_failure")
        save_tags(st.session_state.available_tags)


# ============================================================================
# Sidebar - Navigation & Progress
# ============================================================================

with st.sidebar:
    st.title("Response Evaluation")
    st.caption("📊 Viewing: dev split only")

    # ========================================
    # Experiment/Run Selection
    # ========================================
    exp_ids = get_experiment_ids()
    
    if not exp_ids:
        st.warning("No experiments found in experiments/ directory.")
        st.stop()
    
    # Experiment selector
    def on_exp_change():
        new_exp = st.session_state.exp_selector
        if new_exp != st.session_state.current_exp_id:
            st.session_state.current_exp_id = new_exp
            st.session_state.current_run_id = None
            st.session_state.current_query_id = None
            st.session_state.query_list = []
    
    current_exp_idx = 0
    if st.session_state.current_exp_id in exp_ids:
        current_exp_idx = exp_ids.index(st.session_state.current_exp_id)
    else:
        st.session_state.current_exp_id = exp_ids[0]
    
    st.selectbox(
        "Experiment",
        options=exp_ids,
        index=current_exp_idx,
        key="exp_selector",
        on_change=on_exp_change
    )
    
    # Run selector
    run_ids = get_run_ids(st.session_state.current_exp_id)
    
    if not run_ids:
        st.warning(f"No runs found for {st.session_state.current_exp_id}.")
        st.stop()
    
    def on_run_change():
        new_run = st.session_state.run_selector
        if new_run != st.session_state.current_run_id:
            st.session_state.current_run_id = new_run
            st.session_state.current_query_id = None
            # Reload query list for new run
            st.session_state.query_list = load_all_query_ids(
                st.session_state.current_exp_id, 
                new_run
            )
    
    current_run_idx = 0
    if st.session_state.current_run_id in run_ids:
        current_run_idx = run_ids.index(st.session_state.current_run_id)
    else:
        st.session_state.current_run_id = run_ids[0]
        st.session_state.query_list = load_all_query_ids(
            st.session_state.current_exp_id,
            st.session_state.current_run_id
        )
    
    st.selectbox(
        "Run",
        options=run_ids,
        index=current_run_idx,
        key="run_selector",
        on_change=on_run_change
    )
    
    st.divider()
    
    # ========================================
    # Progress Stats
    # ========================================
    total_queries, annotated_queries = count_annotated_queries(
        st.session_state.current_exp_id,
        st.session_state.current_run_id
    )
    
    col1, col2 = st.columns(2)
    col1.metric("Total Queries", total_queries)
    col2.metric("Annotated", annotated_queries)
    
    if total_queries > 0:
        progress = annotated_queries / total_queries
        st.progress(progress, text=f"{progress:.0%} annotated")
    
    st.divider()
    
    # ========================================
    # Query Navigation
    # ========================================
    query_list = st.session_state.query_list
    
    if query_list:
        # Build display options
        options = [
            f"{qid}: {query[:50]}{'...' if len(query) > 50 else ''}"
            for qid, query in query_list
        ]
        query_ids = [qid for qid, _ in query_list]
        
        # Set default if not set
        if st.session_state.current_query_id is None:
            st.session_state.current_query_id = query_ids[0]
        
        # Find current selection index
        current_idx = 0
        if st.session_state.current_query_id in query_ids:
            current_idx = query_ids.index(st.session_state.current_query_id)
        
        def on_query_select():
            sel = st.session_state.query_selector
            idx = options.index(sel)
            st.session_state.current_query_id = query_ids[idx]
        
        st.selectbox(
            "Select query",
            options=options,
            index=current_idx,
            key="query_selector",
            on_change=on_query_select
        )
        
        # Navigation buttons
        st.caption(f"Query {current_idx + 1} / {len(query_ids)}")
        
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
        
        with nav_col1:
            if st.button("⏮", help="First", key="nav_first"):
                st.session_state.current_query_id = query_ids[0]
                st.rerun()
        
        with nav_col2:
            if st.button("◀", help="Previous", key="nav_prev"):
                new_idx = (current_idx - 1) % len(query_ids)
                st.session_state.current_query_id = query_ids[new_idx]
                st.rerun()
        
        with nav_col3:
            if st.button("▶", help="Next", key="nav_next"):
                new_idx = (current_idx + 1) % len(query_ids)
                st.session_state.current_query_id = query_ids[new_idx]
                st.rerun()
        
        with nav_col4:
            if st.button("⏭", help="Last", key="nav_last"):
                st.session_state.current_query_id = query_ids[-1]
                st.rerun()

        # Video navigation
        st.divider()

        current_result = load_query_result(
            st.session_state.current_exp_id,
            st.session_state.current_run_id,
            st.session_state.current_query_id
        )
        if current_result:
            gold_video_id = current_result.get("gold_video_id")
            if gold_video_id:
                unique_videos = get_unique_videos(
                    st.session_state.current_exp_id,
                    st.session_state.current_run_id
                )
                if len(unique_videos) > 1 and gold_video_id in unique_videos:
                    current_video_idx = unique_videos.index(gold_video_id)

                    st.caption(f"Video {current_video_idx + 1}/{len(unique_videos)}")

                    vid_col1, vid_col2 = st.columns(2)

                    with vid_col1:
                        if st.button("Prev Video ◀", use_container_width=True, key="nav_prev_video"):
                            new_video_idx = (current_video_idx - 1) % len(unique_videos)
                            new_video_id = unique_videos[new_video_idx]
                            first_query = get_first_query_for_video(
                                st.session_state.current_exp_id,
                                st.session_state.current_run_id,
                                new_video_id
                            )
                            if first_query:
                                st.session_state.current_query_id = first_query
                                st.rerun()

                    with vid_col2:
                        if st.button("Next Video ▶", use_container_width=True, key="nav_next_video"):
                            new_video_idx = (current_video_idx + 1) % len(unique_videos)
                            new_video_id = unique_videos[new_video_idx]
                            first_query = get_first_query_for_video(
                                st.session_state.current_exp_id,
                                st.session_state.current_run_id,
                                new_video_id
                            )
                            if first_query:
                                st.session_state.current_query_id = first_query
                                st.rerun()
    else:
        st.warning("No queries found. Run an experiment first.")

    st.divider()
    
    # Export section
    st.subheader("Export")
    if st.button("📤 Export to CSV", use_container_width=True):
        stats = export_to_csv(
            st.session_state.current_exp_id,
            st.session_state.current_run_id
        )

        if stats["total"] == 0:
            st.warning("No queries to export.")
        else:
            filepath = stats.get("path", "")
            filename = Path(filepath).name if filepath else "unknown"
            st.success(f"""
            Export complete!
            - Total: {stats['total']} queries
            - Annotated: {stats['annotated']} queries
            - File: {filename}
            """)


# ============================================================================
# Main Panel - Two Row, Three Column Layout
# ============================================================================

if st.session_state.current_query_id is None:
    st.info("Select an experiment and run from the sidebar to begin.")
else:
    query_id = st.session_state.current_query_id
    result = load_query_result(
        st.session_state.current_exp_id,
        st.session_state.current_run_id,
        query_id
    )
    
    if not result:
        st.error(f"Could not load query: {query_id}")
    else:
        # Header
        st.caption(f"Query ID: `{query_id}`")
        
        st.divider()
        
        # ========================================
        # Row 1: Query | Response | Notes
        # ========================================
        row1_query, row1_response, row1_notes = st.columns([1, 2, 1])
        
        with row1_query:
            st.subheader("Query")
            st.markdown(f"**{result['query']}**")

            # Display query metadata
            if result.get("query_type") or result.get("difficulty"):
                meta_parts = []
                if result.get("query_type"):
                    meta_parts.append(f"`{result['query_type']}`")
                if result.get("difficulty"):
                    meta_parts.append(f"`{result['difficulty']}`")
                st.caption(" • ".join(meta_parts))

            # Display dataset/split on second line
            if result.get("split"):
                st.caption(f"`{result['split']}`")
        
        with row1_response:
            st.subheader("Response")
            st.markdown(result.get("answer", "*No response generated*"))
        
        with row1_notes:
            st.subheader("Notes")
            current_notes = result.get("notes", "")
            notes_input = st.text_area(
                "Your observations",
                value=current_notes,
                height=200,
                key=f"notes_{query_id}",
                label_visibility="collapsed",
                placeholder="Add notes about the response quality..."
            )
            
            # Save button
            if st.button("💾 Save", use_container_width=True, type="primary"):
                # Collect selected tags from session state
                available_tags = st.session_state.available_tags
                current_tags = result.get("tags", [])
                selected_tags = []
                for tag in available_tags:
                    if st.session_state.get(f"tag_{query_id}_{tag}", tag in current_tags):
                        selected_tags.append(tag)
                
                success = update_query_data(
                    st.session_state.current_exp_id,
                    st.session_state.current_run_id,
                    query_id,
                    notes_input,
                    selected_tags
                )
                if success:
                    st.toast("Saved successfully!")
                else:
                    st.error("Failed to save changes.")
        
        st.divider()
        
        # ========================================
        # Row 2: Golden Video | Retrieved Videos | Failure Tags
        # ========================================
        row2_gold, row2_retrieved, row2_tags = st.columns([1, 2, 1])
        
        with row2_gold:
            st.subheader("Golden Video")
            gold_id = result.get("gold_video_id", "N/A")
            
            # Fetch golden video title from database
            gold_video = get_video_by_id(gold_id)
            if gold_video:
                gold_title = gold_video.get("title", gold_id)
            else:
                gold_title = f"(ID: {gold_id})"
            
            youtube_url = f"https://youtube.com/watch?v={gold_id}"
            st.markdown(f"[{gold_title}]({youtube_url})")
        
        with row2_retrieved:
            st.subheader("Retrieved Videos")
            citations = result.get("citations", [])
            
            if citations:
                for citation in citations:
                    video_id = citation.get("video_id", "")
                    title = citation.get("title", "Unknown")
                    youtube_url = f"https://youtube.com/watch?v={video_id}"
                    st.markdown(f"- [{title}]({youtube_url})")
            else:
                st.caption("No citations available")
            
            # Also show hybrid_retrieved_ids if different from citations
            retrieved_ids = result.get("hybrid_retrieved_ids", [])
            citation_ids = [c.get("video_id") for c in citations]
            extra_ids = [vid for vid in retrieved_ids if vid not in citation_ids]
            
            if extra_ids:
                st.caption("Other retrieved (not cited):")
                for vid in extra_ids:
                    # Try to get title from database
                    video_data = get_video_by_id(vid)
                    if video_data:
                        title = video_data.get("title", vid)
                    else:
                        title = vid
                    youtube_url = f"https://youtube.com/watch?v={vid}"
                    st.markdown(f"- [{title}]({youtube_url})")
        
        with row2_tags:
            st.subheader("Failure Tags")
            available_tags = st.session_state.available_tags
            current_tags = result.get("tags", [])

            if available_tags:
                for tag in available_tags:
                    # Check if this is an auto-added tag
                    is_auto_added = (
                        tag == "retrieval_failure" and
                        result.get("gold_video_id") not in result.get("hybrid_retrieved_ids", [])
                    )

                    label = f"{tag} (auto)" if is_auto_added else tag

                    st.checkbox(
                        label,
                        value=tag in current_tags,
                        key=f"tag_{query_id}_{tag}"
                    )
            else:
                st.caption("No tags defined. Add tags below.")
        
        # ============================================================================
        # Tag Management Section (below main content)
        # ============================================================================
        
        st.divider()
        
        with st.expander("🏷️ Manage Tags", expanded=False):
            st.caption("Define custom tags to categorize failure modes.")
            
            tag_col1, tag_col2 = st.columns([3, 1])
            
            with tag_col1:
                new_tag = st.text_input(
                    "New tag name",
                    key="new_tag_input",
                    placeholder="e.g., hallucination, off_topic, incomplete"
                )
            
            with tag_col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("➕ Add Tag", use_container_width=True):
                    if new_tag.strip():
                        clean_tag = new_tag.strip().lower().replace(" ", "_")
                        if clean_tag not in st.session_state.available_tags:
                            st.session_state.available_tags.append(clean_tag)
                            save_tags(st.session_state.available_tags)
                            st.toast(f"Added tag: {clean_tag}")
                            st.rerun()
                        else:
                            st.warning("Tag already exists.")
                    else:
                        st.warning("Enter a tag name.")
            
            # Display existing tags with delete option
            if st.session_state.available_tags:
                st.write("**Existing tags:**")
                tags_to_remove = []
                
                tag_cols = st.columns(min(len(st.session_state.available_tags), 4))
                for i, tag in enumerate(st.session_state.available_tags):
                    with tag_cols[i % 4]:
                        col_tag, col_del = st.columns([3, 1])
                        with col_tag:
                            st.code(tag)
                        with col_del:
                            if st.button("🗑", key=f"del_tag_{tag}", help=f"Delete {tag}"):
                                tags_to_remove.append(tag)
                
                # Process tag deletions
                if tags_to_remove:
                    for tag in tags_to_remove:
                        st.session_state.available_tags.remove(tag)
                    save_tags(st.session_state.available_tags)
                    st.rerun()

        # ============================================================================
        # Retrieval Results Section
        # ============================================================================

        st.divider()

        with st.expander("📊 Retrieval Results", expanded=False):
            st.caption("Comparison of retrieval results from different search methods.")

            # Create three columns for BM25, Chroma, and Hybrid
            col_bm25, col_chroma, col_hybrid = st.columns(3)

            gold_video_id = result.get("gold_video_id")

            # BM25 Column
            with col_bm25:
                st.subheader("BM25")
                bm25_ids = result.get("bm25_retrieved_ids", [])
                bm25_scores = result.get("bm25_scores", [])
                bm25_gold_rank = result.get("bm25_gold_rank", -1)

                if bm25_ids:
                    for rank, (vid, score) in enumerate(zip(bm25_ids, bm25_scores), 1):
                        video = get_video_by_id(vid)
                        title = video.get("title", vid) if video else vid

                        # Check if this is the gold video
                        is_gold = (vid == gold_video_id)
                        rank_label = f"#{rank}"

                        # Format display
                        if is_gold:
                            st.markdown(f"**{rank_label} ⭐** [{title}](https://youtube.com/watch?v={vid})")
                        else:
                            st.markdown(f"{rank_label} [{title}](https://youtube.com/watch?v={vid})")

                        st.caption(f"Score: {score:.4f}")
                else:
                    st.caption("No results retrieved")

            # Chroma Column
            with col_chroma:
                st.subheader("Chroma")
                chroma_ids = result.get("chroma_retrieved_ids", [])
                chroma_scores = result.get("chroma_scores", [])
                chroma_gold_rank = result.get("chroma_gold_rank", -1)

                if chroma_ids:
                    for rank, (vid, score) in enumerate(zip(chroma_ids, chroma_scores), 1):
                        video = get_video_by_id(vid)
                        title = video.get("title", vid) if video else vid

                        is_gold = (vid == gold_video_id)
                        rank_label = f"#{rank}"

                        if is_gold:
                            st.markdown(f"**{rank_label} ⭐** [{title}](https://youtube.com/watch?v={vid})")
                        else:
                            st.markdown(f"{rank_label} [{title}](https://youtube.com/watch?v={vid})")

                        st.caption(f"Distance: {score:.4f}")
                else:
                    st.caption("No results retrieved")

            # Hybrid Column
            with col_hybrid:
                st.subheader("Hybrid (RRF)")
                hybrid_ids = result.get("hybrid_retrieved_ids", [])
                hybrid_scores = result.get("hybrid_scores", [])
                hybrid_gold_rank = result.get("hybrid_gold_rank", -1)

                if hybrid_ids:
                    for rank, (vid, score) in enumerate(zip(hybrid_ids, hybrid_scores), 1):
                        video = get_video_by_id(vid)
                        title = video.get("title", vid) if video else vid

                        is_gold = (vid == gold_video_id)
                        rank_label = f"#{rank}"

                        if is_gold:
                            st.markdown(f"**{rank_label} ⭐** [{title}](https://youtube.com/watch?v={vid})")
                        else:
                            st.markdown(f"{rank_label} [{title}](https://youtube.com/watch?v={vid})")

                        st.caption(f"RRF Score: {score:.4f}")
                else:
                    st.caption("No results retrieved")
